import os
import time
import numpy as np
import pandas as pd
import rasterio
from scipy.stats import linregress

R = 6_371_000
CHLOR_DIR = "data/chlor_oc_cci"
EMISSION_DIR = "data/emission"
OUT_ROOT = "."
def out_dir(year): return os.path.join(OUT_ROOT, f"surrogate_102_{year}_oc_cci")

YEARS = ["2013", "2014", "2015", "2016", "2017"]
WINDOW_SIZE = 11
MIN_VALID = 70
R2_SCREEN = 0.6
SLOPE_ABS_MIN = 0.01
SAVE_ALL_WINDOWS_FOR = {"baseline"}

ZONAL_DEG = [d * s for d in range(10, 65, 5) for s in (1, -1)]
MERID_DEG = [d * s for d in (10, 15, 20) for s in (1, -1)]
DIAG_LON = [d * s for d in (10, 15, 20, 30, 40, 50) for s in (1, -1)]
DIAG_LAT = [d * s for d in (10, 15, 20) for s in (1, -1)]

COLS = ["year", "scenario", "dlon", "dlat", "mode", "shift_dist_deg",
        "n_regressed", "n_screened", "fpr", "n_pos", "n_neg",
        "frac_pos", "pos_area_km2"]


def pixel_area_m2(lat_deg, xres_deg, yres_deg):
    """Surface area of one grid cell at a latitude."""
    lat = np.deg2rad(lat_deg); dlat = np.deg2rad(yres_deg); dlon = np.deg2rad(xres_deg)
    return R**2 * dlon * (np.sin(lat + dlat/2) - np.sin(lat - dlat/2))


def load_pair(chlor_raster_path, emission_raster_path):
    """Read both rasters on one shared grid."""
    with rasterio.open(chlor_raster_path) as src_c, rasterio.open(emission_raster_path) as src_e:
        assert src_c.transform == src_e.transform and src_c.shape == src_e.shape, "Resolution or CRS mismatch"
        return src_c.read(1), src_e.read(1), src_c.transform


def perform_moving_window_regression(chlor_raster_path, emission_raster_path, window_size=WINDOW_SIZE):
    """Regress log1p chlorophyll on log1p emission in every window; drop windows with fewer than MIN_VALID cells,
    constant values or an absolute slope below SLOPE_ABS_MIN."""
    chl, em, tf = load_pair(chlor_raster_path, emission_raster_path)
    rows, cols = chl.shape
    half = window_size // 2
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        keep = np.isfinite(chl) & (chl > 0) & np.isfinite(em) & (em > 0)
        lc, le = np.log1p(chl), np.log1p(em)
    results = []
    for i in range(rows):
        r0, r1 = max(i - half, 0), min(i + half + 1, rows)
        for j in range(cols):
            c0, c1 = max(j - half, 0), min(j + half + 1, cols)
            k = keep[r0:r1, c0:c1].ravel()
            n = int(k.sum())
            if n < MIN_VALID:
                continue
            a_vals = lc[r0:r1, c0:c1].ravel()[k]
            b_vals = le[r0:r1, c0:c1].ravel()[k]
            if np.all(a_vals == a_vals[0]):
                continue
            if np.all(b_vals == b_vals[0]):
                continue
            slope, intercept, r_value, p_value, _ = linregress(b_vals, a_vals)
            if abs(slope) < SLOPE_ABS_MIN:
                continue
            lon, lat = tf * (j + 0.5, i + 0.5)
            results.append({"Longitude": lon, "Latitude": lat,
                            "Intercept": intercept, "Slope": slope,
                            "R": r_value, "R2": r_value**2,
                            "P_Value": p_value, "Count": n})
    return pd.DataFrame(results)


def write_surrogate(src_path, out_path, mode, dlon_px=0, dlat_px=0):
    """Write one shifted or rotated emission raster; a positive latitude shift moves the field north."""
    with rasterio.open(src_path) as src:
        arr = src.read(1); profile = src.profile.copy()
    if mode == "none":
        pert = arr
    elif mode == "rot180":
        pert = np.rot90(arr, 2)
    elif mode == "shift":
        row_px = -dlat_px if profile["transform"].e < 0 else dlat_px
        pert = np.roll(arr, (row_px, dlon_px), axis=(0, 1))
    else:
        raise ValueError(mode)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(pert.astype(profile["dtype"]), 1)
    return out_path


def scenarios():
    """The real field and the 101 surrogate fields: (tag, dlon, dlat, mode)."""
    scen = [("baseline", 0, 0, "none")]
    for d in ZONAL_DEG:
        scen.append((f"lon{d:+d}", d, 0, "shift"))
    for d in MERID_DEG:
        scen.append((f"lat{d:+d}", 0, d, "shift"))
    for dl in DIAG_LON:
        for dt in DIAG_LAT:
            scen.append((f"lon{dl:+d}_lat{dt:+d}", dl, dt, "shift"))
    scen.append(("rot180", 0, 0, "rot180"))
    return scen


def done_rows(csv_path, key_cols):
    """Scenario keys already present in a results CSV."""
    if not os.path.exists(csv_path):
        return set()
    prev = pd.read_csv(csv_path)
    return set(tuple(str(v) for v in row) for row in prev[key_cols].itertuples(index=False))


def raster_path(family_dir, year, tag):
    return os.path.join(family_dir, "rasters", f"NO_{year}_{tag}.tif")


def ensure_raster(emission_path, out_path, mode, dlon, dlat, px_per_deg):
    """Write the surrogate raster if it is absent."""
    if os.path.exists(out_path):
        return out_path
    if mode == "none":
        return emission_path
    return write_surrogate(emission_path, out_path, mode,
                           dlon_px=int(round(dlon * px_per_deg)),
                           dlat_px=int(round(dlat * px_per_deg)))


def summarise(df, scr, xres, yres):
    """Score one screened window set."""
    pos = scr[scr["Slope"] >= 0]; neg = scr[scr["Slope"] < 0]
    pos_area = float(pixel_area_m2(pos["Latitude"].to_numpy(), xres, yres).sum() / 1e6) if len(pos) else 0.0
    return {"n_regressed": len(df), "n_screened": len(scr),
            "fpr": len(scr) / len(df) if len(df) else np.nan,
            "n_pos": len(pos), "n_neg": len(neg),
            "frac_pos": len(pos) / len(scr) if len(scr) else np.nan,
            "pos_area_km2": pos_area}


def save_windows(year, tag, df, scr):
    """Write the window tables of one scenario."""
    wdir = os.path.join(out_dir(year), "windows")
    os.makedirs(wdir, exist_ok=True)
    scr.to_csv(os.path.join(wdir, f"windows_{tag}.csv"), index=False)
    if SAVE_ALL_WINDOWS_FOR == "all" or (SAVE_ALL_WINDOWS_FOR and tag in SAVE_ALL_WINDOWS_FOR):
        df.to_csv(os.path.join(wdir, f"windows_{tag}_all.csv"), index=False)


def main():
    """Build the rasters, run every field for every year, write the results and print a summary per year."""
    for year in YEARS:
        fdir = out_dir(year)
        for d in (os.path.join(fdir, "rasters"), os.path.join(fdir, "windows")):
            os.makedirs(d, exist_ok=True)
        csv_out = os.path.join(fdir, f"results_{year}.csv")
        done = set(k[1] for k in done_rows(csv_out, ["year", "scenario"]))
        chlor_path = os.path.join(CHLOR_DIR, f"chlor_{year}.tif")
        emission_path = os.path.join(EMISSION_DIR, f"NO_{year}_yearly.tif")
        if not (os.path.exists(chlor_path) and os.path.exists(emission_path)):
            print(f"[skip] {year}: input missing"); continue
        with rasterio.open(chlor_path) as src:
            xres, yres = src.res
        px_per_deg = 1.0 / xres
        scen = scenarios()
        todo = [sc for sc in scen if sc[0] not in done]
        print(f"[{year} plan] {len(scen)} scenarios total; {len(todo)} to run ({len(done)} done)")
        for k, (tag, dlon, dlat, mode) in enumerate(todo, 1):
            epath = ensure_raster(emission_path, raster_path(fdir, year, tag), mode, dlon, dlat, px_per_deg)
            print(f"[{year} {k}/{len(todo)}] {tag}")
            t0 = time.time()
            print(f"    regressing ...", flush=True)
            df = perform_moving_window_regression(chlor_path, epath, window_size=WINDOW_SIZE)
            print(f"    {len(df):,} windows entering the screen, {time.time() - t0:.0f} s")
            scr = df[df["R2"] >= R2_SCREEN]
            save_windows(year, tag, df, scr)
            row = {"year": year, "scenario": tag, "dlon": dlon, "dlat": dlat, "mode": mode,
                   "shift_dist_deg": float(np.hypot(dlon, dlat)), **summarise(df, scr, xres, yres)}
            pd.DataFrame([row])[COLS].to_csv(csv_out, mode="a", index=False, header=not os.path.exists(csv_out))
            print(f"    screened={row['n_screened']:>6d} fpr={row['fpr']:.4f} area={row['pos_area_km2']:.3e}")

    for year in YEARS:
        csv_out = os.path.join(out_dir(year), f"results_{year}.csv")
        if not os.path.exists(csv_out):
            continue
        res = pd.read_csv(csv_out).drop_duplicates("scenario", keep="last")
        base = res[res.scenario == "baseline"].iloc[0]; sur = res[res.scenario != "baseline"]
        rank_f = int((sur["fpr"] >= base["fpr"]).sum()) + 1
        rank_a = int((sur["pos_area_km2"] >= base["pos_area_km2"]).sum()) + 1
        n = len(sur) + 1
        print(f"\n================ {n}-FIELD SUMMARY {year} ================")
        print(f"baseline: entering {int(base['n_regressed']):,}, screened {int(base['n_screened']):,}, "
              f"positive {int(base['n_pos']):,}, negative {int(base['n_neg']):,}")
        print(f"baseline fpr      = {base['fpr']:.4f}  rank {rank_f}/{n} (empirical p = {rank_f/n:.4f})")
        print(f"baseline pos_area = {base['pos_area_km2'] / 1e6:.4f}e6 km2  rank {rank_a}/{n} (empirical p = {rank_a/n:.4f})")
        print(f"surrogate fpr: mean {sur['fpr'].mean():.4f}, max {sur['fpr'].max():.4f} "
              f"({sur.loc[sur['fpr'].idxmax(), 'scenario']})")
        print(f"surrogate pos_area: mean {sur['pos_area_km2'].mean() / 1e6:.2f}e6 km2, "
              f"max {sur['pos_area_km2'].max() / 1e6:.2f}e6 km2 ({sur.loc[sur['pos_area_km2'].idxmax(), 'scenario']})")
        for col, name, scale in (("fpr", "fpr", 1.0), ("pos_area_km2", "pos_area (1e6 km2)", 1e-6)):
            top = sur.sort_values(col, ascending=False).head(3)
            lead = base[col] - top.iloc[0][col]
            print(f"top 3 surrogates on {name}: " +
                  "; ".join(f"{r['scenario']} {r[col] * scale:.4f}" for _, r in top.iterrows()) +
                  f" | baseline lead over the best {lead * scale:+.4f} ({100 * lead / top.iloc[0][col]:+.1f}%)")
        print(f"[Saved] {csv_out}  (+ per-scenario windows in {os.path.join(out_dir(year), 'windows')}/)")


if __name__ == "__main__":
    main()
