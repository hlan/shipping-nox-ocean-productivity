import io, os
import warnings
import numpy as np
import pandas as pd
import rasterio
import netCDF4 as nc
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

YEAR = "2017"
WINDOWS = os.path.join("results", "Fig_S31_Table_S13_S14", "intermediate", f"windows_conditions_{YEAR}.csv")
CHLOR = os.path.join("data", "chlor", f"chlor_{YEAR}.tif")
EMISSION = os.path.join("data", "emission", f"NO_{YEAR}_yearly.tif")
BASIN_MASK = os.path.join("data", "basinmask_01.msk")
COV_FILES = {"sst": os.path.join("data", "covariates", f"sst_{YEAR}.nc"),
             "par": os.path.join("data", "covariates", f"par_{YEAR}.nc"),
             "mld": os.path.join("data", "covariates", f"mld_{YEAR}.nc")}
SURROGATE_WINDOWS_DIR = os.path.join(f"surrogate_102_{YEAR}", "windows")
OUT_DIR = os.path.join("results", "Fig_S37_Table_S20_S21")
INT_DIR = os.path.join(OUT_DIR, "intermediate")

WINDOW, MIN_CELLS, MIN_MONTHS = 11, 70, 6
R2_SCREEN, SLOPE_ABS_MIN, PX_PER_DEG = 0.6, 0.01, 2
COLLINEAR = 0.8
DR2_THRESHOLDS = (0.05, 0.10)
FIG_MODEL = "all"
BASELINE_COUNTS = (119759, 5180, 4085)
BASIN_ORDER = ["North Pacific", "South Atlantic", "North Atlantic", "South Pacific", "Indian"]
MODEL_NAMES = {"sst": "SST", "all": "SST, PAR and MLD"}

ZONAL_DEG = [d * s for d in range(10, 65, 5) for s in (1, -1)]
MERID_DEG = [d * s for d in (10, 15, 20) for s in (1, -1)]
DIAG_LON = [d * s for d in (10, 15, 20, 30, 40, 50) for s in (1, -1)]
DIAG_LAT = [d * s for d in (10, 15, 20) for s in (1, -1)]

DIST_BINS = [0.0, 15.0, 25.0, 40.0, 61.0]
DIST_COLORS = ["#08306b", "#2171b5", "#6baed6", "#c6dbef"]; ROT_COLOR = "#f1c232"
PCT_BINS = np.arange(0, 115.1, 2.5); X_MARGIN = 5.0; FONT_FAMILY = "Times New Roman"


def dist_labels(shifted):
    """Label of each displacement range from the distances it contains."""
    cats = pd.cut(shifted["shift_dist_deg"], DIST_BINS)
    out = []
    for iv in cats.cat.categories:
        d = shifted.loc[cats == iv, "shift_dist_deg"]
        out.append(f"{d.min():g}\u00b0" if d.min() == d.max() else f"{d.min():.1f}\u00b0 to {d.max():.1f}\u00b0".replace(".0\u00b0", "\u00b0"))
    return out

def require(paths):
    """Stop with the list of missing input files."""
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit("[ERROR] missing input file(s):\n    " + "\n    ".join(missing))

def pixel_area_km2(lat_deg, res=0.5, r_earth=6_371_000):
    lat = np.deg2rad(lat_deg); d = np.deg2rad(res)
    return r_earth**2 * d * (np.sin(lat + d / 2) - np.sin(lat - d / 2)) / 1e6

def read_field(path):
    with rasterio.open(path) as src:
        a = src.read(1).astype("float64"); nod = src.nodata
        if nod is not None and np.isfinite(nod): a[a == nod] = np.nan
        a[~(np.isfinite(a) & (a > 0))] = np.nan
        return a, src.transform

def load_mask():
    with open(BASIN_MASK, encoding="utf-8", errors="replace") as fh:
        lines = [ln for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
    has_header = any(ch.isalpha() for ch in lines[0])
    m = pd.read_csv(io.StringIO("".join(lines)), header=0 if has_header else None)
    if not has_header: m.columns = ["Latitude", "Longitude"] + [f"Basin_{i}" for i in range(m.shape[1] - 2)]
    low = {c.lower().strip(): c for c in m.columns}
    lat_c = next((c for k, c in low.items() if k.startswith("lat")), m.columns[0])
    lon_c = next((c for k, c in low.items() if k.startswith("lon")), m.columns[1])
    code_c = next((c for k, c in low.items() if k.startswith("basin")), m.columns[2])
    d = pd.DataFrame({"mlat": m[lat_c].astype(float).round(2), "mlon": m[lon_c].astype(float).round(2),
                      "code": pd.to_numeric(m[code_c], errors="coerce")}).dropna()
    d["code"] = d["code"].astype(int); return d.drop_duplicates(["mlat", "mlon"])

def basin_name(code, lat):
    if code == 2: return "North Pacific" if lat >= 0 else "South Pacific"
    if code == 1: return "North Atlantic" if lat >= 0 else "South Atlantic"
    if code == 3: return "Indian"
    if code == 10: return "Southern Ocean"
    return "Other"

def assign_basin(lat, lon, mask):
    key = pd.DataFrame({"mlat": np.floor(lat) + 0.5, "mlon": np.floor(lon) + 0.5})
    key["mlon"] = np.where(key["mlon"] > 180, key["mlon"] - 360, key["mlon"])
    j = key.merge(mask, on=["mlat", "mlon"], how="left")
    out = np.array(["Unmapped"] * len(j), dtype=object); ok = j["code"].notna().to_numpy()
    out[ok] = [basin_name(int(c), la) for c, la in zip(j.loc[ok, "code"], np.asarray(lat)[ok])]
    return out

def window_blocks(A, ii, jj, k=WINDOW):
    half = k // 2; rows, cols = A.shape; n = len(ii)
    out = np.full((n, k * k), np.nan)
    for a, di in enumerate(range(-half, half + 1)):
        for b, dj in enumerate(range(-half, half + 1)):
            r = ii + di; c = jj + dj; ok = (r >= 0) & (r < rows) & (c >= 0) & (c < cols)
            vals = np.full(n, np.nan); vals[ok] = A[r[ok], c[ok]]; out[:, a * k + b] = vals
    return out

def read_covariate(path, checks):
    """Annual mean of a monthly 0.5 degree field, north-up on the chlorophyll a grid, at least MIN_MONTHS valid months."""
    d = nc.Dataset(path)
    lat_name = next(v for v in ("lat", "latitude", "y") if v in d.variables)
    lon_name = next(v for v in ("lon", "longitude", "x") if v in d.variables)
    var = next(v for v in d.variables if v not in ("time", "month", lat_name, lon_name) and d.variables[v].ndim == 3)
    lat = np.array(d.variables[lat_name][:], dtype=float); lon = np.array(d.variables[lon_name][:], dtype=float)
    A = np.array(d.variables[var][:], dtype="float64"); attrs = {k: str(d.variables[var].getncattr(k)) for k in d.variables[var].ncattrs()}
    d.close(); A[~np.isfinite(A)] = np.nan
    assert A.shape == (12, 360, 720), f"{path}: expected (12, 360, 720), got {A.shape}"
    assert np.allclose(np.abs(lat[:2]) % 0.5, 0.25) and np.allclose(np.abs(lon[:2]) % 0.5, 0.25), f"{path}: cell centers are not at x.25 / x.75"
    assert lon[0] < 0, f"{path}: longitude does not start at -180"
    flipped = lat[0] < lat[-1]
    if flipped: A = A[:, ::-1, :]; lat = lat[::-1]
    assert abs(lat[0] - 89.75) < 1e-3, f"{path}: after orientation the first latitude is {lat[0]}"
    n_ok = np.isfinite(A).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ann = np.where(n_ok >= MIN_MONTHS, np.nanmean(A, axis=0), np.nan)
    checks.append(f"{os.path.basename(path)}: variable {var}, long_name {attrs.get('long_name', '')!r}, units {attrs.get('units', '')!r}, "
                  f"stored {'south-up, flipped to north-up' if flipped else 'north-up'}, monthly range {np.nanmin(A):.2f} to {np.nanmax(A):.2f}, "
                  f"cells with at least {MIN_MONTHS} months {int((n_ok >= MIN_MONTHS).sum()):,}")
    return ann

def ols(X, y):
    Xd = np.column_stack([np.ones(len(y)), X]); beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    return beta, float(((y - Xd @ beta) ** 2).sum())

def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan

def per_window(logE, logC, covs, names):
    """Per-window quantities on the cells with chlorophyll a, emission and every covariate."""
    out = {}; ok = np.isfinite(logE) & np.isfinite(logC)
    for c in covs: ok &= np.isfinite(c)
    n = int(ok.sum()); out["n_cov"] = n
    if n < MIN_CELLS: return out
    e, y = logE[ok], logC[ok]; Z = np.column_stack([c[ok] for c in covs]); tss = float(((y - y.mean()) ** 2).sum())
    beta, rss = ols(e[:, None], y); out["slope_base"] = beta[1]; out["R2_base"] = 1 - rss / tss
    for k, nm in enumerate(names):
        out[f"r_C_{nm}"] = corr(y, Z[:, k]); out[f"r_E_{nm}"] = corr(e, Z[:, k])
    models = {"all": list(range(len(names)))}
    if "sst" in names: models["sst"] = [names.index("sst")]
    for mname, idx in models.items():
        Zm = Z[:, idx]; b_full, rss_full = ols(np.column_stack([e, Zm]), y); _, rss_cov = ols(Zm, y)
        out[f"b_E_{mname}"] = b_full[1]; out[f"R2_cov_{mname}"] = 1 - rss_cov / tss; out[f"R2_full_{mname}"] = 1 - rss_full / tss
        out[f"dR2_{mname}"] = out[f"R2_full_{mname}"] - out[f"R2_cov_{mname}"]
    return out

def screen_all_windows(logC, logE, k=WINDOW):
    """Vectorized screening regression over the grid: per-cell n, slope, R2 and the entering flag."""
    ok = np.isfinite(logC) & np.isfinite(logE)
    x = np.where(ok, logE, 0.0); y = np.where(ok, logC, 0.0); w = ok.astype(float)
    half = k // 2; rows, cols = ok.shape; S = {key: np.zeros((rows, cols)) for key in ("n", "x", "y", "xx", "yy", "xy")}
    for di in range(-half, half + 1):
        for dj in range(-half, half + 1):
            r0, r1 = max(0, -di), min(rows, rows - di); c0, c1 = max(0, -dj), min(cols, cols - dj)
            t, s = (slice(r0, r1), slice(c0, c1)), (slice(r0 + di, r1 + di), slice(c0 + dj, c1 + dj))
            S["n"][t] += w[s]; S["x"][t] += x[s]; S["y"][t] += y[s]; S["xx"][t] += x[s] ** 2; S["yy"][t] += y[s] ** 2; S["xy"][t] += x[s] * y[s]
    n = S["n"]
    with np.errstate(invalid="ignore", divide="ignore"):
        sxx = S["xx"] - S["x"] ** 2 / n; syy = S["yy"] - S["y"] ** 2 / n; sxy = S["xy"] - S["x"] * S["y"] / n
        slope = sxy / sxx; r2 = sxy ** 2 / (sxx * syy)
    entering = (n >= MIN_CELLS) & (sxx > 1e-12) & (syy > 1e-12) & (np.abs(slope) >= SLOPE_ABS_MIN)
    return n, slope, r2, entering

def scenarios():
    scen = [("baseline", 0, 0, "none")] + [(f"lon{d:+d}", d, 0, "shift") for d in ZONAL_DEG]
    scen += [(f"lat{d:+d}", 0, d, "shift") for d in MERID_DEG]
    scen += [(f"lon{dl:+d}_lat{dt:+d}", dl, dt, "shift") for dl in DIAG_LON for dt in DIAG_LAT]
    scen.append(("rot180", 0, 0, "rot180")); return scen

def surrogate_field(E_raw, dlon, dlat, mode):
    if mode == "none": return E_raw
    if mode == "rot180": return np.rot90(E_raw, 2)
    return np.roll(E_raw, (-int(round(dlat * PX_PER_DEG)), int(round(dlon * PX_PER_DEG))), axis=(0, 1))

def positive_windows(tag, logC, logE, tf):
    """Retained positive-slope windows of a field, from the Note 1 table when present, else regenerated."""
    path = os.path.join(SURROGATE_WINDOWS_DIR, f"windows_{tag}.csv")
    n, slope, r2, entering = screen_all_windows(logC, logE)
    if os.path.exists(path):
        t = pd.read_csv(path); t = t[(t["R2"] >= R2_SCREEN) & (t["Slope"] > 0) & (t["Slope"].abs() >= SLOPE_ABS_MIN)]
        col, row = (~tf) * (t["Longitude"].to_numpy(), t["Latitude"].to_numpy())
        return np.floor(row).astype(int), np.floor(col).astype(int), n, "table"
    ii, jj = np.where(entering & (r2 >= R2_SCREEN) & (slope > 0))
    if tag == "baseline":
        counts = (int(entering.sum()), int((entering & (r2 >= R2_SCREEN)).sum()), len(ii))
        assert counts == BASELINE_COUNTS, f"baseline regeneration gives {counts}, expected {BASELINE_COUNTS}"
    return ii, jj, n, "regenerated"

def main():
    os.makedirs(INT_DIR, exist_ok=True); checks = []
    require([WINDOWS, CHLOR, EMISSION, BASIN_MASK] + list(COV_FILES.values()))
    names = ["sst", "par", "mld"]
    C, tf = read_field(CHLOR); E_real, tf2 = read_field(EMISSION); assert C.shape == E_real.shape == (360, 720) and tf == tf2
    with rasterio.open(EMISSION) as src: E_raw = src.read(1).astype("float64")
    fields = {}
    for k in names:
        f = read_covariate(COV_FILES[k], checks)
        checks.append(f"{k}: annual mean covers {np.isfinite(f)[np.isfinite(C)].mean() * 100:.1f}% of the chlorophyll a cells")
        fields[k] = np.log1p(f) if k == "mld" else f
    logC_all = np.where(np.isfinite(C), np.log1p(C), np.nan)

    w = pd.read_csv(WINDOWS)
    col, row = (~tf) * (w["Longitude"].to_numpy(), w["Latitude"].to_numpy()); ii, jj = np.floor(row).astype(int), np.floor(col).astype(int)
    both = np.isfinite(C) & np.isfinite(E_real); logE = np.where(both, np.log1p(E_real), np.nan); logC = np.where(both, logC_all, np.nan)
    bE, bC = window_blocks(logE, ii, jj), window_blocks(logC, ii, jj); bX = [window_blocks(fields[k], ii, jj) for k in names]
    chk = np.random.default_rng(0).choice(len(w), 300, replace=False); dmax = 0.0
    for q in chk:
        m = np.isfinite(bE[q]) & np.isfinite(bC[q]); _, rss = ols(bE[q][m][:, None], bC[q][m])
        dmax = max(dmax, abs(1 - rss / float(((bC[q][m] - bC[q][m].mean()) ** 2).sum()) - w["R2"].iloc[q]))
    checks.append(f"screening regression reproduced on 300 windows of the real field: max |dR2| = {dmax:.1e}")
    res = pd.concat([w[["Longitude", "Latitude", "Slope", "R2", "retained", "positive", "n_valid"]].reset_index(drop=True),
                     pd.DataFrame([per_window(bE[q], bC[q], [b[q] for b in bX], names) for q in range(len(w))])], axis=1)
    res["basin"] = assign_basin(res["Latitude"].to_numpy(), res["Longitude"].to_numpy(), load_mask())
    res["area_km2"] = pixel_area_km2(res["Latitude"].to_numpy())
    res.to_csv(os.path.join(INT_DIR, f"windows_covariates_{YEAR}.csv"), index=False, float_format="%.5f")
    pos = res[res["positive"] & (res["n_cov"] >= MIN_CELLS)].copy()
    checks.append(f"windows entering the screen {len(res):,}; retained positive-slope {int(res['positive'].sum()):,}; with covariate data {len(pos):,}")

    def s29(g):
        return {"Windows": len(g), **{f"r(chl, {nm.upper()})": g[f"r_C_{nm}"].median() for nm in names},
                **{f"r(NOx, {nm.upper()})": g[f"r_E_{nm}"].median() for nm in names},
                f"|r(NOx, SST)| >= {COLLINEAR} (%)": (g["r_E_sst"].abs() >= COLLINEAR).mean() * 100}
    t29 = pd.DataFrame([dict(Region="Global", **s29(pos))] + [dict(Region=b, **s29(pos[pos["basin"] == b])) for b in BASIN_ORDER])
    checks.append(f"real field, retained positive-slope windows: median R2 of emission alone {pos['R2_base'].median():.3f}, of SST alone {pos['R2_cov_sst'].median():.3f}, "
                  f"of the three covariates alone {pos['R2_cov_all'].median():.3f}; median dR2 {pos['dR2_sst'].median():.3f} (SST) and {pos['dR2_all'].median():.3f} (all)")
    for thr in DR2_THRESHOLDS:
        for m in ("sst", "all"):
            checks.append(f"real field: surviving windows dR2 > {thr:.2f} after {MODEL_NAMES[m]}: {int((pos[f'dR2_{m}'] > thr).sum()):,} ({(pos[f'dR2_{m}'] > thr).mean() * 100:.1f}%), area {pos['area_km2'][pos[f'dR2_{m}'] > thr].sum() / 1e6:.2f} x 10^6 km2")

    ri, rj = np.floor((90 - pos["Latitude"].to_numpy()) / 0.5).astype(int), np.floor((pos["Longitude"].to_numpy() + 180) / 0.5).astype(int)
    sur_rows, paired = [], []
    for tag, dlon, dlat, mode in scenarios():
        if tag == "baseline":
            _, _, _, source = positive_windows(tag, logC, logE, tf); checks.append(f"baseline windows {source}"); continue
        E = surrogate_field(E_raw, dlon, dlat, mode); E = np.where(np.isfinite(E) & (E > 0), E, np.nan)
        b2 = np.isfinite(C) & np.isfinite(E); lE = np.where(b2, np.log1p(E), np.nan); lC = np.where(b2, logC_all, np.nan)
        si, sj, n_field, source = positive_windows(tag, lC, lE, tf)
        common = n_field >= MIN_CELLS
        sE, sC = window_blocks(lE, si, sj), window_blocks(lC, si, sj); sX = [window_blocks(fields[k], si, sj) for k in names]
        s = pd.DataFrame([per_window(sE[q], sC[q], [b[q] for b in sX], names) for q in range(len(si))])
        s = s[s["n_cov"] >= MIN_CELLS] if len(s) else s
        s.insert(0, "field", tag); s["Latitude"] = 90 - 0.25 - 0.5 * si[s.index]; s["Longitude"] = -180 + 0.25 + 0.5 * sj[s.index]
        s["area_km2"] = pixel_area_km2(s["Latitude"].to_numpy()); sur_rows.append(s)
        r_on = pos[common[ri, rj]]
        prow = {"scenario": tag, "dlon": dlon, "dlat": dlat, "mode": mode, "shift_dist_deg": np.nan if mode == "rot180" else np.hypot(dlon, dlat),
                "windows_source": source, "common windows": int(common.sum()), "surrogate positive windows": len(s), "real positive windows on common": len(r_on)}
        for m in ("sst", "all"):
            for thr in DR2_THRESHOLDS:
                rs, ss = r_on[f"dR2_{m}"] > thr, s[f"dR2_{m}"] > thr
                prow[f"{m}_{thr}_count_real"], prow[f"{m}_{thr}_count_sur"] = int(rs.sum()), int(ss.sum())
                prow[f"{m}_{thr}_area_real"], prow[f"{m}_{thr}_area_sur"] = r_on["area_km2"][rs].sum() / 1e6, s["area_km2"][ss].sum() / 1e6
                prow[f"{m}_{thr}_count_pct"] = prow[f"{m}_{thr}_count_sur"] / prow[f"{m}_{thr}_count_real"] * 100
                prow[f"{m}_{thr}_area_pct"] = prow[f"{m}_{thr}_area_sur"] / prow[f"{m}_{thr}_area_real"] * 100
        paired.append(prow); print(f"[surrogate] {tag:16s} {source:11s} positive windows {len(s):5d}  {MODEL_NAMES['all']}, dR2 > {DR2_THRESHOLDS[0]:.2f}: {prow[f'all_{DR2_THRESHOLDS[0]}_count_pct']:5.1f}% of the real field")
    SUR = pd.concat(sur_rows, ignore_index=True); SUR.to_csv(os.path.join(INT_DIR, f"surrogate_windows_{YEAR}.csv"), index=False, float_format="%.5f")
    t29 = pd.concat([t29, pd.DataFrame([dict(Region=f"Displaced fields, {len(paired)} fields pooled", **s29(SUR))])], ignore_index=True)
    t29.to_csv(os.path.join(OUT_DIR, "Table_S20_covariation_by_basin.csv"), index=False, float_format="%.2f", encoding="utf-8-sig")
    P = pd.DataFrame(paired); P.to_csv(os.path.join(INT_DIR, f"paired_{YEAR}_covariates.csv"), index=False, float_format="%.3f")

    rows30 = []
    for m in ("sst", "all"):
        for thr in DR2_THRESHOLDS:
            for stat, unit in (("count", "windows"), ("area", "10^6 km2")):
                pct = P[f"{m}_{thr}_{stat}_pct"]; k = pct.idxmax(); sur_val = P[f"{m}_{thr}_{stat}_sur"]
                real_val = int((pos[f"dR2_{m}"] > thr).sum()) if stat == "count" else pos["area_km2"][pos[f"dR2_{m}"] > thr].sum() / 1e6
                fmt = (lambda v: f"{v:,.0f}") if stat == "count" else (lambda v: f"{v:.2f}")
                rank = int((pct > 100).sum()) + 1
                rows30.append({"Covariates": MODEL_NAMES[m], "Emission effect threshold (increase in R2)": thr, "Statistic": "Windows keeping an emission effect" if stat == "count" else "Area of these windows (10^6 km2)",
                               "Real field": fmt(real_val), "Surrogate mean": fmt(sur_val.mean()), "Surrogate maximum": fmt(sur_val.max()),
                               "Closest surrogate on common windows (% of real)": f"{pct[k]:.1f}", "Closest surrogate": P.loc[k, "scenario"],
                               "Surrogates above 90% on common windows": int((pct >= 90).sum()), "Rank of the real field": f"{rank} of {len(P) + 1}", "Empirical p": f"{rank / (len(P) + 1):.4f}"})
    pd.DataFrame(rows30).to_csv(os.path.join(OUT_DIR, "Table_S21_surrogate_comparison.csv"), index=False, encoding="utf-8-sig")

    if FONT_FAMILY:
        plt.rcParams.update({"font.family": FONT_FAMILY, "mathtext.fontset": "stix", "font.size": 13, "pdf.fonttype": 42, "ps.fonttype": 42})
    def label(r):
        if r["mode"] == "rot180": return "rotation"
        return " ".join(([f"{abs(int(r['dlon']))}\u00b0 {'E' if r['dlon'] > 0 else 'W'}"] if r["dlon"] else []) + ([f"{abs(int(r['dlat']))}\u00b0 {'N' if r['dlat'] > 0 else 'S'}"] if r["dlat"] else []))
    def panel(ax, ycol, xlabel, letter, title):
        sh, rot = P[P["mode"] != "rot180"], P[P["mode"] == "rot180"]; cats = pd.cut(sh["shift_dist_deg"], DIST_BINS)
        groups = [sh.loc[cats == iv, ycol].to_numpy() for iv in cats.cat.categories] + [rot[ycol].to_numpy()]
        ax.hist(groups, bins=PCT_BINS, stacked=True, color=DIST_COLORS + [ROT_COLOR], edgecolor="white", label=dist_labels(sh) + ["rotation"])
        ax.axvline(100.0, color="red", lw=2.5); rank = int((P[ycol] > 100.0).sum()) + 1; n = len(P) + 1
        ax.text(98.5, ax.get_ylim()[1] * 0.95, f"real field\nrank {rank}/{n}\np = {rank / n:.4f}", color="red", fontsize=12, va="top", ha="right")
        close = P[P[ycol] >= 90].sort_values(ycol, ascending=False)
        txt = ["fields above 90%:"] + [f"{label(r)} ({r[ycol]:.1f})" for _, r in close.iterrows()] if len(close) else [f"no field above 90%", f"closest: {label(P.loc[P[ycol].idxmax()])} ({P[ycol].max():.1f})"]
        ax.text(98.5, ax.get_ylim()[1] * 0.66, "\n".join(txt), fontsize=11, va="top", ha="right", color="0.2")
        ax.set_xticks([0, 20, 40, 60, 80, 90, 100]); ax.set_xlim(-X_MARGIN, 100 + X_MARGIN); ax.set_xlabel(xlabel); ax.set_ylabel("Number of surrogate fields")
        ax.set_title(r"$\bf{" + letter + r"}$ " + title, loc="left", fontsize=12); ax.legend(frameon=True, fontsize=11, title="displacement distance", title_fontsize=11, loc="upper left")
    thr0 = DR2_THRESHOLDS[0]; fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    cov = MODEL_NAMES[FIG_MODEL]
    panel(axes[0], f"{FIG_MODEL}_{thr0}_count_pct", "Number of windows (% of the real field on the common windows)", "a",
          f"Windows where emissions explain chlorophyll $a$ beyond {cov} ({YEAR})")
    panel(axes[1], f"{FIG_MODEL}_{thr0}_area_pct", "Area of windows (% of the real field on the common windows)", "b",
          f"Area of windows where emissions explain chlorophyll $a$ beyond {cov} ({YEAR})")
    fig.tight_layout(); base = os.path.join(OUT_DIR, "Fig_S37_surrogate_comparison_covariates")
    fig.savefig(base + ".png", dpi=300); fig.savefig(base + ".pdf"); plt.close(fig)

    with open(os.path.join(OUT_DIR, "checks.txt"), "w", encoding="utf-8") as fh: fh.write("\n".join(checks) + "\n")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print("\n".join(checks)); print(); print(t29.round(2).to_string(index=False)); print(); print(pd.DataFrame(rows30).round(3).to_string(index=False))
    print(f"[Saved] {base}.png / .pdf")


if __name__ == "__main__":
    main()
