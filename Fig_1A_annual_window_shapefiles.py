import os
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from scipy.stats import linregress
from shapely.geometry import box

YEARS = ["2013", "2014", "2015", "2016", "2017"]
CHLOR_PATH = "data/chlor/chlor_{year}.tif"
EMISSION_PATH = "data/emission/NO_{year}_yearly.tif"
RESULTS_CSV = "surrogate_102_{year}/results_{year}.csv"
OUT_DIR = os.path.join("results", "Fig_1A")
DIR_SHP = os.path.join(OUT_DIR, "shp")
DIR_WINDOWS = os.path.join(OUT_DIR, "windows")

WINDOW_SIZE = 11
MIN_VALID = 70
R2_SCREEN = 0.6
SLOPE_ABS_MIN = 0.01


def perform_moving_window_regression(chlor_raster_path, emission_raster_path, window_size=WINDOW_SIZE):
    """Regress log1p chlorophyll a on log1p emission in every window over the cells valid in both fields; windows with fewer
    than MIN_VALID cells, constant values or an absolute slope below SLOPE_ABS_MIN are dropped."""
    with rasterio.open(chlor_raster_path) as src_c, rasterio.open(emission_raster_path) as src_e:
        assert src_c.transform == src_e.transform and src_c.shape == src_e.shape, "Resolution or CRS mismatch"
        chl = src_c.read(1).astype("float64"); em = src_e.read(1).astype("float64")
        for arr, nod in ((chl, src_c.nodata), (em, src_e.nodata)):
            if nod is not None and np.isfinite(nod):
                arr[arr == nod] = np.nan
        tf = src_c.transform
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
            if np.all(a_vals == a_vals[0]) or np.all(b_vals == b_vals[0]):
                continue
            slope, intercept, r_value, p_value, _ = linregress(b_vals, a_vals)
            if abs(slope) < SLOPE_ABS_MIN:
                continue
            slope_raw = linregress(em[r0:r1, c0:c1].ravel()[k], chl[r0:r1, c0:c1].ravel()[k]).slope
            lon, lat = tf * (j + 0.5, i + 0.5)
            results.append({"row": i, "col": j, "Longitude": lon, "Latitude": lat, "Intercept": intercept, "Slope": slope,
                            "R": r_value, "R2": r_value**2, "P_Value": p_value, "Count": n, "Slope_raw": slope_raw})
    return pd.DataFrame(results)


def consistency(year, df_all, df_ret):
    """Compare the counts of one year with the surrogate results CSV; report only."""
    p = RESULTS_CSV.format(year=year)
    if not os.path.exists(p):
        print(f"[Consistency {year}] {p} not found, nothing to compare"); return
    res = pd.read_csv(p); row = res[(res["year"].astype(str) == str(year)) & (res["scenario"] == "baseline")]
    if row.empty:
        print(f"[Consistency {year}] no baseline row in {p}"); return
    r = row.iloc[-1]; n_pos = int((df_ret["Slope"] >= 0).sum())
    mark = lambda name, got, ref: f"{name} {got:,} -> " + ("ok" if got == ref else f"differs from {ref:,} by {got - ref:+,}")
    print(f"[Consistency {year}] " + "; ".join([mark("entering", len(df_all), int(r["n_regressed"])), mark("retained", len(df_ret), int(r["n_screened"])),
                                               mark("positive", n_pos, int(r["n_pos"]))]))


def write_shapefiles(df, year, xres, yres, crs):
    """The seven shapefiles of one year, one square cell per retained window."""
    half_lon, half_lat = xres / 2, yres / 2
    def shp(sub, tag):
        geoms = [box(lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat) for lon, lat in zip(sub["Longitude"], sub["Latitude"])]
        out = os.path.join(DIR_SHP, f"moving_window_regression_{year}_{tag}.shp")
        gpd.GeoDataFrame(sub.copy(), geometry=geoms, crs=crs).to_file(out)
        print(f"  [Saved] {out} ({len(sub):,} windows)")
    shp(df[df["Slope"] >= 0], "pos"); shp(df[df["Slope"] < 0], "neg")
    shp(df[np.isfinite(df["R"])], "r_all"); shp(df[df["R"] >= 0], "r_pos"); shp(df[df["R"] < 0], "r_neg")
    shp(df[df["Slope_raw"] >= 0], "raw_pos"); shp(df[df["Slope_raw"] < 0], "raw_neg")


def main():
    for d in (DIR_SHP, DIR_WINDOWS):
        os.makedirs(d, exist_ok=True)
    for year in YEARS:
        chlor_path, emission_path = CHLOR_PATH.format(year=year), EMISSION_PATH.format(year=year)
        missing = [p for p in (chlor_path, emission_path) if not os.path.exists(p)]
        if missing:
            raise SystemExit("[ERROR] missing input file(s):\n    " + "\n    ".join(missing))
        print(f"Processing {year} ...")
        with rasterio.open(chlor_path) as src:
            xres, yres = src.res; crs = src.crs
        df = perform_moving_window_regression(chlor_path, emission_path)
        df.to_csv(os.path.join(DIR_WINDOWS, f"windows_{year}_all.csv"), index=False)
        ret = df[df["R2"] >= R2_SCREEN].reset_index(drop=True)
        ret.to_csv(os.path.join(DIR_WINDOWS, f"windows_{year}_retained.csv"), index=False)
        consistency(year, df, ret)
        write_shapefiles(ret, year, xres, yres, crs)
    print(f"[Done] Fig. 1A and Fig. 3B shapefiles in {DIR_SHP}/")


if __name__ == "__main__":
    main()
