import os
import numpy as np
import pandas as pd
import rasterio
from scipy.stats import fisher_exact

WINDOWS_PATH = os.path.join("surrogate_102_2017", "windows", "windows_baseline_all.csv")
RESULTS_CSV = os.path.join("surrogate_102_2017", "results_2017.csv")
NMEAN_CSV = os.path.join("data", "Nmean_60m_data_updated.csv")
CHLOR_PATH = os.path.join("data", "chlor", "chlor_2017.tif")
OUT_DIR = os.path.join("results", "Table_S2")

N_LIMIT = 2.0
R2_THRESH = 0.6
R_EARTH = 6_371_000


def require(paths):
    """Stop with the list of missing input files."""
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit("[ERROR] missing input file(s):\n    " + "\n    ".join(missing))


def pixel_area_m2(lat_deg, xres_deg, yres_deg):
    """Surface area of one grid cell at a latitude."""
    lat = np.deg2rad(lat_deg); dlat = np.deg2rad(yres_deg); dlon = np.deg2rad(xres_deg)
    return R_EARTH**2 * dlon * (np.sin(lat + dlat / 2) - np.sin(lat - dlat / 2))


def consistency(name, got, results_csv, column):
    """Compare one count with the surrogate results CSV; report only."""
    if not os.path.exists(results_csv):
        print(f"[Consistency] {name} {got:,} (results CSV not found, nothing to compare)")
        return
    res = pd.read_csv(results_csv)
    row = res[(res["year"].astype(str) == "2017") & (res["scenario"] == "baseline")]
    if row.empty:
        print(f"[Consistency] {name} {got:,} (no 2017 baseline row in the results CSV)")
        return
    ref = int(row.iloc[-1][column])
    mark = "ok" if got == ref else f"differs by {got - ref:+,}"
    print(f"[Consistency] {name} {got:,} vs results CSV {ref:,} -> {mark}")


def map_windows_to_nmean(df, nmean_csv_path):
    """Nearest 1 degree center lookup of the WOA nitrogen value, as in Fig. 2A."""
    nmean_df = pd.read_csv(nmean_csv_path)
    lon_centers = np.sort(nmean_df['Longitude'].unique())
    lat_centers = np.sort(nmean_df['Latitude'].unique())

    def nearest_center(vals, centers):
        idx = np.searchsorted(centers, vals)
        idx0 = np.clip(idx - 1, 0, len(centers) - 1)
        idx1 = np.clip(idx, 0, len(centers) - 1)
        pick0 = np.abs(vals - centers[idx0]) <= np.abs(vals - centers[idx1])
        return np.where(pick0, centers[idx0], centers[idx1])

    out = df.copy()
    out['lon_c'] = nearest_center(out['Longitude'].values, lon_centers)
    out['lat_c'] = nearest_center(out['Latitude'].values, lat_centers)

    nmean_idx = nmean_df.set_index(['Longitude', 'Latitude'])
    keys = list(zip(out['lon_c'].values, out['lat_c'].values))
    out['mean_N_Depth_60'] = nmean_idx.reindex(keys)['mean_N_Depth_60'].values
    return out


def main():
    """Classify the windows by nitrogen limitation and write the ratio and the Fisher table."""
    require([WINDOWS_PATH, NMEAN_CSV, CHLOR_PATH])
    os.makedirs(OUT_DIR, exist_ok=True)

    entering = pd.read_csv(WINDOWS_PATH)
    print(f"[Loaded] {len(entering):,} windows entering the screen from {WINDOWS_PATH}")
    consistency("windows entering the screen", len(entering), RESULTS_CSV, "n_regressed")
    n_pos_all = int(((entering["R2"] >= R2_THRESH) & (entering["Slope"] > 0)).sum())
    consistency("retained positive windows", n_pos_all, RESULTS_CSV, "n_pos")

    with rasterio.open(CHLOR_PATH) as src:
        xres, yres = src.res
    entering = map_windows_to_nmean(entering, NMEAN_CSV)
    entering["N60"] = entering["mean_N_Depth_60"]
    has_n = np.isfinite(entering["N60"])
    print(f"[WOA] {int(has_n.sum()):,} windows have a nitrogen value at the center cell; {int((~has_n).sum()):,} excluded")
    w = entering[has_n].copy()

    w["limited"] = w["N60"] < N_LIMIT
    w["positive"] = (w["R2"] >= R2_THRESH) & (w["Slope"] > 0)
    w["area"] = pixel_area_m2(w["Latitude"].to_numpy(), xres, yres)

    pos, other = w[w["positive"]], w[~w["positive"]]
    a = int((pos["limited"]).sum());   b = int((~pos["limited"]).sum())
    c = int((other["limited"]).sum()); d = int((~other["limited"]).sum())

    base_count = w["limited"].mean()
    base_area = w.loc[w["limited"], "area"].sum() / w["area"].sum()
    cond_count = pos["limited"].mean()
    cond_area = pos.loc[pos["limited"], "area"].sum() / pos["area"].sum()
    odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")

    row = {"windows_with_woa": len(w), "positive_windows": len(pos),
           "base_rate_count": base_count, "base_rate_area": base_area,
           "conditional_rate_count": cond_count, "conditional_rate_area": cond_area,
           "ratio_count": cond_count / base_count, "ratio_area": cond_area / base_area,
           "odds_ratio": odds, "p_one_sided_nominal": p}
    pd.DataFrame([row]).to_csv(os.path.join(OUT_DIR, "Table_S2_retained_to_all_ratio_2017.csv"), index=False)

    def f0(v): return f"{int(v):,}"
    def f3(v): return f"{v:.3f}"
    other_count = c / (c + d)
    other_area = other.loc[other["limited"], "area"].sum() / other["area"].sum()
    table = pd.DataFrame(
        [["Retained windows with positive slopes", f0(a + b), f0(a), f3(cond_count), f3(cond_area)],
         ["All other windows entering the screen", f0(c + d), f0(c), f3(other_count), f3(other_area)],
         ["All windows entering the screen", f0(len(w)), f0(a + c), f3(base_count), f3(base_area)]],
        columns=["Window set", "Windows", "In nitrogen-limited water", "Fraction by count", "Fraction by area"])
    table.to_csv(os.path.join(OUT_DIR, "Table_S2_N_limitation.csv"), index=False, encoding="utf-8-sig")

    print(f"[Result] base rate {base_count:.3%} by count, {base_area:.3%} by area")
    print(f"[Result] conditional rate {cond_count:.3%} by count, {cond_area:.3%} by area")
    print(f"[Result] retained-to-all ratio {cond_count / base_count:.3f} by count, {cond_area / base_area:.3f} by area")
    print(f"[Result] two-by-two table: positive {a:,} / {b:,}; others {c:,} / {d:,}; odds ratio {odds:.3f}, one-sided p {p:.3g} (nominal)")
    with pd.option_context("display.width", 160, "display.max_columns", 10):
        print(table.to_string(index=False))
    print(f"[Saved] {os.path.join(OUT_DIR, 'Table_S2_retained_to_all_ratio_2017.csv')}\n[Saved] {os.path.join(OUT_DIR, 'Table_S2_N_limitation.csv')}")
    print(f"[Done] Table S2 in {OUT_DIR}/")


if __name__ == "__main__":
    main()
