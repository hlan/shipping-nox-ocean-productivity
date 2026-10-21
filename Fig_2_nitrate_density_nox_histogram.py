import os
import numpy as np
import pandas as pd
import rasterio
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

R = 6_371_000
YEARS = ["2013", "2014", "2015", "2016", "2017"]
YEAR_A = "2017"
YEAR_B = "2017"
LABEL_POOLED = "2013-2017"
NMEAN_CSV = "data/Nmean_60m_data_updated.csv"
CHLOR_PATH = "data/chlor/chlor_{year}.tif"
EMISSION_PATH = "data/emission/NO_{year}_yearly.tif"
WINDOWS_ALL = "surrogate_102_{year}/windows/windows_baseline_all.csv"
WINDOWS_RETAINED = "surrogate_102_{year}/windows/windows_baseline.csv"
RESULTS_CSV = "surrogate_102_{year}/results_{year}.csv"
OUT_DIR = os.path.join("results", "Fig_2")
DIR_A = os.path.join(OUT_DIR, "panel_a")
DIR_B_POS = os.path.join(OUT_DIR, "panel_b", "positive")
DIR_B_NEG = os.path.join(OUT_DIR, "panel_b", "negative")
DIR_B_BOTH = os.path.join(OUT_DIR, "panel_b", "combined")

FONT = "Times New Roman"
R2_MIN = 0.6
LOW_THRESHOLD = 2.0
XLIM_MAX = 8.0
BW_COMMON = 0.25    
FILL_ALPHA = 0.18
BINS = 30
LABEL_FS = 12
TICK_FS = 11
TITLE_FS = 13
BOX_FS = 11
PANEL_LETTER_FS = 14
PANEL_LETTER_X = -0.1
PANEL_GAP = 1.0
GROUPS_A = [("Positively correlated", "#ff7f0e", "-"), ("All grids", "#4d4d4d", "--"), ("Negatively correlated", "#1f77b4", "-")]
TITLE_A = "Distribution of Mean Nitrate Concentration in the Upper 60 m by Correlation Group"
XLABEL_A = "Mean Nitrate Concentration in the Upper 60 m (μmol kg$^{-1}$)"
YLABEL_A = "Normalized Area-Weighted Density"
NOX = "NO$_{\\mathrm{x}}$"
XLABEL_B = NOX + " Emissions (g m$^{-2}$ yr$^{-1}$)"
YLABEL_B = "Area (10$^{6}$ km$^{2}$)"

plt.rcParams.update({"font.family": FONT, "font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12, "xtick.labelsize": 12,
                     "ytick.labelsize": 12, "legend.fontsize": 10, "axes.unicode_minus": False, "mathtext.fontset": "stix",
                     "pdf.fonttype": 42})


def pixel_area_m2(lat_deg, xres_deg, yres_deg):
    lat = np.deg2rad(lat_deg); dlat = np.deg2rad(yres_deg); dlon = np.deg2rad(xres_deg)
    return R**2 * dlon * (np.sin(lat + dlat / 2) - np.sin(lat - dlat / 2))


def consistency(year, name, got, column):
    """Compare one count with the surrogate results CSV; report only."""
    p = RESULTS_CSV.format(year=year)
    if not os.path.exists(p):
        print(f"[Consistency {year}] {name} {got:,} ({p} not found, nothing to compare)"); return
    res = pd.read_csv(p); row = res[(res["year"].astype(str) == year) & (res["scenario"] == "baseline")]
    if row.empty:
        print(f"[Consistency {year}] {name} {got:,} (no baseline row in {p})"); return
    ref = int(row.iloc[-1][column])
    print(f"[Consistency {year}] {name} {got:,} vs results CSV {ref:,} -> " + ("ok" if got == ref else f"differs by {got - ref:+,}"))


def nmean_values_and_area_weights(df, nmean_df, xres, yres):
    """WOA nitrate at the nearest 1 degree center of each window, with the surface-area weight of the center cell."""
    lon_centers = np.sort(nmean_df["Longitude"].unique()); lat_centers = np.sort(nmean_df["Latitude"].unique())
    def nearest_center(vals, centers):
        idx = np.searchsorted(centers, vals)
        idx0 = np.clip(idx - 1, 0, len(centers) - 1); idx1 = np.clip(idx, 0, len(centers) - 1)
        pick0 = np.abs(vals - centers[idx0]) <= np.abs(vals - centers[idx1])
        return np.where(pick0, centers[idx0], centers[idx1])
    d = df[["Longitude", "Latitude"]].dropna().copy()
    if d.empty:
        return np.array([]), np.array([])
    d["lon_c"] = nearest_center(d["Longitude"].values, lon_centers); d["lat_c"] = nearest_center(d["Latitude"].values, lat_centers)
    idx = nmean_df.set_index(["Longitude", "Latitude"])
    d["N60"] = idx.reindex(list(zip(d["lon_c"].values, d["lat_c"].values)))["mean_N_Depth_60"].values
    d = d[np.isfinite(d["N60"].values)]
    if d.empty:
        return np.array([]), np.array([])
    return d["N60"].values.astype(float), pixel_area_m2(d["Latitude"].values, xres, yres) / 1e12


def reflected_kde(vals, w, bw):
    """Area-weighted Gaussian KDE with the fixed bandwidth bw, reflected at 0 (2 x this density integrates to 1 over x >= 0)."""
    vv = np.concatenate([vals, -vals]); ww = np.concatenate([w, w]); ww = ww / np.sum(ww)
    sd = float(np.sqrt(np.cov(vv, aweights=ww)))
    return gaussian_kde(vv, weights=ww, bw_method=bw / sd)


def draw_density(ax, df_all, nmean_df, xres, yres):
    """Panel a: area-weighted density of the nitrate value for all windows entering the screen and the retained groups."""
    df_all = df_all[np.isfinite(df_all["Slope"])].copy()
    df_base = df_all[df_all["R2"] >= R2_MIN]
    sets = {"Positively correlated": df_base[df_base["Slope"] > 0], "All grids": df_all, "Negatively correlated": df_base[df_base["Slope"] < 0]}
    x = np.linspace(0.0, XLIM_MAX, 800); max_y = 0.0
    for label, color, ls in GROUPS_A:
        vals, w = nmean_values_and_area_weights(sets[label], nmean_df, xres, yres)
        if vals.size < 5:
            continue
        m = np.isfinite(vals) & np.isfinite(w) & (w > 0); vals, w = vals[m], w[m]
        y = 2 * reflected_kde(vals, w, BW_COMMON)(x)
        max_y = max(max_y, np.nanmax(y))
        ax.plot(x, y, color=color, linewidth=1.6, linestyle=ls, label=label)
        ax.fill_between(x, 0, y, color=color, alpha=FILL_ALPHA)
    ax.axvline(LOW_THRESHOLD, color="red", linestyle="--", linewidth=2)
    ymax = max_y if max_y > 0 else ax.get_ylim()[1]
    ax.text(LOW_THRESHOLD + 0.12, ymax * 0.92, "N limited regions (<2 μmol kg$^{-1}$)", color="red", fontsize=10, rotation=90, ha="left", va="top")
    ax.set_title(TITLE_A)
    ax.set_xlabel(XLABEL_A)
    ax.set_ylabel(YLABEL_A)
    ax.set_xlim(0, XLIM_MAX); ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    if max_y > 0:
        ax.set_ylim(0, max_y * 1.05)
    ax.legend(frameon=True, loc="best")


def extract_nox_and_area(df, emission_path, xres, yres):
    """NOx emission value and surface area (10^6 km2) of the center cell of each retained window."""
    if df.empty:
        return np.array([]), np.array([])
    with rasterio.open(emission_path) as src:
        nox = np.array([s[0] if s is not None and len(s) > 0 else np.nan for s in src.sample(list(zip(df["Longitude"].values, df["Latitude"].values)))], dtype=float)
    area = pixel_area_m2(df["Latitude"].values, xres, yres) / 1e12
    ok = np.isfinite(nox) & np.isfinite(area) & (area > 0)
    return nox[ok], area[ok]


def draw_single_histogram(ax, nox, area, year_label, color, sign_label):
    hist, edges = np.histogram(nox, bins=BINS, weights=area)
    centers = (edges[:-1] + edges[1:]) / 2
    ax.bar(centers, hist, width=np.diff(edges), edgecolor="black", color=color)
    ax.yaxis.set_major_locator(MultipleLocator(0.5)); ax.set_ylim(0, np.ceil(ax.get_ylim()[1] * 2) / 2 * 1.05)
    ax.set_xlabel(XLABEL_B, fontsize=LABEL_FS); ax.set_ylabel(YLABEL_B, fontsize=LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.set_title(f"{year_label} Chlor-{NOX} {sign_label} Correlated Area vs {NOX} Emission Rate", fontsize=TITLE_FS)
    ax.text(0.5, 0.98, f"Total pixels: {int(nox.size)}\nTotal area: {float(np.nansum(hist)):.2f} × 10⁶ km²", transform=ax.transAxes,
            va="top", ha="center", multialignment="left", bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.9), fontsize=BOX_FS)


def draw_combined_histogram(ax, nox_pos, area_pos, nox_neg, area_neg, year_label):
    """Panel b: area of the positive and negative retained windows by the NOx emission of their center cells."""
    all_nox = np.concatenate([v for v in (nox_pos, nox_neg) if v.size > 0])
    edges = np.histogram_bin_edges(all_nox, bins=BINS); centers = (edges[:-1] + edges[1:]) / 2
    handles = []
    if nox_pos.size:
        h, _ = np.histogram(nox_pos, bins=edges, weights=area_pos)
        ax.bar(centers, h, width=np.diff(edges), edgecolor="black", color="#1f77b4", alpha=0.7)
        handles.append(Patch(facecolor="#1f77b4", edgecolor="black", label=f"Positive: {int(nox_pos.size)} pixels, {float(np.nansum(area_pos)):.2f} × 10⁶ km²"))
    if nox_neg.size:
        h, _ = np.histogram(nox_neg, bins=edges, weights=area_neg)
        ax.bar(centers, h, width=np.diff(edges), edgecolor="black", color="#f0c929", alpha=0.7)
        handles.append(Patch(facecolor="#f0c929", edgecolor="black", label=f"Negative: {int(nox_neg.size)} pixels, {float(np.nansum(area_neg)):.2f} × 10⁶ km²"))
    ax.yaxis.set_major_locator(MultipleLocator(0.5)); ax.set_ylim(0, np.ceil(ax.get_ylim()[1] * 2) / 2 * 1.05)
    ax.set_xlabel(XLABEL_B, fontsize=LABEL_FS); ax.set_ylabel(YLABEL_B, fontsize=LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.set_title(f"{year_label} Chlor-{NOX} Positive and Negative Correlated Area vs {NOX} Emission Rate", fontsize=TITLE_FS)
    ax.legend(handles=handles, loc="upper center", fontsize=BOX_FS, framealpha=0.9, edgecolor="black", fancybox=True, borderpad=0.6)


def save(fig, base):
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight"); fig.savefig(base + ".pdf", bbox_inches="tight"); plt.close(fig)
    print(f"[Saved] {base}.png")


def weighted_quantile(values, quantiles, weights):
    v = np.asarray(values, float); w = np.asarray(weights, float); m = np.isfinite(v) & np.isfinite(w) & (w > 0); v, w = v[m], w[m]
    if v.size == 0:
        return np.array([np.nan] * len(quantiles))
    o = np.argsort(v); v, w = v[o], w[o]; cw = np.cumsum(w) / np.sum(w)
    return np.interp(quantiles, cw, v)


def summarize_nox(nox, area):
    """Area-weighted summary of the NOx emission of the positive windows."""
    m = np.isfinite(nox) & np.isfinite(area) & (area > 0); nox, area = nox[m], area[m]
    if nox.size == 0:
        return {"PixelCount": 0, "TotalArea_1e6km2": 0.0, "Mean_w": np.nan, "SD_w": np.nan, "Median_w": np.nan, "Q25_w": np.nan, "Q75_w": np.nan, "Min": np.nan, "Max": np.nan}
    mu = np.sum(area * nox) / np.sum(area); sd = np.sqrt(np.sum(area * (nox - mu) ** 2) / np.sum(area))
    q25, med, q75 = weighted_quantile(nox, [0.25, 0.5, 0.75], area)
    return {"PixelCount": int(nox.size), "TotalArea_1e6km2": float(np.sum(area)), "Mean_w": float(mu), "SD_w": float(sd), "Median_w": float(med),
            "Q25_w": float(q25), "Q75_w": float(q75), "Min": float(np.nanmin(nox)), "Max": float(np.nanmax(nox))}


def main():
    for d in (DIR_A, DIR_B_POS, DIR_B_NEG, DIR_B_BOTH):
        os.makedirs(d, exist_ok=True)
    needed = [NMEAN_CSV, CHLOR_PATH.format(year=YEAR_A), WINDOWS_ALL.format(year=YEAR_A)] + [WINDOWS_RETAINED.format(year=y) for y in YEARS] + [EMISSION_PATH.format(year=y) for y in YEARS]
    missing = [p for p in needed if not os.path.exists(p)]
    if missing:
        raise SystemExit("[ERROR] missing input file(s):\n    " + "\n    ".join(missing))

    # panel a
    with rasterio.open(CHLOR_PATH.format(year=YEAR_A)) as src:
        xres, yres = src.res
    df_all = pd.read_csv(WINDOWS_ALL.format(year=YEAR_A)); nmean_df = pd.read_csv(NMEAN_CSV)
    print(f"[Loaded] {len(df_all):,} windows entering the screen from {WINDOWS_ALL.format(year=YEAR_A)}")
    consistency(YEAR_A, "windows entering the screen", len(df_all), "n_regressed")
    fig, ax = plt.subplots(figsize=(8, 6)); draw_density(ax, df_all, nmean_df, xres, yres); save(fig, os.path.join(DIR_A, f"nmean_density_{YEAR_A}_3groups"))

    # panel b: every year and pooled, positive, negative and combined
    pooled = {"pos": ([], []), "neg": ([], [])}; summary_rows = []; data_b = {}
    for year in YEARS:
        df = pd.read_csv(WINDOWS_RETAINED.format(year=year)); emission_path = EMISSION_PATH.format(year=year)
        with rasterio.open(emission_path) as src:
            xres_y, yres_y = src.res
        nox_pos, area_pos = extract_nox_and_area(df[df["Slope"] > 0], emission_path, xres_y, yres_y)
        nox_neg, area_neg = extract_nox_and_area(df[df["Slope"] < 0], emission_path, xres_y, yres_y)
        consistency(year, "retained positive windows", int((df["Slope"] > 0).sum()), "n_pos")
        data_b[year] = (nox_pos, area_pos, nox_neg, area_neg)
        pooled["pos"][0].append(nox_pos); pooled["pos"][1].append(area_pos); pooled["neg"][0].append(nox_neg); pooled["neg"][1].append(area_neg)
        summary_rows.append({"Year": year, **summarize_nox(nox_pos, area_pos)})
    data_b[LABEL_POOLED] = tuple(np.concatenate(v) if v else np.array([]) for v in (pooled["pos"][0], pooled["pos"][1], pooled["neg"][0], pooled["neg"][1]))
    summary_rows.append({"Year": LABEL_POOLED, **summarize_nox(data_b[LABEL_POOLED][0], data_b[LABEL_POOLED][1])})
    for label, (nox_pos, area_pos, nox_neg, area_neg) in data_b.items():
        for nox, area, d, color, sign in ((nox_pos, area_pos, DIR_B_POS, "#1f77b4", "Positive"), (nox_neg, area_neg, DIR_B_NEG, "#f0c929", "Negative")):
            if nox.size:
                fig, ax = plt.subplots(figsize=(8, 6)); draw_single_histogram(ax, nox, area, label, color, sign); save(fig, os.path.join(d, f"nox_histogram_{label}_{sign.lower()[:3]}"))
        if nox_pos.size or nox_neg.size:
            fig, ax = plt.subplots(figsize=(8, 6)); draw_combined_histogram(ax, nox_pos, area_pos, nox_neg, area_neg, label); save(fig, os.path.join(DIR_B_BOTH, f"nox_histogram_{label}_combined"))

    # NOx summary of the positive windows
    summary = pd.DataFrame(summary_rows); excel_out = os.path.join(OUT_DIR, "nox_summary_2013_2017.xlsx")
    with pd.ExcelWriter(excel_out, engine="openpyxl") as writer:
        summary.to_excel(writer, index=False, sheet_name="Summary")
        pd.DataFrame({"NOx_emission": data_b[LABEL_POOLED][0], "AreaWeight_1e6km2": data_b[LABEL_POOLED][1]}).to_excel(writer, index=False, sheet_name="PooledRaw")
    r = summary.iloc[-1]
    print(f"[Saved] {excel_out}\n[Pooled {LABEL_POOLED}, positive windows] NOx emission: median={r['Median_w']:.4g} (IQR {r['Q25_w']:.4g}-{r['Q75_w']:.4g}) g m-2 yr-1; "
          f"area-weighted mean={r['Mean_w']:.4g} +/- {r['SD_w']:.4g}; total area={r['TotalArea_1e6km2']:.2f} x10^6 km2; pixels={int(r['PixelCount'])}.")

    # assembled figure: panel a for YEAR_A, panel b for YEAR_B (a year or LABEL_POOLED)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    draw_density(axes[0], df_all, nmean_df, xres, yres)
    draw_combined_histogram(axes[1], *data_b[YEAR_B], YEAR_B)
    for ax, letter in zip(axes, "ab"):
        ax.text(PANEL_LETTER_X, 1.0, letter, transform=ax.transAxes + ax.titleOffsetTrans, fontsize=PANEL_LETTER_FS, fontweight="bold", ha="left", va="baseline")
    fig.tight_layout(w_pad=PANEL_GAP)
    save(fig, os.path.join(OUT_DIR, "Fig_2"))
    print(f"[Done] Fig. 2 in {OUT_DIR}/")


if __name__ == "__main__":
    main()