import os
import numpy as np
import pandas as pd
import rasterio
from scipy.stats import linregress
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

YEAR = "2017"
CHLOR_PATH = os.path.join("data", "chlor", f"chlor_{YEAR}.tif")
EMISSION_PATH = os.path.join("data", "emission", f"NO_{YEAR}_yearly.tif")
RESULTS_CSV = os.path.join(f"surrogate_102_{YEAR}", f"results_{YEAR}.csv")

OUT_DIR = os.path.join("results", "Fig_S20")
FIG_STEM = os.path.join(OUT_DIR, f"Fig_S20_slope_distribution_{YEAR}")
STATS_PATH = os.path.join(OUT_DIR, "Fig_S20_slope_distribution_stats.csv")
WINDOWS_PATH = os.path.join(OUT_DIR, f"Fig_S20_windows_all_{YEAR}.csv")

WINDOW_SIZE = 11
MIN_VALID = 70
R2_THRESH = 0.6
SLOPE_ABS_MIN = 0.01


FONT_SIZE = 13
DPI = 400
N_BINS = 120
LEGEND_HEIGHT = 0.09
PANEL_LETTER_X = -0.1


def regress_all_windows(chlor_raster_path, emission_raster_path, window_size=WINDOW_SIZE):
    """Every window regressed in log1p space from the cells with a valid value (finite and > 0) in both rasters, paired cell by cell; the slope floor is recorded as a flag instead of dropping the window."""
    with rasterio.open(chlor_raster_path) as src_c, rasterio.open(emission_raster_path) as src_e:
        assert src_c.transform == src_e.transform and src_c.shape == src_e.shape, "Resolution or CRS mismatch"
        chl, em, tf = src_c.read(1), src_e.read(1), src_c.transform
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
            lon, lat = tf * (j + 0.5, i + 0.5)
            results.append({"Longitude": lon, "Latitude": lat,
                            "Intercept": intercept, "Slope": slope,
                            "R": r_value, "R2": r_value**2,
                            "P_Value": p_value, "Count": n,
                            "below_slope_floor": abs(slope) < SLOPE_ABS_MIN})
    return pd.DataFrame(results)


def require(paths):
    """Stop with the list of missing input files."""
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit("[ERROR] missing input file(s):\n    " + "\n    ".join(missing))


def consistency(entering, screened, scr_pos, scr_neg):
    """Compare the 2017 counts with the surrogate results CSV; report only."""
    if not os.path.exists(RESULTS_CSV):
        print(f"[Consistency] entering {entering:,}; screened {screened:,} (results CSV not found, nothing to compare)")
        return
    res = pd.read_csv(RESULTS_CSV)
    row = res[(res["year"].astype(str) == YEAR) & (res["scenario"] == "baseline")]
    if row.empty:
        print("[Consistency] no 2017 baseline row in the results CSV")
        return
    r = row.iloc[-1]
    def mark(name, got, ref):
        return f"{name} {got:,} -> " + ("ok" if got == ref else f"differs from {ref:,} by {got - ref:+,}")
    print("[Consistency] " + "; ".join([mark("entering", entering, int(r["n_regressed"])),
                                        mark("screened", screened, int(r["n_screened"])),
                                        mark("positive", scr_pos, int(r["n_pos"])),
                                        mark("negative", scr_neg, int(r["n_neg"]))]))


def describe(arr, label):
    """One stats row: n, mean, median, p01, p25, p75, p99, min, max."""
    arr = np.asarray(arr, dtype="float64")
    q = np.percentile(arr, [1, 25, 75, 99])
    return {"group": label, "n": int(arr.size), "mean": float(np.mean(arr)), "median": float(np.median(arr)),
            "p01": float(q[0]), "p25": float(q[1]), "p75": float(q[2]), "p99": float(q[3]),
            "min": float(np.min(arr)), "max": float(np.max(arr))}


def set_style():
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": FONT_SIZE,
                         "axes.titlesize": FONT_SIZE, "axes.labelsize": FONT_SIZE,
                         "xtick.labelsize": FONT_SIZE, "ytick.labelsize": FONT_SIZE,
                         "legend.fontsize": FONT_SIZE})


def panel_letter(ax, letter, subtitle):
    ax.set_title(subtitle, loc="center")
    ax.text(PANEL_LETTER_X, 1.0, letter, transform=ax.transAxes + ax.titleOffsetTrans, fontweight="bold", ha="left", va="baseline")


def main():
    """Regress the 2017 rasters, report the counts against the results CSV, write the window table, the stats CSV and the two-panel figure."""
    require([CHLOR_PATH, EMISSION_PATH])
    os.makedirs(OUT_DIR, exist_ok=True)
    set_style()

    print(f"[Regress] {CHLOR_PATH} on {EMISSION_PATH}, {WINDOW_SIZE} x {WINDOW_SIZE} windows, at least {MIN_VALID} valid pixels")
    df = regress_all_windows(CHLOR_PATH, EMISSION_PATH)
    df.to_csv(WINDOWS_PATH, index=False)
    slope = df["Slope"].to_numpy(dtype="float64")
    r2 = df["R2"].to_numpy(dtype="float64")

    floor_removed = np.abs(slope) < SLOPE_ABS_MIN
    screened = ~floor_removed & (r2 >= R2_THRESH)
    scr_pos = screened & (slope > 0)
    scr_neg = screened & (slope < 0)
    consistency(int((~floor_removed).sum()), int(screened.sum()), int(scr_pos.sum()), int(scr_neg.sum()))

    all_pos = slope > 0
    stats = pd.DataFrame([describe(slope, "all windows"),
                          describe(slope[all_pos], "all windows, positive slope"),
                          describe(slope[screened], "screened"),
                          describe(slope[scr_pos], "screened positive"),
                          describe(slope[scr_neg], "screened negative")])
    stats.to_csv(STATS_PATH, index=False, encoding="utf-8-sig")
    print(f"[Bias] median positive slope, screened vs all positive: "
          f"{np.median(slope[scr_pos]):.4g} vs {np.median(slope[all_pos]):.4g} "
          f"(ratio {np.median(slope[scr_pos]) / np.median(slope[all_pos]):.3f})")

    lim_a = np.percentile(np.abs(slope), 99.9)
    lim_b = max(np.percentile(np.abs(slope), 99.0), 5 * SLOPE_ABS_MIN)
    bins_a = np.linspace(-lim_a, lim_a, N_BINS + 1)
    bins_b = np.linspace(-lim_b, lim_b, N_BINS + 1)
    clipped = int((np.abs(slope) > lim_a).sum())
    clipped_scr = int((np.abs(slope[screened]) > lim_a).sum())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(slope, bins=bins_a, color="0.6", label="all windows before screening")
    ax.hist(slope[screened], bins=bins_a, color="#d95f02", alpha=0.85, label="screened (R$^2$ $\\geq$ 0.6)")
    ax.set_yscale("log")
    ax.set_xlabel("slope (log1p space)")
    ax.set_ylabel("number of windows")
    panel_letter(ax, "a", "All windows before screening and the screened subset")
    note = (f"n = {slope.size:,}\nscreened = {int(screened.sum()):,}\n"
            f"({int(scr_pos.sum()):,} positive / {int(scr_neg.sum()):,} negative)")
    ax.text(0.02, 0.97, note, transform=ax.transAxes, va="top")

    ax = axes[1]
    ax.hist(slope, bins=bins_b, color="0.6", label="all windows before screening")
    ax.hist(slope[scr_pos], bins=bins_b, histtype="step", lw=1.8, color="#d7301f", label="screened positive")
    ax.hist(slope[scr_neg], bins=bins_b, histtype="step", lw=1.8, color="#0570b0", label="screened negative")
    ax.set_yscale("log")
    ax.set_xlabel("slope (log1p space)")
    ax.set_ylabel("number of windows")
    panel_letter(ax, "b", "Screened windows by sign of slope, central range")
    ax.text(0.02, 0.97, f"{int(floor_removed.sum()):,} windows removed\nwith |slope| < {SLOPE_ABS_MIN}",
            transform=ax.transAxes, va="top")

    handles, labels = {}, []
    for a in axes:
        for h, l in zip(*a.get_legend_handles_labels()):
            if l not in handles:
                handles[l] = h; labels.append(l)
    fig.legend([handles[l] for l in labels], labels, loc="lower center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, 0.0), columnspacing=2.0, handlelength=1.6)
    fig.tight_layout(rect=[0, LEGEND_HEIGHT, 1, 1])
    fig.savefig(FIG_STEM + ".png", dpi=DPI)
    fig.savefig(FIG_STEM + ".pdf")
    plt.close(fig)

    with pd.option_context("display.width", 160, "display.max_columns", 20, "display.float_format", lambda v: f"{v:.4g}"):
        print(stats.to_string(index=False))
    print(f"[Caption] {clipped} windows lie outside the panel A range, {clipped_scr} of them screened")
    print(f"[Saved] {WINDOWS_PATH}\n[Saved] {FIG_STEM}.png\n[Saved] {FIG_STEM}.pdf\n[Saved] {STATS_PATH}")
    print(f"[Done] Fig. S20 in {OUT_DIR}/")


if __name__ == "__main__":
    main()
