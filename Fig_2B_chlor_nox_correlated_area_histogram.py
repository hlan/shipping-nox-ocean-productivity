import os
import re
import glob
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

# Config
CHLOR_DIR    = "data/chlor"
EMISSION_DIR = "data/emission"

OUTPUT_ROOT     = "chlor_nox_histo"
OUTPUT_DIR_POS  = os.path.join(OUTPUT_ROOT, "positive")
OUTPUT_DIR_NEG  = os.path.join(OUTPUT_ROOT, "negative")
OUTPUT_DIR_BOTH = os.path.join(OUTPUT_ROOT, "combined")

WINDOW_SIZE   = 11
MIN_SAMPLES   = 70
MIN_ABS_SLOPE = 0.01
R2_THRESHOLD  = 0.6
BINS = 30
R_EARTH = 6_371_000

LABEL_FS = 12
TICK_FS  = 11
TITLE_FS = 13
BOX_FS   = 11


def pixel_area_m2(lat_deg, xres_deg, yres_deg):
    lat  = np.deg2rad(lat_deg)
    dlat = np.deg2rad(yres_deg)
    dlon = np.deg2rad(xres_deg)
    lat1 = lat - dlat / 2
    lat2 = lat + dlat / 2
    return R_EARTH**2 * dlon * (np.sin(lat2) - np.sin(lat1))


def _read_window(dataset, row, col, half):
    try:
        win = Window(col - half, row - half, 2 * half + 1, 2 * half + 1)
        data = dataset.read(1, window=win).flatten()
        data = data[np.isfinite(data) & (data > 0)]
        return np.log1p(data)
    except Exception:
        return np.array([])


# Moving-window regression
def moving_window_regression(chlor_path, emission_path):
    results = []
    half = WINDOW_SIZE // 2

    with rasterio.open(chlor_path) as src_c, \
         rasterio.open(emission_path) as src_e:

        assert src_c.transform == src_e.transform
        rows, cols = src_c.shape
        tf = src_c.transform

        for i in range(rows):
            for j in range(cols):
                a = _read_window(src_c, i, j, half)
                b = _read_window(src_e, i, j, half)

                n = min(len(a), len(b))
                if n < MIN_SAMPLES:
                    continue

                a, b = a[:n], b[:n]
                if np.all(a == a[0]):
                    continue

                slope, intercept, r_val, p_val, _ = linregress(a, b)
                if abs(slope) < MIN_ABS_SLOPE:
                    continue

                lon, lat = tf * (j + 0.5, i + 0.5)
                results.append({
                    "Longitude": lon,
                    "Latitude":  lat,
                    "Slope":     slope,
                    "R":         r_val,
                    "R2":        r_val ** 2,
                    "P_Value":   p_val,
                    "Count":     n,
                })

    return pd.DataFrame(results)


# NOx sampling
def extract_nox_and_area(df, emission_path, xres, yres):
    if df.empty:
        return np.array([]), np.array([])

    coords = list(zip(df["Longitude"].values, df["Latitude"].values))
    with rasterio.open(emission_path) as src:
        samples = list(src.sample(coords))
        nox = np.array(
            [s[0] if s is not None and len(s) > 0 else np.nan for s in samples],
            dtype=float,
        )

    area = np.array(
        [pixel_area_m2(lat, xres, yres) / 1e12 for lat in df["Latitude"].values],
        dtype=float,
    )

    ok = np.isfinite(nox) & np.isfinite(area) & (area > 0)
    return nox[ok], area[ok]


# Plotting
def _single_histogram(nox, area, output_png, year_label, color, sign_label):
    if nox.size == 0:
        print(f"[{year_label} {sign_label}] No valid samples, skipping.")
        return

    hist, edges = np.histogram(nox, bins=BINS, weights=area)
    centers = (edges[:-1] + edges[1:]) / 2
    total_area = float(np.nansum(hist))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(centers, hist, width=np.diff(edges), edgecolor="black", color=color)

    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ymax_round = np.ceil(ax.get_ylim()[1] * 2) / 2
    ax.set_ylim(0, ymax_round * 1.05)

    ax.set_xlabel("NOx emission (g·m⁻²·yr⁻¹)", fontsize=LABEL_FS)
    ax.set_ylabel("Area (×10⁶ km²)", fontsize=LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.set_title(
        f"{year_label} Chlor-NOx {sign_label} Correlated Area "
        f"vs NOx Emission Rate (R² ≥ {R2_THRESHOLD})",
        fontsize=TITLE_FS,
    )

    box_text = (
        f"Total pixels: {int(nox.size)}\n"
        f"Total area: {total_area:.2f} × 10⁶ km²"
    )
    ax.text(
        0.5, 0.98, box_text,
        transform=ax.transAxes, va="top", ha="center", multialignment="left",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.9),
        fontsize=BOX_FS,
    )

    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {output_png}")


def _combined_histogram(nox_pos, area_pos, nox_neg, area_neg,
                        output_png, year_label):
    has_pos = nox_pos.size > 0
    has_neg = nox_neg.size > 0
    if not has_pos and not has_neg:
        print(f"[{year_label} Combined] No data, skipping.")
        return

    all_nox = np.concatenate([v for v in [nox_pos, nox_neg] if v.size > 0])
    edges = np.histogram_bin_edges(all_nox, bins=BINS)
    centers = (edges[:-1] + edges[1:]) / 2

    fig, ax = plt.subplots(figsize=(8, 6))

    if has_pos:
        h_pos, _ = np.histogram(nox_pos, bins=edges, weights=area_pos)
        ax.bar(centers, h_pos, width=np.diff(edges),
               edgecolor="black", color="#1f77b4", alpha=0.7)

    if has_neg:
        h_neg, _ = np.histogram(nox_neg, bins=edges, weights=area_neg)
        ax.bar(centers, h_neg, width=np.diff(edges),
               edgecolor="black", color="#f0c929", alpha=0.7)

    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ymax_round = np.ceil(ax.get_ylim()[1] * 2) / 2
    ax.set_ylim(0, ymax_round * 1.05)

    ax.set_xlabel("NOx emission (g·m⁻²·yr⁻¹)", fontsize=LABEL_FS)
    ax.set_ylabel("Area (×10⁶ km²)", fontsize=LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.set_title(
        f"{year_label} Chlor-NOx Correlated Area "
        f"vs NOx Emission Rate (R² ≥ {R2_THRESHOLD})",
        fontsize=TITLE_FS,
    )

    pos_area = float(np.nansum(area_pos)) if has_pos else 0.0
    neg_area = float(np.nansum(area_neg)) if has_neg else 0.0

    handles = []
    if has_pos:
        handles.append(Patch(
            facecolor="#1f77b4", edgecolor="black",
            label=f"Positive: {int(nox_pos.size)} pixels, {pos_area:.2f} × 10⁶ km²",
        ))
    if has_neg:
        handles.append(Patch(
            facecolor="#f0c929", edgecolor="black",
            label=f"Negative: {int(nox_neg.size)} pixels, {neg_area:.2f} × 10⁶ km²",
        ))

    ax.legend(
        handles=handles, loc="upper center", fontsize=BOX_FS,
        framealpha=0.9, edgecolor="black",
        fancybox=True, borderpad=0.6,
    )
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {output_png}")


# Main
def main():
    for d in [OUTPUT_DIR_POS, OUTPUT_DIR_NEG, OUTPUT_DIR_BOTH]:
        os.makedirs(d, exist_ok=True)

    chlor_files = sorted(glob.glob(os.path.join(CHLOR_DIR, "chlor_*.tif")))

    pooled_pos_nox, pooled_pos_area = [], []
    pooled_neg_nox, pooled_neg_area = [], []

    for chlor_path in chlor_files:
        match = re.search(r"chlor_(\d{4})\.tif$", os.path.basename(chlor_path))
        if not match:
            continue
        year = match.group(1)

        emission_path = os.path.join(EMISSION_DIR, f"NO_{year}_yearly.tif")
        if not os.path.exists(emission_path):
            print(f"[Skip] {emission_path} not found")
            continue

        print(f"Processing {year} ...")

        df = moving_window_regression(chlor_path, emission_path)
        if df.empty:
            print(f"[{year}] No regression results, skipping.")
            continue

        df = df[df["R2"] >= R2_THRESHOLD]
        if df.empty:
            print(f"[{year}] No pixels pass R² threshold, skipping.")
            continue

        with rasterio.open(emission_path) as src:
            xres, yres = src.res

        df_pos = df[df["Slope"] > 0]
        df_neg = df[df["Slope"] < 0]

        nox_pos, area_pos = extract_nox_and_area(df_pos, emission_path, xres, yres)
        nox_neg, area_neg = extract_nox_and_area(df_neg, emission_path, xres, yres)

        _single_histogram(
            nox_pos, area_pos,
            os.path.join(OUTPUT_DIR_POS, f"nox_histogram_{year}_pos.png"),
            year, color="#1f77b4", sign_label="Positive",
        )
        _single_histogram(
            nox_neg, area_neg,
            os.path.join(OUTPUT_DIR_NEG, f"nox_histogram_{year}_neg.png"),
            year, color="#f0c929", sign_label="Negative",
        )
        _combined_histogram(
            nox_pos, area_pos, nox_neg, area_neg,
            os.path.join(OUTPUT_DIR_BOTH, f"nox_histogram_{year}_combined.png"),
            year,
        )

        if nox_pos.size > 0:
            pooled_pos_nox.append(nox_pos)
            pooled_pos_area.append(area_pos)
        if nox_neg.size > 0:
            pooled_neg_nox.append(nox_neg)
            pooled_neg_area.append(area_neg)

    label = "2013-2017"

    p_nox  = np.concatenate(pooled_pos_nox)  if pooled_pos_nox  else np.array([])
    p_area = np.concatenate(pooled_pos_area) if pooled_pos_area else np.array([])
    n_nox  = np.concatenate(pooled_neg_nox)  if pooled_neg_nox  else np.array([])
    n_area = np.concatenate(pooled_neg_area) if pooled_neg_area else np.array([])

    _single_histogram(
        p_nox, p_area,
        os.path.join(OUTPUT_DIR_POS, f"nox_histogram_{label}_pos.png"),
        label, color="#1f77b4", sign_label="Positive",
    )
    _single_histogram(
        n_nox, n_area,
        os.path.join(OUTPUT_DIR_NEG, f"nox_histogram_{label}_neg.png"),
        label, color="#f0c929", sign_label="Negative",
    )
    _combined_histogram(
        p_nox, p_area, n_nox, n_area,
        os.path.join(OUTPUT_DIR_BOTH, f"nox_histogram_{label}_combined.png"),
        label,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
