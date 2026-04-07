import os
import re
import glob
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy.stats import linregress, gaussian_kde
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker

R = 6_371_000

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10,
    "axes.unicode_minus": False,
})

OUT_DIR = "fig_2a_output"
os.makedirs(OUT_DIR, exist_ok=True)

NMEAN_CSV = "data/Nmean_60m_data_updated.csv"


def pixel_area_m2(lat_deg, xres_deg, yres_deg):
    lat = np.deg2rad(lat_deg)
    dlat = np.deg2rad(yres_deg)
    dlon = np.deg2rad(xres_deg)
    lat1 = lat - dlat / 2
    lat2 = lat + dlat / 2
    return R**2 * dlon * (np.sin(lat2) - np.sin(lat1))


def log_transform(values):
    return np.log1p(values) if len(values) > 0 else np.array([])


def extract_moving_window_values(dataset, row, col, window_size):
    half = window_size // 2
    try:
        win = Window(col - half, row - half, window_size, window_size)
        data = dataset.read(1, window=win).flatten()
        data = data[np.isfinite(data) & (data > 0)]
        return log_transform(data)
    except Exception:
        return np.array([])


def perform_moving_window_regression(chlor_raster_path, emission_raster_path, window_size=11):
    results = []
    with rasterio.open(chlor_raster_path) as src_c, rasterio.open(emission_raster_path) as src_e:
        assert src_c.transform == src_e.transform, "Resolution/CRS mismatch"
        rows, cols = src_c.shape
        tf = src_c.transform

        for i in range(rows):
            for j in range(cols):
                a_vals = extract_moving_window_values(src_c, i, j, window_size)
                b_vals = extract_moving_window_values(src_e, i, j, window_size)

                if min(len(a_vals), len(b_vals)) < 70:
                    continue
                n = min(len(a_vals), len(b_vals))
                a_vals, b_vals = a_vals[:n], b_vals[:n]

                if np.all(a_vals == a_vals[0]):
                    continue

                slope, intercept, r_value, p_value, _ = linregress(a_vals, b_vals)
                if abs(slope) < 0.01:
                    continue

                lon, lat = tf * (j + 0.5, i + 0.5)
                results.append({
                    "Longitude": lon,
                    "Latitude": lat,
                    "Slope": slope,
                    "R2": r_value**2,
                    "P_Value": p_value,
                    "Count": n
                })

    return pd.DataFrame(results)


def get_nmean_values_and_area_weights(df, nmean_csv_path, xres, yres):
    nmean_df = pd.read_csv(nmean_csv_path)
    lon_centers = np.sort(nmean_df['Longitude'].unique())
    lat_centers = np.sort(nmean_df['Latitude'].unique())

    def nearest_center(vals, centers):
        idx = np.searchsorted(centers, vals)
        idx0 = np.clip(idx - 1, 0, len(centers) - 1)
        idx1 = np.clip(idx, 0, len(centers) - 1)
        pick0 = np.abs(vals - centers[idx0]) <= np.abs(vals - centers[idx1])
        return np.where(pick0, centers[idx0], centers[idx1])

    df_loc = df[['Longitude', 'Latitude']].dropna().copy()
    if df_loc.empty:
        return np.array([]), np.array([])

    df_loc['lon_c'] = nearest_center(df_loc['Longitude'].values, lon_centers)
    df_loc['lat_c'] = nearest_center(df_loc['Latitude'].values, lat_centers)

    nmean_idx = nmean_df.set_index(['Longitude', 'Latitude'])
    keys = list(zip(df_loc['lon_c'].values, df_loc['lat_c'].values))
    df_loc['mean_N_Depth_60'] = nmean_idx.reindex(keys)['mean_N_Depth_60'].values

    df_valid = df_loc[np.isfinite(df_loc['mean_N_Depth_60'].values)]
    if df_valid.empty:
        return np.array([]), np.array([])

    vals = df_valid['mean_N_Depth_60'].values.astype(float)
    w_area = np.array(
        [pixel_area_m2(lat, xres, yres) / 1e12 for lat in df_valid['Latitude']],
        dtype=float
    )
    return vals, w_area


def plot_2017_threegroup_density(df_all, nmean_csv_path, xres, yres, output_png,
                                 alpha_sig=0.05, r2_min=0.6, low_threshold=2.0,
                                 bw_method="scott", fill_alpha=0.18, xlim_max=8.0):
    df_all = df_all[np.isfinite(df_all["Slope"]) & np.isfinite(df_all["P_Value"])].copy()
    df_base = df_all[df_all["R2"] >= r2_min].copy()

    df_pos = df_base[(df_base["Slope"] > 0) & (df_base["P_Value"] < alpha_sig)]
    df_neg = df_base[(df_base["Slope"] < 0) & (df_base["P_Value"] < alpha_sig)]

    vals_all, w_all = get_nmean_values_and_area_weights(df_all, nmean_csv_path, xres, yres)
    vals_pos, w_pos = get_nmean_values_and_area_weights(df_pos, nmean_csv_path, xres, yres)
    vals_neg, w_neg = get_nmean_values_and_area_weights(df_neg, nmean_csv_path, xres, yres)

    x = np.linspace(0.0, xlim_max, 800)

    fig, ax = plt.subplots(figsize=(8, 6))
    max_y = 0.0

    groups = [
        ("Significantly Positive (p<0.05)", vals_pos, w_pos, "#ff7f0e", "-"),
        ("All grids", vals_all, w_all, "#4d4d4d", "--"),
        ("Significantly Negative (p<0.05)", vals_neg, w_neg, "#1f77b4", "-"),
    ]

    for label, vals, w, color, ls in groups:
        if vals.size < 5:
            continue
        m = np.isfinite(vals) & np.isfinite(w) & (w > 0)
        vals, w = vals[m], w[m]
        w_norm = w / np.sum(w)

        kde = gaussian_kde(vals, weights=w_norm, bw_method=bw_method)
        y = kde(x)
        max_y = max(max_y, np.nanmax(y))

        ax.plot(x, y, color=color, linewidth=1.6, linestyle=ls, label=label)
        ax.fill_between(x, 0, y, color=color, alpha=fill_alpha)

    ax.axvline(low_threshold, color="red", linestyle="--", linewidth=2)
    ymax = max_y if max_y > 0 else ax.get_ylim()[1]
    ax.text(
        low_threshold + 0.12, ymax * 0.92,
        "N limited regions (<2 umol/kg)",
        color="red", fontsize=10, rotation=90, ha="left", va="top"
    )

    ax.set_title(f"2017 distribution of mean N at 60 m by correlation group (p<{alpha_sig} & R2>={r2_min})")
    ax.set_xlabel("Mean N concentration at 60 meters (umol/kg)")
    ax.set_ylabel("Normalized area-weighted density")
    ax.set_xlim(0, xlim_max)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    if max_y > 0:
        ax.set_ylim(0, max_y * 1.05)

    ax.legend(frameon=True, loc="best")
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    chlor_path = "data/chlor/chlor_2017.tif"
    emission_path = "data/emission/NO_2017_yearly.tif"

    print("Processing 2017 ...")

    with rasterio.open(chlor_path) as src:
        xres, yres = src.res

    df = perform_moving_window_regression(chlor_path, emission_path, window_size=11)

    output_png = os.path.join(OUT_DIR, "nmean_density_2017_3groups.png")
    plot_2017_threegroup_density(df, NMEAN_CSV, xres, yres, output_png)
    print(f"[Saved] {output_png}")
