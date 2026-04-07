import os
import calendar
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from scipy.stats import linregress
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as patches
from matplotlib.lines import Line2D

# Config
CHLOR_DIR = "data/monthly_data/chlor_data_monthly"
NOX_DIR = "data/monthly_data/nox_monthly"
OUTPUT_ROOT = "static_map_monthly"

DIR_R = os.path.join(OUTPUT_ROOT, "r")
DIR_R2 = os.path.join(OUTPUT_ROOT, "r2")
DIR_SLP = os.path.join(OUTPUT_ROOT, "slope")

for d in [DIR_R, DIR_R2, DIR_SLP]:
    os.makedirs(d, exist_ok=True)

YEAR = 2017
WINDOW_SIZE = 11
MIN_SAMPLES = 70
R2_THRESH = 0.6
SLOPE_ABS_MIN = 1e-6
CHLOR_NODATA_VALUE = 99999.0


def extract_moving_window_values(dataset, row, col, window_size,
                                 nodata_value=None, allow_zero=True, scale_factor=1.0):
    half = window_size // 2
    try:
        win = Window(col - half, row - half, window_size, window_size)
        data = dataset.read(1, window=win).astype("float64").flatten()

        if nodata_value is not None:
            data[data == nodata_value] = np.nan
        if scale_factor != 1.0:
            data = data * scale_factor

        m = np.isfinite(data)
        if allow_zero:
            m = m & (data >= 0)
        else:
            m = m & (data > 0)

        data = data[m]
        if data.size == 0:
            return np.array([])
        return np.log1p(data)
    except Exception:
        return np.array([])


# Moving-window regression
def perform_moving_window_regression(chlor_path, nox_path, window_size=11, nox_scale_factor=1.0):
    results = []

    with rasterio.open(chlor_path) as src_c, rasterio.open(nox_path) as src_e_raw:
        need_warp = (src_c.crs != src_e_raw.crs) or \
                    (src_c.transform != src_e_raw.transform) or \
                    (src_c.shape != src_e_raw.shape)

        if need_warp:
            src_e = WarpedVRT(src_e_raw, crs=src_c.crs, transform=src_c.transform,
                              width=src_c.width, height=src_c.height,
                              resampling=Resampling.bilinear)
        else:
            src_e = src_e_raw

        rows, cols = src_c.shape
        tf = src_c.transform

        for i in range(rows):
            for j in range(cols):
                a_vals = extract_moving_window_values(
                    src_c, i, j, window_size,
                    nodata_value=CHLOR_NODATA_VALUE, allow_zero=False)
                b_vals = extract_moving_window_values(
                    src_e, i, j, window_size,
                    nodata_value=None, allow_zero=True, scale_factor=nox_scale_factor)

                if min(len(a_vals), len(b_vals)) < MIN_SAMPLES:
                    continue

                n = min(len(a_vals), len(b_vals))
                a_vals, b_vals = a_vals[:n], b_vals[:n]

                if np.all(a_vals == a_vals[0]):
                    continue

                slope, intercept, r_value, p_value, _ = linregress(a_vals, b_vals)

                if (not np.isfinite(slope)) or (abs(slope) < SLOPE_ABS_MIN):
                    continue

                lon, lat = tf * (j + 0.5, i + 0.5)
                results.append({
                    "Longitude": lon, "Latitude": lat,
                    "Slope": slope, "R": r_value, "R2": r_value**2, "Count": n
                })

        if need_warp:
            src_e.close()

    return pd.DataFrame(results)


# Static maps
def _base_map():
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='white')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    ax.plot([-180, 180], [0, 0], color="green", linewidth=2.5, transform=ccrs.PlateCarree())
    return fig, ax


def visualize_static_map_with_r2(df, output_png, title, xres, yres):
    fig, ax = _base_map()
    half_lon, half_lat = xres / 2, yres / 2

    def classify_color(slope, r2):
        if slope > 0:
            return "#ffcccc" if r2 < 0.7 else "#ff9999" if r2 < 0.8 else "#ff5555" if r2 < 0.9 else "#ff0000"
        else:
            return "#ccccff" if r2 < 0.7 else "#9999ff" if r2 < 0.8 else "#5555ff" if r2 < 0.9 else "#0000ff"

    for _, row in df.iterrows():
        c = classify_color(row["Slope"], row["R2"])
        ax.add_patch(patches.Rectangle(
            (row["Longitude"] - half_lon, row["Latitude"] - half_lat), xres, yres,
            linewidth=0.3, edgecolor=c, facecolor=c, alpha=0.6, transform=ccrs.PlateCarree()))

    legend_handles = [
        Line2D([0], [0], marker='s', color='#ffcccc', label='slope>0, R2<0.7', markersize=9, linestyle='None'),
        Line2D([0], [0], marker='s', color='#ff9999', label='slope>0, R2<0.8', markersize=9, linestyle='None'),
        Line2D([0], [0], marker='s', color='#ff5555', label='slope>0, R2<0.9', markersize=9, linestyle='None'),
        Line2D([0], [0], marker='s', color='#ff0000', label='slope>0, R2>=0.9', markersize=9, linestyle='None'),
        Line2D([0], [0], marker='s', color='#ccccff', label='slope<0, R2<0.7', markersize=9, linestyle='None'),
        Line2D([0], [0], marker='s', color='#9999ff', label='slope<0, R2<0.8', markersize=9, linestyle='None'),
        Line2D([0], [0], marker='s', color='#5555ff', label='slope<0, R2<0.9', markersize=9, linestyle='None'),
        Line2D([0], [0], marker='s', color='#0000ff', label='slope<0, R2>=0.9', markersize=9, linestyle='None'),
        Line2D([0], [0], color="green", lw=2.5, label='Equator')
    ]
    ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1), framealpha=1)
    plt.title(title)
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close(fig)


def visualize_static_map_with_r(df, output_png, title, xres, yres):
    fig, ax = _base_map()
    half_lon, half_lat = xres / 2, yres / 2

    edges = np.array([-1.0, -0.9, -0.8, -0.7, -0.6, 0.0, 0.6, 0.7, 0.8, 0.9, 1.0])
    colors = ["#0000cc", "#4040ff", "#8080ff", "#b3b3ff", "#e0e0ff",
              "#ffe0e0", "#ffb3b3", "#ff8080", "#ff4040", "#cc0000"]

    def r_color(r):
        r = float(np.clip(r, -1.0, 1.0 - 1e-12))
        idx = int(np.clip(np.searchsorted(edges, r, side="right") - 1, 0, len(colors) - 1))
        return colors[idx]

    df_use = df[np.isfinite(df["R"])].copy()
    for _, row in df_use.iterrows():
        c = r_color(row["R"])
        ax.add_patch(patches.Rectangle(
            (row["Longitude"] - half_lon, row["Latitude"] - half_lat), xres, yres,
            linewidth=0.3, edgecolor=c, facecolor=c, alpha=0.6, transform=ccrs.PlateCarree()))

    labels = ["[-1.0,-0.9)", "[-0.9,-0.8)", "[-0.8,-0.7)", "[-0.7,-0.6)", "[-0.6,0.0)",
              "[0.0,0.6)", "[0.6,0.7)", "[0.7,0.8)", "[0.8,0.9)", "[0.9,1.0]"]
    legend_handles = [Line2D([0], [0], marker='s', color=colors[i], label=labels[i],
                             markersize=9, linestyle='None') for i in range(10)]
    legend_handles.append(Line2D([0], [0], color="green", lw=2.5, label='Equator'))
    ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1), framealpha=1)
    plt.title(title)
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close(fig)


def visualize_static_map_with_slope(df, output_png, title, xres, yres):
    fig, ax = _base_map()
    half_lon, half_lat = xres / 2, yres / 2

    df_use = df[np.isfinite(df["Slope"])].copy()
    if df_use.empty:
        plt.title(title + " (no data)")
        plt.savefig(output_png, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return

    abs_s = np.abs(df_use["Slope"].values)
    q1, q2, q3 = np.quantile(abs_s, [0.25, 0.50, 0.75])

    neg_colors = ["#0000ff", "#5555ff", "#9999ff", "#ccccff"]
    pos_colors = ["#ffcccc", "#ff9999", "#ff5555", "#ff0000"]

    def slope_color(s):
        s = float(s)
        if s < -q3: return neg_colors[0]
        if s < -q2: return neg_colors[1]
        if s < -q1: return neg_colors[2]
        if s < 0:   return neg_colors[3]
        if s < q1:  return pos_colors[0]
        if s < q2:  return pos_colors[1]
        if s < q3:  return pos_colors[2]
        return pos_colors[3]

    for _, row in df_use.iterrows():
        c = slope_color(row["Slope"])
        ax.add_patch(patches.Rectangle(
            (row["Longitude"] - half_lon, row["Latitude"] - half_lat), xres, yres,
            linewidth=0.3, edgecolor=c, facecolor=c, alpha=0.6, transform=ccrs.PlateCarree()))

    labels = [
        f"slope <= {-q3:.3g}", f"{-q3:.3g} < slope <= {-q2:.3g}",
        f"{-q2:.3g} < slope <= {-q1:.3g}", f"{-q1:.3g} < slope <= 0",
        f"0 <= slope < {q1:.3g}", f"{q1:.3g} <= slope < {q2:.3g}",
        f"{q2:.3g} <= slope < {q3:.3g}", f"slope >= {q3:.3g}"
    ]
    all_colors = neg_colors + pos_colors
    legend_handles = [Line2D([0], [0], marker='s', color=all_colors[i], label=labels[i],
                             markersize=9, linestyle='None') for i in range(8)]
    legend_handles.append(Line2D([0], [0], color="green", lw=2.5, label='Equator'))
    ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1), framealpha=1)
    plt.title(title)
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close(fig)


# Main
def main():
    for month in range(1, 13):
        mm = f"{month:02d}"

        chlor_path = os.path.join(CHLOR_DIR, f"MY1DMM_CHLORA_{YEAR}-{mm}-01_rgb_720x360.FLOAT.TIFF")
        nox_path = os.path.join(NOX_DIR, f"NOxFlux_{YEAR}_{mm}.tif")

        if not os.path.exists(chlor_path):
            print(f"[Skip] missing chlor: {chlor_path}")
            continue
        if not os.path.exists(nox_path):
            print(f"[Skip] missing NOx: {nox_path}")
            continue

        print(f"\nProcessing {YEAR}-{mm}")

        seconds_this_month = calendar.monthrange(YEAR, month)[1] * 24 * 3600
        nox_scale = 1000.0 * seconds_this_month  # kg m-2 s-1 -> g m-2 month-1

        with rasterio.open(chlor_path) as src:
            xres, yres = src.res

        df = perform_moving_window_regression(chlor_path, nox_path,
                                              window_size=WINDOW_SIZE,
                                              nox_scale_factor=nox_scale)

        if df.empty:
            print(f"[{YEAR}-{mm}] df empty, skipping.")
            continue

        df = df[df["R2"] >= R2_THRESH].copy()
        if df.empty:
            print(f"[{YEAR}-{mm}] no pixels after R2 >= {R2_THRESH}")
            continue

        label = f"{YEAR}-{mm}"

        out_r = os.path.join(DIR_R, f"chlor_nox_{YEAR}_{mm}_r.png")
        out_r2 = os.path.join(DIR_R2, f"chlor_nox_{YEAR}_{mm}_r2.png")
        out_sl = os.path.join(DIR_SLP, f"chlor_nox_{YEAR}_{mm}_slope.png")

        visualize_static_map_with_r(df, out_r,
            f"Chlor-NOx Correlation (r), {label}, R2 >= {R2_THRESH}", xres, yres)
        visualize_static_map_with_r2(df, out_r2,
            f"Chlor-NOx Association (R2), {label}, R2 >= {R2_THRESH}", xres, yres)
        visualize_static_map_with_slope(df, out_sl,
            f"Chlor-NOx Regression Slope, {label}, R2 >= {R2_THRESH}", xres, yres)

        print(f"[Saved] {out_r}")
        print(f"[Saved] {out_r2}")
        print(f"[Saved] {out_sl}")


if __name__ == "__main__":
    main()
