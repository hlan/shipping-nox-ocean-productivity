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
CHLOR_DIR = "monthly_data/chlor_data_monthly"
NOX_DIR = "monthly_data/nox_monthly"
OUTPUT_ROOT = "static_map_seasonal"

DIR_TMP = os.path.join(OUTPUT_ROOT, "tmp_seasonal_rasters")
DIR_R = os.path.join(OUTPUT_ROOT, "r")
DIR_R2 = os.path.join(OUTPUT_ROOT, "r2")
DIR_SLP = os.path.join(OUTPUT_ROOT, "slope")

for d in [DIR_TMP, DIR_R, DIR_R2, DIR_SLP]:
    os.makedirs(d, exist_ok=True)

YEAR = 2017
WINDOW_SIZE = 11
MIN_SAMPLES = 70
R2_THRESH = 0.6
SLOPE_ABS_MIN = 1e-6
CHLOR_NODATA_VALUE = 99999.0
MIN_VALID_MONTHS_IN_SEASON = 2


def seconds_in_month(y, m):
    return calendar.monthrange(y, m)[1] * 24 * 3600


def chlor_path_for(y, m):
    return os.path.join(CHLOR_DIR, f"MY1DMM_CHLORA_{y}-{m:02d}-01_rgb_720x360.FLOAT.TIFF")


def nox_path_for(y, m):
    return os.path.join(NOX_DIR, f"NOxFlux_{y}_{m:02d}.tif")


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
def perform_moving_window_regression(chlor_path, nox_path, window_size=11):
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
                    nodata_value=None, allow_zero=True)

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


# Seasonal aggregation
def build_season_rasters(target_year, season):
    season = season.lower().strip()
    if season == "summer":
        ym_list = [(target_year, 6), (target_year, 7), (target_year, 8)]
        tag = f"summer_{target_year}"
    elif season == "winter":
        ym_list = [(target_year - 1, 12), (target_year, 1), (target_year, 2)]
        tag = f"winter_{target_year}"
    else:
        raise ValueError("season must be 'summer' or 'winter'")

    ref_chl = None
    for (y, m) in ym_list:
        p = chlor_path_for(y, m)
        if os.path.exists(p):
            ref_chl = p
            break
    if ref_chl is None:
        raise FileNotFoundError(f"No chlor files found for {tag}")

    out_chl = os.path.join(DIR_TMP, f"chlor_{tag}.tif")
    out_nox = os.path.join(DIR_TMP, f"nox_{tag}.tif")

    with rasterio.open(ref_chl) as ref:
        profile = ref.profile.copy()
        profile.update(count=1, dtype="float32", nodata=CHLOR_NODATA_VALUE, compress="lzw")
        H, W = ref.height, ref.width

        chl_sum = np.zeros((H, W), dtype="float64")
        chl_cnt = np.zeros((H, W), dtype="uint8")
        nox_sum_g = np.zeros((H, W), dtype="float64")
        nox_cnt = np.zeros((H, W), dtype="uint8")

        for (y, m) in ym_list:
            chl_p = chlor_path_for(y, m)
            nox_p = nox_path_for(y, m)

            if (not os.path.exists(chl_p)) or (not os.path.exists(nox_p)):
                print(f"[Season {tag}] skip month {y}-{m:02d}: missing file(s)")
                continue

            with rasterio.open(chl_p) as src_c:
                c = src_c.read(1).astype("float64")
            c[c == CHLOR_NODATA_VALUE] = np.nan
            c_valid = np.isfinite(c) & (c > 0)
            chl_sum[c_valid] += c[c_valid]
            chl_cnt[c_valid] += 1

            with rasterio.open(nox_p) as src_e_raw:
                need_warp = (ref.crs != src_e_raw.crs) or \
                            (ref.transform != src_e_raw.transform) or \
                            (ref.shape != src_e_raw.shape)
                if need_warp:
                    src_e = WarpedVRT(src_e_raw, crs=ref.crs, transform=ref.transform,
                                      width=ref.width, height=ref.height,
                                      resampling=Resampling.bilinear)
                else:
                    src_e = src_e_raw
                e = src_e.read(1).astype("float64")
                if need_warp:
                    src_e.close()

            e_nodata = None
            try:
                with rasterio.open(nox_p) as tmp:
                    e_nodata = tmp.nodata
            except Exception:
                pass
            if e_nodata is not None and np.isfinite(e_nodata):
                e[e == e_nodata] = np.nan

            e_valid = np.isfinite(e) & (e >= 0)
            e_g = e * float(seconds_in_month(y, m)) * 1000.0
            nox_sum_g[e_valid] += e_g[e_valid]
            nox_cnt[e_valid] += 1

        chl_out = np.full((H, W), CHLOR_NODATA_VALUE, dtype="float32")
        ok_chl = chl_cnt >= MIN_VALID_MONTHS_IN_SEASON
        chl_out[ok_chl] = (chl_sum[ok_chl] / chl_cnt[ok_chl]).astype("float32")

        NOX_NODATA_OUT = -9999.0
        nox_profile = profile.copy()
        nox_profile.update(nodata=NOX_NODATA_OUT)
        nox_out = np.full((H, W), NOX_NODATA_OUT, dtype="float32")
        ok_nox = nox_cnt >= MIN_VALID_MONTHS_IN_SEASON
        nox_out[ok_nox] = nox_sum_g[ok_nox].astype("float32")

    with rasterio.open(out_chl, "w", **profile) as dst:
        dst.write(chl_out, 1)
    with rasterio.open(out_nox, "w", **nox_profile) as dst:
        dst.write(nox_out, 1)

    print(f"[Season {tag}] saved chlor: {out_chl}")
    print(f"[Season {tag}] saved nox: {out_nox}")

    return out_chl, out_nox


# Main
def main():
    for season in ["summer", "winter"]:
        print(f"\nBuilding season: {season}, year={YEAR}")

        chlor_season_path, nox_season_path = build_season_rasters(YEAR, season)

        with rasterio.open(chlor_season_path) as src:
            xres, yres = src.res

        print(f"Running moving-window regression: {season} {YEAR}")
        df = perform_moving_window_regression(chlor_season_path, nox_season_path, window_size=WINDOW_SIZE)

        if df.empty:
            print(f"[{season} {YEAR}] df empty, skipping.")
            continue

        df = df[df["R2"] >= R2_THRESH].copy()
        if df.empty:
            print(f"[{season} {YEAR}] no pixels after R2 >= {R2_THRESH}")
            continue

        label = f"{season.capitalize()} {YEAR}"

        out_r = os.path.join(DIR_R, f"chlor_nox_{season}_{YEAR}_r.png")
        out_r2 = os.path.join(DIR_R2, f"chlor_nox_{season}_{YEAR}_r2.png")
        out_sl = os.path.join(DIR_SLP, f"chlor_nox_{season}_{YEAR}_slope.png")

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
