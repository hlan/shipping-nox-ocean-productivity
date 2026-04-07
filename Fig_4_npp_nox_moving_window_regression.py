import os
import re
import glob
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy.stats import linregress
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import geopandas as gpd
from shapely.geometry import box

R = 6_371_000

OUTPUT_ROOT = "npp_nox_result"
DIR_STATIC_R2 = os.path.join(OUTPUT_ROOT, "static_map")
DIR_STATIC_R = os.path.join(OUTPUT_ROOT, "static_map_r")
DIR_STATIC_SLOPE = os.path.join(OUTPUT_ROOT, "static_map_slope")
DIR_SHP = os.path.join(OUTPUT_ROOT, "shp")


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


# Moving-window regression
def perform_moving_window_regression(npp_raster_path, emission_raster_path, window_size=11):
    results = []
    with rasterio.open(npp_raster_path) as src_c, rasterio.open(emission_raster_path) as src_e:
        assert src_c.transform == src_e.transform, "Resolution or CRS mismatch"
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
                    "Intercept": intercept,
                    "Slope": slope,
                    "R": r_value,
                    "R2": r_value**2,
                    "P_Value": p_value,
                    "Count": n
                })

    return pd.DataFrame(results)


# Static maps
def visualize_static_map_with_regression(df, output_png, title, xres, yres):
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
    ax.plot([-180, 180], [0, 0], color="green", linewidth=3, transform=ccrs.PlateCarree())

    half_lon = xres / 2
    half_lat = yres / 2

    def classify_color(slope, r2):
        if slope > 0:
            return "#ffcccc" if r2 < 0.7 else "#ff9999" if r2 < 0.8 else "#ff5555" if r2 < 0.9 else "#ff0000"
        else:
            return "#ccccff" if r2 < 0.7 else "#9999ff" if r2 < 0.8 else "#5555ff" if r2 < 0.9 else "#0000ff"

    for _, row in df.iterrows():
        if pd.notna(row["R2"]) and row["R2"] >= 0.6:
            c = classify_color(row["Slope"], row["R2"])
            rect = patches.Rectangle(
                (row["Longitude"] - half_lon, row["Latitude"] - half_lat), xres, yres,
                linewidth=0.5, edgecolor=c, facecolor=c, alpha=0.6,
                transform=ccrs.PlateCarree()
            )
            ax.add_patch(rect)

    legend_handles = [
        Line2D([0], [0], marker='s', color='#ffcccc', label='R2 < 0.7', markersize=10, linestyle='None'),
        Line2D([0], [0], marker='s', color='#ff9999', label='R2 < 0.8', markersize=10, linestyle='None'),
        Line2D([0], [0], marker='s', color='#ff5555', label='R2 < 0.9', markersize=10, linestyle='None'),
        Line2D([0], [0], marker='s', color='#ff0000', label='R2 >= 0.9', markersize=10, linestyle='None'),
        Line2D([0], [0], marker='s', color='#ccccff', label='R2 < 0.7 (neg)', markersize=10, linestyle='None'),
        Line2D([0], [0], marker='s', color='#9999ff', label='R2 < 0.8 (neg)', markersize=10, linestyle='None'),
        Line2D([0], [0], marker='s', color='#5555ff', label='R2 < 0.9 (neg)', markersize=10, linestyle='None'),
        Line2D([0], [0], marker='s', color='#0000ff', label='R2 >= 0.9 (neg)', markersize=10, linestyle='None'),
        Line2D([0], [0], color="green", lw=3, label="Equator line")
    ]
    ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.05, 1), framealpha=1)
    plt.title(title)
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close(fig)


def visualize_static_map_with_r(df, output_png, title, xres, yres):
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
    ax.plot([-180, 180], [0, 0], color="green", linewidth=3, transform=ccrs.PlateCarree())

    half_lon = xres / 2
    half_lat = yres / 2

    def classify_color_by_r(r):
        ar = abs(r)
        pos_colors = ["#ffe0e0", "#ffb3b3", "#ff8080", "#ff4040", "#cc0000"]
        neg_colors = ["#e0e0ff", "#b3b3ff", "#8080ff", "#4040ff", "#0000cc"]
        if ar < 0.6:
            idx = 0
        elif ar < 0.7:
            idx = 1
        elif ar < 0.8:
            idx = 2
        elif ar < 0.9:
            idx = 3
        else:
            idx = 4
        return pos_colors[idx] if r >= 0 else neg_colors[idx]

    for _, row in df.iterrows():
        if pd.notna(row["R2"]) and row["R2"] >= 0.6 and pd.notna(row["R"]):
            c = classify_color_by_r(row["R"])
            rect = patches.Rectangle(
                (row["Longitude"] - half_lon, row["Latitude"] - half_lat), xres, yres,
                linewidth=0.5, edgecolor=c, facecolor=c, alpha=0.6,
                transform=ccrs.PlateCarree()
            )
            ax.add_patch(rect)

    legend_handles = [
        Line2D([0], [0], marker='s', color="#ffe0e0", label='0 <= r < 0.6', markersize=10, linestyle='None'),
        Line2D([0], [0], marker='s', color="#ffb3b3", label='0.6 <= r < 0.7', markersize=10, linestyle='None'),
        Line2D([0], [0], marker='s', color="#ff8080", label='0.7 <= r < 0.8', markersize=10, linestyle='None'),
        Line2D([0], [0], marker='s', color="#ff4040", label='0.8 <= r < 0.9', markersize=10, linestyle='None'),
        Line2D([0], [0], marker='s', color="#cc0000", label='r >= 0.9', markersize=10, linestyle='None'),
        Line2D([0], [0], marker='s', color="#e0e0ff", label='-0.6 < r < 0', markersize=10, linestyle='None'),
        Line2D([0], [0], marker='s', color="#b3b3ff", label='-0.7 < r <= -0.6', markersize=10, linestyle='None'),
        Line2D([0], [0], marker='s', color="#8080ff", label='-0.8 < r <= -0.7', markersize=10, linestyle='None'),
        Line2D([0], [0], marker='s', color="#4040ff", label='-0.9 < r <= -0.8', markersize=10, linestyle='None'),
        Line2D([0], [0], marker='s', color="#0000cc", label='r <= -0.9', markersize=10, linestyle='None'),
        Line2D([0], [0], color="green", lw=3, label="Equator line")
    ]
    ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.05, 1), framealpha=1)
    plt.title(title)
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close(fig)


def visualize_static_map_with_slope(df, output_png, title, xres, yres):
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
    ax.plot([-180, 180], [0, 0], color="green", linewidth=3, transform=ccrs.PlateCarree())

    half_lon = xres / 2
    half_lat = yres / 2

    df_use = df[(df["R2"] >= 0.6) & np.isfinite(df["Slope"])].copy()
    if df_use.empty:
        plt.title(title + " (no data)")
        plt.savefig(output_png, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return

    abs_s = np.abs(df_use["Slope"].values)
    q1, q2, q3 = np.quantile(abs_s, [0.25, 0.50, 0.75])

    def slope_color(s):
        a = abs(s)
        if s >= 0:
            return "#ffcccc" if a < q1 else "#ff9999" if a < q2 else "#ff5555" if a < q3 else "#ff0000"
        else:
            return "#ccccff" if a < q1 else "#9999ff" if a < q2 else "#5555ff" if a < q3 else "#0000ff"

    for _, row in df_use.iterrows():
        c = slope_color(row["Slope"])
        rect = patches.Rectangle(
            (row["Longitude"] - half_lon, row["Latitude"] - half_lat), xres, yres,
            linewidth=0.5, edgecolor=c, facecolor=c, alpha=0.6,
            transform=ccrs.PlateCarree()
        )
        ax.add_patch(rect)

    legend_handles = [
        Line2D([0], [0], marker='s', color='#ffcccc', label=f'0 <= slope < {q1:.4g}', markersize=10, linestyle='None'),
        Line2D([0], [0], marker='s', color='#ff9999', label=f'{q1:.4g} <= slope < {q2:.4g}', markersize=10, linestyle='None'),
        Line2D([0], [0], marker='s', color='#ff5555', label=f'{q2:.4g} <= slope < {q3:.4g}', markersize=10, linestyle='None'),
        Line2D([0], [0], marker='s', color='#ff0000', label=f'slope >= {q3:.4g}', markersize=10, linestyle='None'),
        Line2D([0], [0], marker='s', color='#ccccff', label=f'{-q1:.4g} < slope <= 0', markersize=10, linestyle='None'),
        Line2D([0], [0], marker='s', color='#9999ff', label=f'{-q2:.4g} < slope <= {-q1:.4g}', markersize=10, linestyle='None'),
        Line2D([0], [0], marker='s', color='#5555ff', label=f'{-q3:.4g} < slope <= {-q2:.4g}', markersize=10, linestyle='None'),
        Line2D([0], [0], marker='s', color='#0000ff', label=f'slope <= {-q3:.4g}', markersize=10, linestyle='None'),
        Line2D([0], [0], color="green", lw=3, label="Equator line")
    ]
    ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.05, 1), framealpha=1)
    plt.title(title)
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close(fig)


# Main
def main():
    for d in [DIR_STATIC_R2, DIR_STATIC_R, DIR_STATIC_SLOPE, DIR_SHP]:
        os.makedirs(d, exist_ok=True)

    total_areas = []

    for npp_path in sorted(glob.glob("data/npp/npp_yearmean_rate_*.tif")):
        match = re.search(r"npp_yearmean_rate_(\d{4})\.tif$", os.path.basename(npp_path))
        if not match:
            continue

        year = match.group(1)
        emission_path = f"data/emission/NO_{year}_yearly.tif"
        if not os.path.exists(emission_path):
            print(f"[Skip] Emission file not found: {emission_path}")
            continue

        print(f"Processing year {year} ...")

        with rasterio.open(npp_path) as src:
            xres, yres = src.res
            crs = src.crs

        df = perform_moving_window_regression(npp_path, emission_path, window_size=11)

        if df.empty:
            print(f"[Year {year}] Regression df is empty, skipping.")
            continue

        df = df[df['R2'] >= 0.6]

        half_lon, half_lat = xres / 2, yres / 2

        # Shapefiles
        df_pos = df[df['Slope'] >= 0]
        geoms_pos = [box(lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat)
                     for lon, lat in zip(df_pos['Longitude'], df_pos['Latitude'])]
        gdf_pos = gpd.GeoDataFrame(df_pos.copy(), geometry=geoms_pos, crs=crs)
        shp_pos_out = os.path.join(DIR_SHP, f"npp_nox_regression_{year}_pos.shp")
        gdf_pos.to_file(shp_pos_out)
        print(f"  Saved positive slope shapefile: {shp_pos_out}")

        df_neg = df[df['Slope'] < 0]
        geoms_neg = [box(lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat)
                     for lon, lat in zip(df_neg['Longitude'], df_neg['Latitude'])]
        gdf_neg = gpd.GeoDataFrame(df_neg.copy(), geometry=geoms_neg, crs=crs)
        shp_neg_out = os.path.join(DIR_SHP, f"npp_nox_regression_{year}_neg.shp")
        gdf_neg.to_file(shp_neg_out)
        print(f"  Saved negative slope shapefile: {shp_neg_out}")

        df_r_all = df[np.isfinite(df["R"])].copy()
        geoms_r_all = [box(lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat)
                       for lon, lat in zip(df_r_all["Longitude"], df_r_all["Latitude"])]
        gdf_r_all = gpd.GeoDataFrame(df_r_all, geometry=geoms_r_all, crs=crs)
        shp_r_all_out = os.path.join(DIR_SHP, f"npp_nox_regression_{year}_r_all.shp")
        gdf_r_all.to_file(shp_r_all_out)
        print(f"  Saved r shapefile (all): {shp_r_all_out}")

        df_r_pos = df[df['R'] >= 0]
        geoms_r_pos = [box(lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat)
                       for lon, lat in zip(df_r_pos['Longitude'], df_r_pos['Latitude'])]
        gdf_r_pos = gpd.GeoDataFrame(df_r_pos.copy(), geometry=geoms_r_pos, crs=crs)
        shp_r_pos_out = os.path.join(DIR_SHP, f"npp_nox_regression_{year}_r_pos.shp")
        gdf_r_pos.to_file(shp_r_pos_out)
        print(f"  Saved r >= 0 shapefile: {shp_r_pos_out}")

        df_r_neg = df[df['R'] < 0]
        geoms_r_neg = [box(lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat)
                       for lon, lat in zip(df_r_neg['Longitude'], df_r_neg['Latitude'])]
        gdf_r_neg = gpd.GeoDataFrame(df_r_neg.copy(), geometry=geoms_r_neg, crs=crs)
        shp_r_neg_out = os.path.join(DIR_SHP, f"npp_nox_regression_{year}_r_neg.shp")
        gdf_r_neg.to_file(shp_r_neg_out)
        print(f"  Saved r < 0 shapefile: {shp_r_neg_out}")

        # Static maps
        png_r2_out = os.path.join(DIR_STATIC_R2, f"npp_nox_regression_map_{year}.png")
        visualize_static_map_with_regression(
            df, png_r2_out,
            f"NPP-NOx Moving Window Regression {year} (R2, R2 >= 0.6)",
            xres, yres)

        png_r_out = os.path.join(DIR_STATIC_R, f"npp_nox_regression_map_{year}_r.png")
        visualize_static_map_with_r(
            df, png_r_out,
            f"NPP-NOx Moving Window Regression {year} (r, R2 >= 0.6)",
            xres, yres)

        png_slope_out = os.path.join(DIR_STATIC_SLOPE, f"npp_nox_regression_map_{year}_slope.png")
        visualize_static_map_with_slope(
            df, png_slope_out,
            f"NPP-NOx Moving Window Regression {year} (slope, R2 >= 0.6)",
            xres, yres)

        # Area summary
        df_pos_hist = df[df['Slope'] > 0].copy()
        areas_m2 = [pixel_area_m2(lat, xres, yres) for lat in df_pos_hist['Latitude']]
        total_area_km2 = sum(areas_m2) / 1e6
        total_areas.append(total_area_km2)
        print(f"  Year {year}: positive-slope & R2>=0.6 pixel count = {len(df_pos_hist)}, total area = {total_area_km2:.2f} km2")

    if total_areas:
        avg_area = sum(total_areas) / len(total_areas)
        print(f"Average total area across years = {avg_area:.2f} km2")


if __name__ == '__main__':
    main()
