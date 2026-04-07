import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
import os

OUT_DIR = "results_aoi_chlor_nox"
os.makedirs(OUT_DIR, exist_ok=True)


def generate_region_mask(raster_path, geojson_path):
    with rasterio.open(raster_path) as src:
        print(f"Raster size: {src.height} x {src.width}")
        gdf = gpd.read_file(geojson_path)
        print(f"GeoJSON records: {len(gdf)}")

        shapes = [
            geom for geom in gdf.geometry
            if geom.is_valid and geom.geom_type in ["Polygon", "MultiPolygon"]
        ]
        if not shapes:
            raise ValueError("No valid Polygon or MultiPolygon geometries in GeoJSON.")
        print(f"Using {len(shapes)} valid geometries for masking.")

        mask = rasterize(
            [(geom, 1) for geom in shapes],
            out_shape=(src.height, src.width),
            transform=src.transform,
            fill=0,
            dtype="uint8",
            all_touched=True
        ).astype(bool)

        return mask, src.transform


def extract_region_time_series(chl_paths, nox_paths, region_mask):
    all_data = []
    for chl_path, nox_path in zip(chl_paths, nox_paths):
        year = int(chl_path.split("_")[-1].split(".")[0])
        with rasterio.open(chl_path) as chl_src, rasterio.open(nox_path) as nox_src:
            chl = chl_src.read(1)
            nox = nox_src.read(1)

            chl_values = chl[region_mask]
            nox_values = nox[region_mask]

            valid = (chl_values > 0) & (nox_values > 0)
            chl_values = chl_values[valid]
            nox_values = nox_values[valid]

            if len(chl_values) and len(nox_values):
                all_data.extend(
                    {"Year": year, "Chlorophyll": float(c), "NOx": float(n)}
                    for c, n in zip(chl_values, nox_values)
                )

    return pd.DataFrame(all_data)


if __name__ == "__main__":
    aoi_list = [
        ("NW Pacific off Japan (AOI_1)", "aoi/eastern_japan_aoi.geojson"),
        ("Central South Atlantic (AOI_2)", "aoi2_Central_South_Atlantic/Central_South_Atlantic.geojson"),
        ("North Atlantic (AOI_3)", "aoi3/aoi3_North_Atlantic_Ocean.geojson"),
        ("NE Pacific (AOI_4)", "aoi4/aoi4_North_East_Pacific.geojson"),
        ("Eastern Pacific (AOI_5)", "aoi5/aoi5.geojson"),
    ]

    years = range(2013, 2018)
    chl_paths = [f"data/chlor/chlor_{y}.tif" for y in years]
    nox_paths = [f"data/emission/NO_{y}_yearly.tif" for y in years]

    all_dfs = []
    for aoi_name, geojson_path in aoi_list:
        try:
            mask, _ = generate_region_mask(chl_paths[0], geojson_path)
            df_one = extract_region_time_series(chl_paths, nox_paths, mask)
            if not df_one.empty:
                df_one["AOI"] = aoi_name
                all_dfs.append(df_one)
                print(f"[OK] {aoi_name}: extracted {len(df_one)} records.")
            else:
                print(f"[WARN] {aoi_name}: no valid pixels after filtering.")
        except Exception as e:
            print(f"[WARN] {aoi_name} failed: {e}")

    if not all_dfs:
        raise RuntimeError("No AOI data extracted. Check aoi_list paths and data alignment.")

    df_all = pd.concat(all_dfs, ignore_index=True)
    print(f"[DATA] Total records: {len(df_all)}")

    # Global linear fit using AOI 1-4 only
    aoi_fit_names = [aoi_list[i][0] for i in range(4)]
    df_fit = df_all[df_all["AOI"].isin(aoi_fit_names)].copy()

    if df_fit.empty:
        raise RuntimeError("AOI 1-4 data is empty. Check AOI names or extraction results.")

    X_fit = df_fit["NOx"].values.reshape(-1, 1)
    y_fit = df_fit["Chlorophyll"].values

    lr_global = LinearRegression().fit(X_fit, y_fit)
    y_fit_pred = lr_global.predict(X_fit)
    slope = lr_global.coef_[0]
    intercept = lr_global.intercept_
    r2_global = r2_score(y_fit, y_fit_pred)

    X_sm = sm.add_constant(df_fit["NOx"].values)
    ols = sm.OLS(y_fit, X_sm).fit()

    intercept_ols = float(ols.params[0])
    slope_ols = float(ols.params[1])
    ci_params = ols.conf_int(alpha=0.05)
    intercept_ci_low, intercept_ci_high = ci_params[0]
    slope_ci_low, slope_ci_high = ci_params[1]
    slope_ci_halfwidth = (slope_ci_high - slope_ci_low) / 2.0

    print("\n[GLOBAL LINEAR FIT] Using AOI 1-4 (sklearn)")
    print(f"  n              = {len(df_fit)}")
    print(f"  slope          = {slope:.8g}")
    print(f"  intercept      = {intercept:.8g}")
    print(f"  R2             = {r2_global:.4f}")
    print(f"  mean NOx       = {X_fit.mean():.8g}")
    print(f"  mean Chl-a     = {y_fit.mean():.8g}")
    print()

    print("[OLS 95% CI] Using AOI 1-4 (pooled)")
    print(f"  slope (OLS)        = {slope_ols:.8g}")
    print(f"  slope 95% CI       = [{slope_ci_low:.8g}, {slope_ci_high:.8g}]")
    print(f"  slope +/- halfwidth = {slope_ci_halfwidth:.8g}")
    print(f"  intercept (OLS)    = {intercept_ols:.8g}")
    print(f"  intercept 95% CI   = [{intercept_ci_low:.8g}, {intercept_ci_high:.8g}]")
    print()

    # Scatter plot
    x_min = df_all["NOx"].min()
    x_max = df_all["NOx"].max()
    x_line = np.linspace(x_min, x_max, 200)

    X_line_sm = sm.add_constant(x_line)
    pred = ols.get_prediction(X_line_sm).summary_frame(alpha=0.05)
    y_line_ols = pred["mean"].to_numpy()
    y_ci_low = pred["mean_ci_lower"].to_numpy()
    y_ci_high = pred["mean_ci_upper"].to_numpy()

    palette = {
        "NW Pacific off Japan (AOI_1)": "#1f77b4",
        "Central South Atlantic (AOI_2)": "#ff7f0e",
        "North Atlantic (AOI_3)": "#2ca02c",
        "NE Pacific (AOI_4)": "#9467bd",
        "Eastern Pacific (AOI_5)": "#8c564b",
    }

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.linewidth": 1.2,
    })

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df_all, x="NOx", y="Chlorophyll",
        hue="AOI", palette=palette, alpha=0.6, s=12
    )
    plt.fill_between(x_line, y_ci_low, y_ci_high, alpha=0.2, label="95% CI (linear fit)")
    plt.plot(x_line, y_line_ols, color="red", linewidth=2.5, label="Linear fit using AOI 1-4 data")

    plt.title("NOx Emissions - Chlorophyll a Relationship Across All AOIs (2013-2017)")
    plt.xlabel("NOx Emissions (g m-2 yr-1)")
    plt.ylabel("Chlorophyll a Concentrations (mg m-3)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    scatter_path = os.path.join(OUT_DIR, "nox_chlor_scatter_by_aoi_with_fit.png")
    plt.savefig(scatter_path, dpi=400)
    plt.show()
    print(f"[Saved] Scatter plot: {scatter_path}")
