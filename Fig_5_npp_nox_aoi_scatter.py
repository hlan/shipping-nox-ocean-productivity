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

OUT_DIR = "results_npp_nox_aoi"
os.makedirs(OUT_DIR, exist_ok=True)


def generate_region_mask(raster_path, geojson_path):
    with rasterio.open(raster_path) as src:
        gdf = gpd.read_file(geojson_path)
        shapes = [
            geom for geom in gdf.geometry
            if geom.is_valid and geom.geom_type in ["Polygon", "MultiPolygon"]
        ]
        if not shapes:
            raise ValueError("No valid Polygon or MultiPolygon geometries in GeoJSON.")

        mask = rasterize(
            [(geom, 1) for geom in shapes],
            out_shape=(src.height, src.width),
            transform=src.transform,
            fill=0,
            dtype="uint8",
            all_touched=True
        ).astype(bool)

        return mask, src.transform


def extract_region_time_series(npp_paths, nox_paths, region_mask):
    all_data = []
    for npp_path, nox_path in zip(npp_paths, nox_paths):
        year = int(os.path.basename(npp_path).split("_")[-1].split(".")[0])
        with rasterio.open(npp_path) as npp_src, rasterio.open(nox_path) as nox_src:
            npp = npp_src.read(1)
            nox = nox_src.read(1)

            npp_values = npp[region_mask]
            nox_values = nox[region_mask]

            valid = (npp_values > 0) & (nox_values > 0)
            npp_values = npp_values[valid]
            nox_values = nox_values[valid]

            if len(npp_values) and len(nox_values):
                all_data.extend(
                    {"Year": year, "NPP": float(c), "NOx": float(n)}
                    for c, n in zip(npp_values, nox_values)
                )

    return pd.DataFrame(all_data)


if __name__ == "__main__":
    aoi_list = [
        ("NW Pacific off Japan (AOI 1)", "aoi/eastern_japan_aoi.geojson"),
        ("Central South Atlantic (AOI 2)", "aoi2_Central_South_Atlantic/Central_South_Atlantic.geojson"),
        ("North Atlantic (AOI 3)", "aoi3/aoi3_North_Atlantic_Ocean.geojson"),
        ("NE Pacific (AOI 4)", "aoi4/aoi4_North_East_Pacific.geojson"),
        ("Eastern Pacific (AOI 5)", "aoi5/aoi5.geojson"),
    ]

    years = range(2013, 2018)
    npp_paths = [f"data/npp/npp_yearmean_rate_{y}.tif" for y in years]
    nox_paths = [f"data/emission/NO_{y}_yearly.tif" for y in years]

    all_dfs = []
    for aoi_name, geojson_path in aoi_list:
        try:
            mask, _ = generate_region_mask(npp_paths[0], geojson_path)
            df_one = extract_region_time_series(npp_paths, nox_paths, mask)
            if not df_one.empty:
                df_one["AOI"] = aoi_name
                all_dfs.append(df_one)
            else:
                print(f"[WARN] {aoi_name}: no valid pixels after filtering.")
        except Exception as e:
            print(f"[WARN] {aoi_name} failed: {e}")

    if not all_dfs:
        raise RuntimeError("No AOI data extracted. Check aoi_list paths and data alignment.")

    df_all = pd.concat(all_dfs, ignore_index=True)

    # Global linear fit using AOI 1-4
    aoi_fit_names = [aoi_list[i][0] for i in range(4)]
    df_fit = df_all[df_all["AOI"].isin(aoi_fit_names)].copy()

    if df_fit.empty:
        raise RuntimeError("AOI 1-4 data is empty. Check AOI names or extraction results.")

    X_fit = df_fit["NOx"].values.reshape(-1, 1)
    y_fit = df_fit["NPP"].values

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
    print(f"  mean NPP       = {y_fit.mean():.8g}")
    print()

    print("[OLS 95% CI] Using AOI 1-4 (pooled)")
    print(f"  slope (OLS)        = {slope_ols:.8g}")
    print(f"  slope 95% CI       = [{slope_ci_low:.8g}, {slope_ci_high:.8g}]")
    print(f"  slope +/- halfwidth = {slope_ci_halfwidth:.8g}")
    print(f"  intercept (OLS)    = {intercept_ols:.8g}")
    print(f"  intercept 95% CI   = [{intercept_ci_low:.8g}, {intercept_ci_high:.8g}]")
    print()

    # AOI 5 linear fit + CI
    df_aoi5 = df_all[df_all["AOI"] == aoi_list[4][0]].copy()
    if df_aoi5.empty:
        print("[WARN] AOI 5 data is empty, skipping AOI 5 CI.\n")
    else:
        X5 = df_aoi5["NOx"].values.reshape(-1, 1)
        y5 = df_aoi5["NPP"].values

        lr5 = LinearRegression().fit(X5, y5)
        y5_pred = lr5.predict(X5)
        aoi5_slope = float(lr5.coef_[0])
        aoi5_intercept = float(lr5.intercept_)
        aoi5_r2 = float(r2_score(y5, y5_pred))

        X5_sm = sm.add_constant(df_aoi5["NOx"].values)
        ols5 = sm.OLS(y5, X5_sm).fit()
        ci5 = ols5.conf_int(alpha=0.05)
        aoi5_ci_low, aoi5_ci_high = float(ci5[1, 0]), float(ci5[1, 1])
        aoi5_hw = (aoi5_ci_high - aoi5_ci_low) / 2.0

        print("[AOI 5 LINEAR FIT + 95% CI] (2013-2017 pooled)")
        print(f"  n                  = {len(df_aoi5)}")
        print(f"  slope              = {aoi5_slope:.8g}")
        print(f"  intercept          = {aoi5_intercept:.8g}")
        print(f"  R2                 = {aoi5_r2:.4f}")
        print(f"  slope 95% CI       = [{aoi5_ci_low:.8g}, {aoi5_ci_high:.8g}]")
        print(f"  slope +/- halfwidth = {aoi5_hw:.8g}\n")

    # Scatter plot
    x_min = df_all["NOx"].min()
    x_max = df_all["NOx"].max()
    x_line = np.linspace(x_min, x_max, 200)

    X_line_sm = sm.add_constant(x_line)
    pred = ols.get_prediction(X_line_sm).summary_frame(alpha=0.05)
    y_line_ols = pred["mean"].to_numpy()

    palette = {
        "NW Pacific off Japan (AOI 1)": "#1f77b4",
        "Central South Atlantic (AOI 2)": "#ff7f0e",
        "North Atlantic (AOI 3)": "#2ca02c",
        "NE Pacific (AOI 4)": "#9467bd",
        "Eastern Pacific (AOI 5)": "#8c564b",
    }

    plt.rcParams.update({
        "font.family": "sans-serif",
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
        data=df_all, x="NOx", y="NPP",
        hue="AOI", palette=palette, alpha=0.6, s=12
    )
    plt.plot(x_line, y_line_ols, color="red", linewidth=2.5, label="Linear fit using AOI 1-4 data")

    plt.title("NPP - NOx Emissions Relationship Across All AOIs (2013-2017)")
    plt.xlabel("NOx Emissions (g m-2 yr-1)")
    plt.ylabel("NPP (g C m-2 yr-1)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    scatter_path = os.path.join(OUT_DIR, "nox_npp_scatter_by_aoi_with_fit.png")
    plt.savefig(scatter_path, dpi=400)
    plt.show()
    print(f"[Saved] {scatter_path}")
