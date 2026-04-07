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

OUT_DIR = "results_each_aoi_chlor_nox"
os.makedirs(OUT_DIR, exist_ok=True)

GLOBAL_SLOPE = 0.81947275
GLOBAL_INTERCEPT = 0.03710177


def ols_slope_ci(x, y, alpha=0.05):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    intercept = float(model.params[0])
    slope = float(model.params[1])
    ci = model.conf_int(alpha=alpha)
    slope_ci_low, slope_ci_high = float(ci[1, 0]), float(ci[1, 1])
    slope_ci_halfwidth = (slope_ci_high - slope_ci_low) / 2.0
    return slope, intercept, slope_ci_low, slope_ci_high, slope_ci_halfwidth


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


def fit_piecewise_linear_hinge(x, y, n_breaks=50, min_frac=0.05):
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y)
    x_min, x_max = x.min(), x.max()

    margin = (x_max - x_min) * min_frac
    candidate_x0 = np.linspace(x_min + margin, x_max - margin, n_breaks)

    best_r2 = -np.inf
    best_params = None

    for x0 in candidate_x0:
        h = np.maximum(0, x - x0)
        X_design = np.hstack([x, h])
        lr = LinearRegression().fit(X_design, y)
        y_pred = lr.predict(X_design)
        r2 = r2_score(y, y_pred)

        if r2 > best_r2:
            best_r2 = r2
            a = lr.intercept_
            b1, b_delta = lr.coef_
            best_params = {
                "x0": float(x0),
                "a": float(a),
                "slope1": float(b1),
                "slope2": float(b1 + b_delta),
                "R2_total": float(r2),
            }

    return best_params


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
            else:
                print(f"[WARN] {aoi_name}: no valid pixels after filtering.")
        except Exception as e:
            print(f"[WARN] {aoi_name} failed: {e}")

    if not all_dfs:
        raise RuntimeError("No AOI data extracted. Check aoi_list paths and data alignment.")

    df_all = pd.concat(all_dfs, ignore_index=True)

    X_all = df_all["NOx"].values
    y_all = df_all["Chlorophyll"].values
    y_all_pred_global = GLOBAL_SLOPE * X_all + GLOBAL_INTERCEPT
    R2_global_all = r2_score(y_all, y_all_pred_global)

    print("\n[GLOBAL LINEAR LINE] Fixed parameters")
    print(f"  n (all AOIs pooled) = {len(df_all)}")
    print(f"  slope               = {GLOBAL_SLOPE:.8g}")
    print(f"  intercept           = {GLOBAL_INTERCEPT:.8g}")
    print(f"  R2 (vs all data)    = {R2_global_all:.4f}\n")

    for idx, (aoi_name, _) in enumerate(aoi_list):
        df_aoi = df_all[df_all["AOI"] == aoi_name].copy()
        if df_aoi.empty:
            print(f"[WARN] {aoi_name}: no data, skipping.")
            continue

        X_aoi = df_aoi["NOx"].values
        y_aoi = df_aoi["Chlorophyll"].values

        x_min_aoi = X_aoi.min()
        x_max_aoi = X_aoi.max()
        x_line_aoi = np.linspace(x_min_aoi, x_max_aoi, 200)

        if idx == 0:
            # AOI 1: piecewise linear fit
            params_pw = fit_piecewise_linear_hinge(X_aoi, y_aoi, n_breaks=60, min_frac=0.05)

            x0 = params_pw["x0"]
            a = params_pw["a"]
            slope1 = params_pw["slope1"]
            slope2 = params_pw["slope2"]
            R2_total = params_pw["R2_total"]

            intercept1 = a
            intercept2 = a - (slope2 - slope1) * x0

            mask_seg1 = X_aoi < x0
            mask_seg2 = ~mask_seg1

            y_pred_seg1 = intercept1 + slope1 * X_aoi[mask_seg1]
            y_pred_seg2 = intercept2 + slope2 * X_aoi[mask_seg2]

            R2_seg1 = r2_score(y_aoi[mask_seg1], y_pred_seg1) if mask_seg1.sum() > 1 else np.nan
            R2_seg2 = r2_score(y_aoi[mask_seg2], y_pred_seg2) if mask_seg2.sum() > 1 else np.nan

            print(f"[FIT] {aoi_name} piecewise linear:")
            print(f"  n_total           = {len(df_aoi)}")
            print(f"  breakpoint x0     = {x0:.8g}")
            print(f"  Segment 1 (x < x0): n={int(mask_seg1.sum())}, slope={slope1:.8g}, intercept={intercept1:.8g}, R2={R2_seg1:.4f}")
            print(f"  Segment 2 (x >= x0): n={int(mask_seg2.sum())}, slope={slope2:.8g}, intercept={intercept2:.8g}, R2={R2_seg2:.4f}")
            print(f"  Overall R2        = {R2_total:.4f}\n")

            y_line_aoi = np.empty_like(x_line_aoi)
            mask_line_seg1 = x_line_aoi < x0
            mask_line_seg2 = ~mask_line_seg1
            y_line_aoi[mask_line_seg1] = intercept1 + slope1 * x_line_aoi[mask_line_seg1]
            y_line_aoi[mask_line_seg2] = intercept2 + slope2 * x_line_aoi[mask_line_seg2]

            line_label = "AOI-specific piecewise fit"

        else:
            # AOI 2-5: linear fit
            lr_aoi = LinearRegression().fit(X_aoi.reshape(-1, 1), y_aoi)
            y_pred_aoi = lr_aoi.predict(X_aoi.reshape(-1, 1))
            slope_aoi = lr_aoi.coef_[0]
            intercept_aoi = lr_aoi.intercept_
            R2_aoi = r2_score(y_aoi, y_pred_aoi)

            print(f"[FIT] {aoi_name} linear:")
            print(f"  n_total       = {len(df_aoi)}")
            print(f"  slope         = {slope_aoi:.8g}")
            print(f"  intercept     = {intercept_aoi:.8g}")
            print(f"  R2            = {R2_aoi:.4f}")

            if "AOI_5" in aoi_name:
                s_ols, i_ols, ci_low, ci_high, ci_hw = ols_slope_ci(X_aoi, y_aoi, alpha=0.05)
                print(f"  [OLS 95% CI] slope={s_ols:.8g}, CI=[{ci_low:.8g}, {ci_high:.8g}], +/-{ci_hw:.8g}")

            print()

            y_line_aoi = lr_aoi.predict(x_line_aoi.reshape(-1, 1))
            line_label = "AOI-specific linear fit"

        # Plot
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.size": 12,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "axes.linewidth": 1.2,
        })

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=df_aoi, x="NOx", y="Chlorophyll",
            hue="Year", palette="viridis", alpha=0.6, s=12
        )

        plt.plot(x_line_aoi, y_line_aoi, color="red", linewidth=2.0, linestyle="-", label=line_label)

        y_line_global_aoi = GLOBAL_SLOPE * x_line_aoi + GLOBAL_INTERCEPT
        plt.plot(x_line_aoi, y_line_global_aoi, color="black", linewidth=1.8, linestyle="--", label="Global linear line")

        plt.title(f"Chlor-NOx Relationship in {aoi_name} (2013-2017)")
        plt.xlabel("NOx Emissions (g m-2 yr-1)")
        plt.ylabel("Chlorophyll a Concentrations (mg m-3)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        safe_name = aoi_name.replace(" ", "_").replace("(", "").replace(")", "")
        out_fig = os.path.join(OUT_DIR, f"nox_chlor_{safe_name}_fit.png")
        plt.savefig(out_fig, dpi=300)
        plt.show()
        print(f"[Saved] {out_fig}")
