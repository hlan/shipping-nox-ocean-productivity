import os
import warnings
import numpy as np
import pandas as pd
import rasterio
import netCDF4 as nc
import statsmodels.api as sm
import geopandas as gpd
from rasterio.features import rasterize
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

YEARS_TABLE1 = [2013, 2014, 2015, 2016, 2017]
COV = lambda var, y: os.path.join("data", "covariates", f"{var}_{y}.nc")
MIN_MONTHS = 6
OUT_DIR = os.path.join("results", "Fig_3_Table_1_Table_S22")
TABLE_1 = os.path.join(OUT_DIR, "Table_1.csv")
TABLE_1_FITS = os.path.join(OUT_DIR, "Table_1_chl_aoi_fits.csv")
OUT_TABLE1 = os.path.join(OUT_DIR, "Table_1_covariate_slopes.csv")
FIG3_DIR = os.path.join(OUT_DIR, "Fig_3")
DRAW_CONTROL_LINE = True
FONT_FAMILY = "Times New Roman"
FIG3_FONT = 11
FIG3_TICKS = 5
FIG3_SIZE = (6, 4.5)
FIG3_AXES = [0.125, 0.13, 0.86, 0.85]
XLABEL = "NO$_{\\mathrm{x}}$ Emissions (g m$^{-2}$ yr$^{-1}$)"
YLABEL = "Chlorophyll $a$ Concentrations (mg m$^{-3}$)"
LABEL_AOI_FIT = "AOI-specific linear fit"
LABEL_CONTROL_FIT = "AOI-specific linear fit after control"
LABEL_GLOBAL = "Global linear line"
AOI_LIST = [("NW Pacific off Japan (AOI_1)", "all_AOIs_geojson/AOI1_NW_Pacific_off_Japan.geojson"),
            ("Central South Atlantic (AOI_2)", "all_AOIs_geojson/AOI2_Central_South_Atlantic.geojson"),
            ("North Atlantic (AOI_3)", "all_AOIs_geojson/AOI3_North_Atlantic.geojson"),
            ("NE Pacific (AOI_4)", "all_AOIs_geojson/AOI4_NE_Pacific.geojson"),
            ("Eastern Pacific (AOI_5)", "all_AOIs_geojson/AOI5_Eastern_Pacific.geojson")]
AOI_LABELS = ["AOI-1", "AOI-2", "AOI-3", "AOI-4", "AOI-5"]


def require(paths):
    """Stop with the list of missing input files."""
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit("[ERROR] missing input file(s):\n    " + "\n    ".join(missing))


def read_covariate(path, checks):
    """Annual mean of a monthly 0.5 degree field, north-up on the chlorophyll a grid, at least MIN_MONTHS valid months."""
    d = nc.Dataset(path)
    lat_name = next(v for v in ("lat", "latitude", "y") if v in d.variables)
    lon_name = next(v for v in ("lon", "longitude", "x") if v in d.variables)
    var = next(v for v in d.variables if v not in ("time", "month", lat_name, lon_name) and d.variables[v].ndim == 3)
    lat = np.array(d.variables[lat_name][:], dtype=float); lon = np.array(d.variables[lon_name][:], dtype=float)
    A = np.array(d.variables[var][:], dtype="float64"); attrs = {k: str(d.variables[var].getncattr(k)) for k in d.variables[var].ncattrs()}
    d.close(); A[~np.isfinite(A)] = np.nan
    assert A.shape == (12, 360, 720), f"{path}: expected (12, 360, 720), got {A.shape}"
    assert np.allclose(np.abs(lat[:2]) % 0.5, 0.25) and np.allclose(np.abs(lon[:2]) % 0.5, 0.25), f"{path}: cell centers are not at x.25 / x.75"
    assert lon[0] < 0, f"{path}: longitude does not start at -180"
    flipped = lat[0] < lat[-1]
    if flipped: A = A[:, ::-1, :]; lat = lat[::-1]
    assert abs(lat[0] - 89.75) < 1e-3, f"{path}: after orientation the first latitude is {lat[0]}"
    n_ok = np.isfinite(A).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ann = np.where(n_ok >= MIN_MONTHS, np.nanmean(A, axis=0), np.nan)
    checks.append(f"{os.path.basename(path)}: variable {var}, long_name {attrs.get('long_name', '')!r}, units {attrs.get('units', '')!r}, "
                  f"stored {'south-up, flipped to north-up' if flipped else 'north-up'}, monthly range {np.nanmin(A):.2f} to {np.nanmax(A):.2f}, "
                  f"cells with at least {MIN_MONTHS} months {int((n_ok >= MIN_MONTHS).sum()):,}")
    return ann


def region_mask(raster_path, geojson_path):
    with rasterio.open(raster_path) as src:
        gdf = gpd.read_file(geojson_path)
        shapes = [g for g in gdf.geometry if g.is_valid and g.geom_type in ("Polygon", "MultiPolygon")]
        return rasterize([(g, 1) for g in shapes], out_shape=(src.height, src.width), transform=src.transform, fill=0, dtype="uint8", all_touched=True).astype(bool)


def extract_all(mask, years):
    """Grid-cell years of an AOI with chlorophyll a and emission above zero (the Table 1 population)."""
    rows = []
    for y in years:
        with rasterio.open(f"data/chlor/chlor_{y}.tif") as c, rasterio.open(f"data/emission/NO_{y}_yearly.tif") as e:
            chl = c.read(1).astype("float64")[mask]; nox = e.read(1).astype("float64")[mask]
        ok = (chl > 0) & (nox > 0)
        rows.append(pd.DataFrame({"Year": y, "Chlorophyll": chl[ok], "NOx": nox[ok]}))
    return pd.concat(rows, ignore_index=True)


def extract(mask, years):
    """Grid-cell years of an AOI with chlorophyll a, emission and the three covariates."""
    rows = []
    for y in years:
        with rasterio.open(f"data/chlor/chlor_{y}.tif") as c, rasterio.open(f"data/emission/NO_{y}_yearly.tif") as e:
            chl = c.read(1).astype("float64")[mask]; nox = e.read(1).astype("float64")[mask]
        sst, par, mld = (read_covariate(COV(v, y), [])[mask] for v in ("sst", "par", "mld"))
        ok = (chl > 0) & (nox > 0) & np.isfinite(sst) & np.isfinite(par) & np.isfinite(mld) & (mld > 0)
        rows.append(pd.DataFrame({"Year": y, "Chlorophyll": chl[ok], "NOx": nox[ok], "SST": sst[ok], "PAR": par[ok], "logMLD": np.log1p(mld[ok])}))
    return pd.concat(rows, ignore_index=True)


def ols_fit(x, y):
    """OLS of y on x: slope, intercept, 95% CI half-width of the slope, R2 and p of the slope."""
    m = sm.OLS(np.asarray(y, float), sm.add_constant(np.asarray(x, float))).fit(); ci = m.conf_int(0.05)
    return float(m.params[1]), float(m.params[0]), float((ci[1, 1] - ci[1, 0]) / 2), float(m.rsquared), float(m.pvalues[1])


def slope_ci(X, y, k=1):
    """OLS of y on the columns of X (constant added): coefficient k, its 95% CI half-width and the R2."""
    m = sm.OLS(y, sm.add_constant(X)).fit(); ci = m.conf_int(0.05)
    return float(m.params[k]), float((ci[k, 1] - ci[k, 0]) / 2), float(m.rsquared)


def fit_hinge(x, y, n_breaks=60, min_frac=0.05):
    """Grid search of the breakpoint of one continuous two-segment line; returns x0, intercept, the two slopes and the R2."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    margin = (x.max() - x.min()) * min_frac
    best = None
    for x0 in np.linspace(x.min() + margin, x.max() - margin, n_breaks):
        X = np.column_stack([np.ones_like(x), x, np.maximum(0.0, x - x0)])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r2 = 1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        if best is None or r2 > best["R2_total"]:
            best = {"x0": float(x0), "a": float(beta[0]), "slope1": float(beta[1]), "slope2": float(beta[1] + beta[2]), "R2_total": float(r2)}
    return best


def hinge_slopes(df, x0, with_cov):
    """OLS on the hinge design with the breakpoint fixed: (slope1, CI), (slope2, CI), R2, and the p of both slopes."""
    x = df["NOx"].to_numpy(); h = np.maximum(0.0, x - x0)
    X = np.column_stack([x, h] + ([df["SST"], df["PAR"], df["logMLD"]] if with_cov else []))
    m = sm.OLS(df["Chlorophyll"].to_numpy(), sm.add_constant(X)).fit(); ci1 = m.conf_int(0.05)[1]
    t2 = m.t_test(np.array([[0.0, 1.0, 1.0] + [0.0] * (X.shape[1] - 2)])); ci2 = np.asarray(t2.conf_int(0.05)).ravel()
    return ((float(m.params[1]), float((ci1[1] - ci1[0]) / 2)), (float(m.params[1] + m.params[2]), float((ci2[1] - ci2[0]) / 2)),
            float(m.rsquared), float(m.pvalues[1]), float(np.asarray(t2.pvalue).ravel()[0]))


def f2(v): return "" if v is None or not np.isfinite(v) else f"{v:.2f}"
def fp(v): return "" if v is None or not np.isfinite(v) else ("<0.001" if v < 0.001 else f"{v:.3f}")
def fn(v): return "" if v is None else f"{int(v):,}"
def pm(s, c): return "" if s == "" else f"{s} ± {c}"


def table1_fits(dfs):
    """Table 1 fits on all grid-cell years (emission alone): rows of Table_1_chl_aoi_fits.csv, the AOI 1 breakpoint and the
    numbers needed for the figure and the merged table."""
    rows, info = [], {}
    df_fit = pd.concat([dfs[AOI_LABELS[i]] for i in range(4)], ignore_index=True)
    s_g, i_g, ci_g, r2_g, p_g = ols_fit(df_fit["NOx"], df_fit["Chlorophyll"])
    rows.append(["Global (AOI 1-4)", "linear", fn(len(df_fit)), f2(s_g), f2(ci_g), f2(i_g), f2(r2_g), fp(p_g)])
    info["Global (AOI 1-4)"] = dict(n=len(df_fit), slope=s_g, ci=ci_g, intercept=i_g, r2=r2_g)
    print(f"[Global (AOI 1-4)] n={len(df_fit):,}, slope={s_g:.8g} (+/-{ci_g:.8g}), intercept={i_g:.8g}, R2={r2_g:.4f}, p={p_g:.3g}")
    for idx, lab in enumerate(AOI_LABELS):
        df = dfs[lab]; x = df["NOx"].to_numpy(); y = df["Chlorophyll"].to_numpy()
        if idx == 0:
            pw = fit_hinge(x, y); x0 = pw["x0"]
            (s1, c1), (s2, c2), r2_total, p1, p2 = hinge_slopes(df, x0, False)
            i1 = pw["a"]; i2 = pw["a"] - (pw["slope2"] - pw["slope1"]) * x0
            m1 = x < x0; m2 = ~m1
            r2_seg = lambda m, i_, s_: 1 - ((y[m] - (i_ + s_ * x[m])) ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum() if m.sum() > 1 else np.nan
            r21, r22 = r2_seg(m1, i1, s1), r2_seg(m2, i2, s2)
            print(f"[{lab}] piecewise linear: n={len(df):,}, breakpoint x0={x0:.8g}; segment 1 n={int(m1.sum()):,}, slope={s1:.8g} (+/-{c1:.8g}, p={p1:.3g}), R2={r21:.4f}; "
                  f"segment 2 n={int(m2.sum()):,}, slope={s2:.8g} (+/-{c2:.8g}, p={p2:.3g}), R2={r22:.4f}; overall R2={r2_total:.4f}")
            rows.append([f"{lab}_seg1", "Piecewise linear", fn(m1.sum()), f2(s1), f2(c1), f2(i1), f2(r21), fp(p1)])
            rows.append([f"{lab}_seg2", "", fn(m2.sum()), f2(s2), f2(c2), f2(i2), f2(r22), fp(p2)])
            rows.append([lab, f"Breakpoint*: x={x0:.3f}", fn(len(df)), "", "", "", f2(r2_total), ""])
            info[f"{lab}_seg1"] = dict(n=int(m1.sum()), slope=s1, ci=c1, intercept=i1, r2=r21)
            info[f"{lab}_seg2"] = dict(n=int(m2.sum()), slope=s2, ci=c2, intercept=i2, r2=r22)
            info[lab] = dict(n=len(df), slope="", ci="", intercept="", r2=r2_total, x0=x0)
        else:
            s, i_, c, r2, p = ols_fit(x, y)
            print(f"[{lab}] linear: n={len(df):,}, slope={s:.8g} (+/-{c:.8g}), intercept={i_:.8g}, R2={r2:.4f}, p={p:.3g}")
            rows.append([lab, "linear", fn(len(df)), f2(s), f2(c), f2(i_), f2(r2), fp(p)])
            info[lab] = dict(n=len(df), slope=s, ci=c, intercept=i_, r2=r2)
    T = pd.DataFrame(rows, columns=["AOI", "Model", "n (grid-cell years)", "Slope (mg m⁻³ per g m⁻² yr⁻¹)", "95% CI half-width", "Intercept (mg m⁻³)", "R²", "p"])
    T.to_csv(TABLE_1_FITS, index=False, encoding="utf-8-sig"); print(f"[Saved] {TABLE_1_FITS}")
    return info


def covariate_fits(x0, info):
    """Table 1 fits with SST, PAR and log1p(MLD) as covariates on the cells with covariate data (Table_1_covariate_slopes.csv),
    Table S22 (A, B); returns the after-control numbers by Table 1 row."""
    years = [y for y in YEARS_TABLE1 if all(os.path.exists(COV(v, y)) for v in ("sst", "par", "mld"))]
    assert years, "no year has all three covariate files"
    checks = [f"years with covariates: {years} (Table 1 pools {YEARS_TABLE1})", f"AOI 1 breakpoint from the hinge search at full precision: x0 = {x0}"]
    dfs = {AOI_LABELS[i]: extract(region_mask(f"data/chlor/chlor_{years[0]}.tif", gj), years) for i, (_, gj) in enumerate(AOI_LIST)}
    rows, after = [], {}
    def add(label, model, df, seg=None):
        n = len(df); y = df["Chlorophyll"].to_numpy()
        if seg is None:
            s0, c0, r0 = slope_ci(df[["NOx"]].to_numpy(), y); s1, c1, r1 = slope_ci(df[["NOx", "SST", "PAR", "logMLD"]].to_numpy(), y)
            rows.append([label, model, f"{n:,}", f2(s0), f2(c0), f2(r0), f2(s1), f2(c1), f2(r1)])
            after[label] = dict(n=n, slope=s1, ci=c1, r2=r1)
        else:
            (a0, ca0), (b0, cb0), r0, _, _ = hinge_slopes(df, seg, False); (a1, ca1), (b1, cb1), r1, _, _ = hinge_slopes(df, seg, True)
            n1 = int((df["NOx"] < seg).sum())
            checks.append(f"{label}: cells below the breakpoint {n1:,}, at or above {n - n1:,} (all cells: {info[f'{label}_seg1']['n']:,} and {info[f'{label}_seg2']['n']:,}); "
                          f"emission-alone slopes on the covariate cells {a0:.4f} and {b0:.4f} (all cells: {info[f'{label}_seg1']['slope']:.4f} and {info[f'{label}_seg2']['slope']:.4f})")
            rows.append([f"{label}_seg1", "Piecewise linear", f"{n1:,}", f2(a0), f2(ca0), "", f2(a1), f2(ca1), ""])
            rows.append([f"{label}_seg2", "", f"{n - n1:,}", f2(b0), f2(cb0), "", f2(b1), f2(cb1), ""])
            rows.append([label, f"Breakpoint*: x={seg:.3f}", f"{n:,}", "", "", f2(r0), "", "", f2(r1)])
            after[f"{label}_seg1"] = dict(n=n1, slope=a1, ci=ca1, r2=""); after[f"{label}_seg2"] = dict(n=n - n1, slope=b1, ci=cb1, r2="")
            after[label] = dict(n=n, slope="", ci="", r2=r1, x0=seg)
    df_fit = pd.concat([dfs[AOI_LABELS[i]] for i in range(4)], ignore_index=True)
    add("Global (AOI 1-4)", "linear", df_fit)
    for i, lab in enumerate(AOI_LABELS):
        add(lab, "linear" if i else "", dfs[lab], seg=x0 if i == 0 else None)
    yrs = f"{years[0]}" if len(years) == 1 else f"{years[0]} to {years[-1]}"
    header = ["AOI", "Model", f"n (grid-cell years, {yrs})", f"Slope, emission alone ({yrs})", "95% CI half-width", "R²",
              f"Slope after controlling for SST, PAR and MLD ({yrs})", "95% CI half-width ", "R² "]
    T = pd.DataFrame(rows, columns=header); T.to_csv(OUT_TABLE1, index=False, encoding="utf-8-sig")

    yearly = {}
    for lab in AOI_LABELS:
        df = dfs[lab]; recs = []
        for y, g in df.groupby("Year"):
            yv = g["Chlorophyll"].to_numpy()
            s0, c0, _ = slope_ci(g[["NOx"]].to_numpy(), yv); s1, c1, _ = slope_ci(g[["NOx", "SST", "PAR", "logMLD"]].to_numpy(), yv)
            recs.append({"Year": int(y), "Cells": len(g), "SST (°C)": g["SST"].mean(), "MLD (m)": np.expm1(g["logMLD"]).mean(),
                         "Chlorophyll a (mg m-3)": yv.mean(), "Emission (g NO m-2 yr-1)": g["NOx"].mean(),
                         "Slope, emission alone": s0, "95% CI half-width": c0, "Slope after controlling for SST, PAR and MLD": s1, "95% CI half-width ": c1})
        yearly[lab] = pd.DataFrame(recs)
    yearly["AOI-5"].to_csv(os.path.join(OUT_DIR, "Table_S22_A_AOI5_by_year.csv"), index=False, float_format="%.2f", encoding="utf-8-sig")
    rowsB = []
    for lab in ["AOI-2", "AOI-3", "AOI-4", "AOI-5"]:
        df = dfs[lab]; yv = df["Chlorophyll"].to_numpy(); Y = yearly[lab]
        p0, _, _ = slope_ci(df[["NOx"]].to_numpy(), yv); p1, _, _ = slope_ci(df[["NOx", "SST", "PAR", "logMLD"]].to_numpy(), yv)
        rowsB.append({"AOI": lab.replace("-", " "), "Pooled slope, emission alone": p0,
                      "Yearly slopes, emission alone": f"{Y['Slope, emission alone'].min():.2f} to {Y['Slope, emission alone'].max():.2f}",
                      "Pooled slope after controlling for SST, PAR and MLD": p1,
                      "Yearly slopes after controlling for SST, PAR and MLD": f"{Y['Slope after controlling for SST, PAR and MLD'].min():.2f} to {Y['Slope after controlling for SST, PAR and MLD'].max():.2f}"})
    pd.DataFrame(rowsB).to_csv(os.path.join(OUT_DIR, "Table_S22_B_pooled_and_yearly_slopes_AOI2_to_AOI5.csv"), index=False, float_format="%.2f", encoding="utf-8-sig")
    checks.append("Table S22 (A): AOI 5 by year, single linear fits per year. Table S22 (B): pooled slopes (single linear fits on all years) against the range of yearly slopes, AOI 2 to AOI 5")
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print("\nTable S22 (A) AOI 5 by year, mean conditions and the slope of chlorophyll a on emission, with and without covariates"); print(yearly["AOI-5"].round(2).to_string(index=False))
        print("\nTable S22 (B) AOI 2 to AOI 5, pooled slope against the range of yearly slopes, with and without covariates"); print(pd.DataFrame(rowsB).round(2).to_string(index=False))
    with open(os.path.join(OUT_DIR, "checks_table_1_covariate_slopes.txt"), "w", encoding="utf-8") as fh: fh.write("\n".join(checks) + "\n")
    with pd.option_context("display.width", 220, "display.max_columns", 20): print("\n".join(checks)); print(T.to_string(index=False))
    print(f"[Saved] {OUT_TABLE1}")
    return after


def write_table_1(info, after):
    """Table 1 as it appears in the manuscript: both slope columns as slope ± 95% CI half-width, no p column."""
    rows = []
    for lab in ["Global (AOI 1-4)", "AOI-1_seg1", "AOI-1_seg2", "AOI-1", "AOI-2", "AOI-3", "AOI-4", "AOI-5"]:
        b, a = info[lab], after[lab]
        model = "linear" if lab in ("Global (AOI 1-4)", "AOI-2", "AOI-3", "AOI-4", "AOI-5") else ("Piecewise linear" if lab.endswith("seg1") else ("" if lab.endswith("seg2") else f"Breakpoint*: x={b['x0']:.3f}"))
        rows.append([lab, model, fn(b["n"]), pm(f2(b["slope"]) if b["slope"] != "" else "", f2(b["ci"]) if b["ci"] != "" else ""), f2(b["intercept"]) if b["intercept"] != "" else "", f2(b["r2"]),
                     fn(a["n"]), pm(f2(a["slope"]) if a["slope"] != "" else "", f2(a["ci"]) if a["ci"] != "" else ""), f2(a["r2"]) if a["r2"] != "" else ""])
    T = pd.DataFrame(rows, columns=["AOI", "Model", "n (grid-cell years)", "Slope ± 95% CI half-width (mg m⁻³ per g m⁻² yr⁻¹)", "Intercept (mg m⁻³)", "R²",
                                    "n with covariate data", "Slope after controlling for SST, PAR and MLD ± 95% CI half-width", "R² after control"])
    T.to_csv(TABLE_1, index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 250, "display.max_columns", 20): print("\nTable 1"); print(T.to_string(index=False))
    print(f"[Saved] {TABLE_1}")


def fig_3(dfs, info, after):
    """Fig. 3: one scatter panel per AOI (no title, no legend, the axes box at the same place in every panel so the files have one
    size) with the AOI fit, the global line and, when DRAW_CONTROL_LINE, the slope after the control through the centroid; plus one
    legend strip for the assembled figure."""
    plt.rcParams.update({"font.family": FONT_FAMILY, "font.size": FIG3_FONT, "axes.labelsize": FIG3_FONT, "xtick.labelsize": FIG3_FONT,
                         "ytick.labelsize": FIG3_FONT, "legend.fontsize": FIG3_FONT, "axes.linewidth": 1.2, "mathtext.fontset": "stix", "pdf.fonttype": 42})
    g = info["Global (AOI 1-4)"]; cmap = plt.get_cmap("viridis"); years = YEARS_TABLE1
    year_color = {yr: cmap(k / (len(years) - 1)) for k, yr in enumerate(years)}
    for idx, lab in enumerate(AOI_LABELS):
        df = dfs[lab]; x = df["NOx"].to_numpy(); y = df["Chlorophyll"].to_numpy()
        xl = np.linspace(x.min(), x.max(), 200)
        if idx == 0:
            x0 = info[lab]["x0"]; s1, s2 = info["AOI-1_seg1"]["slope"], info["AOI-1_seg2"]["slope"]; i1 = info["AOI-1_seg1"]["intercept"]
            hinge = lambda s_a, s_b, xx: s_a * xx + (s_b - s_a) * np.maximum(0.0, xx - x0)
            yl = i1 + hinge(s1, s2, xl)
            sa, sb = after["AOI-1_seg1"]["slope"], after["AOI-1_seg2"]["slope"]
            yc = y.mean() + hinge(sa, sb, xl) - hinge(sa, sb, x.mean())
        else:
            yl = info[lab]["intercept"] + info[lab]["slope"] * xl
            yc = y.mean() + after[lab]["slope"] * (xl - x.mean())
        fig = plt.figure(figsize=FIG3_SIZE); ax = fig.add_axes(FIG3_AXES)
        for yr in years:
            m = df["Year"].to_numpy() == yr
            ax.scatter(x[m], y[m], s=12, alpha=0.6, color=year_color[yr], linewidths=0)
        ax.plot(xl, yl, color="red", lw=2.0)
        if DRAW_CONTROL_LINE:
            ax.plot(xl, yc, color="red", lw=1.8, ls=":")
        ax.plot(xl, g["slope"] * xl + g["intercept"], color="black", lw=1.8, ls="--")
        ax.xaxis.set_major_locator(MaxNLocator(FIG3_TICKS)); ax.yaxis.set_major_locator(MaxNLocator(FIG3_TICKS))
        ax.set_xlabel(XLABEL); ax.set_ylabel(YLABEL)
        ax.grid(True, alpha=0.3)
        out = os.path.join(FIG3_DIR, f"fig3_aoi_{idx + 1}")
        fig.savefig(out + ".png", dpi=300); fig.savefig(out + ".pdf"); plt.close(fig)
        print(f"[Saved] {out}.png")
    handles = [Line2D([0], [0], marker="o", color=year_color[yr], linestyle="None", markersize=6, label=str(yr)) for yr in years]
    handles.append(Line2D([0], [0], color="red", lw=2.0, label=LABEL_AOI_FIT))
    if DRAW_CONTROL_LINE:
        handles.append(Line2D([0], [0], color="red", lw=1.8, ls=":", label=LABEL_CONTROL_FIT))
    handles.append(Line2D([0], [0], color="black", lw=1.8, ls="--", label=LABEL_GLOBAL))
    fig = plt.figure(figsize=(12, 0.6))
    fig.legend(handles=handles, loc="center", ncol=len(handles), frameon=False, handletextpad=0.5, columnspacing=1.5)
    out = os.path.join(FIG3_DIR, "fig3_legend")
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight", pad_inches=0.02); fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.02); plt.close(fig)
    print(f"[Saved] {out}.png")


def main():
    os.makedirs(FIG3_DIR, exist_ok=True)
    require([f"data/chlor/chlor_{y}.tif" for y in YEARS_TABLE1] + [f"data/emission/NO_{y}_yearly.tif" for y in YEARS_TABLE1] + [gj for _, gj in AOI_LIST])
    dfs = {AOI_LABELS[i]: extract_all(region_mask(f"data/chlor/chlor_{YEARS_TABLE1[0]}.tif", gj), YEARS_TABLE1) for i, (_, gj) in enumerate(AOI_LIST)}
    info = table1_fits(dfs)
    after = covariate_fits(info["AOI-1"]["x0"], info)
    write_table_1(info, after)
    fig_3(dfs, info, after)


if __name__ == "__main__":
    main()
