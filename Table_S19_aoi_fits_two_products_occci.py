import os
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from rasterio.features import rasterize
import statsmodels.api as sm

PRODUCTS = [("Aqua/MODIS", "data/chlor/chlor_{y}.tif"), ("OC-CCI", "data/chlor_oc_cci/chlor_{y}.tif")]
YEARS = [2013, 2014, 2015, 2016, 2017]
EMISSION = "data/emission/NO_{y}_yearly.tif"
AOI_LIST = [("AOI-1", "all_AOIs_geojson/AOI1_NW_Pacific_off_Japan.geojson"),
            ("AOI-2", "all_AOIs_geojson/AOI2_Central_South_Atlantic.geojson"),
            ("AOI-3", "all_AOIs_geojson/AOI3_North_Atlantic.geojson"),
            ("AOI-4", "all_AOIs_geojson/AOI4_NE_Pacific.geojson"),
            ("AOI-5", "all_AOIs_geojson/AOI5_Eastern_Pacific.geojson")]
TABLE1_FITS = os.path.join("results", "Fig_3_Table_1_Table_S22", "Table_1_chl_aoi_fits.csv")
OUT_DIR = os.path.join("results_cci", "Table_S19")
ROW_ORDER = [("Global (AOI 1-4)", "linear"), ("AOI-1_seg1", "Piecewise linear"), ("AOI-1_seg2", ""), ("AOI-1", "breakpoint"),
             ("AOI-2", "linear"), ("AOI-3", "linear"), ("AOI-4", "linear"), ("AOI-5", "linear"), ("AOI-5, 2013 only", "linear")]
HEADER = ["AOI", "Model"] + [f"{lab}: {c}" for lab, _ in PRODUCTS
                             for c in ("n (grid-cell years)", "slope ± 95% CI half-width (mg m⁻³ per g m⁻² yr⁻¹)", "intercept (mg m⁻³)", "R²")]


def require(paths):
    """Stop with the list of missing input files."""
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit("[ERROR] missing input file(s):\n    " + "\n    ".join(missing))


def region_mask(raster_path, geojson_path):
    with rasterio.open(raster_path) as src:
        gdf = gpd.read_file(geojson_path)
        shapes = [g for g in gdf.geometry if g.is_valid and g.geom_type in ("Polygon", "MultiPolygon")]
        return rasterize([(g, 1) for g in shapes], out_shape=(src.height, src.width), transform=src.transform, fill=0, dtype="uint8", all_touched=True).astype(bool)


def extract_all(chl_tpl, mask):
    """Grid-cell years of an AOI with chlorophyll a and emission above zero (the Table 1 population)."""
    rows = []
    for y in YEARS:
        with rasterio.open(chl_tpl.format(y=y)) as c, rasterio.open(EMISSION.format(y=y)) as e:
            chl = c.read(1).astype("float64")[mask]; nox = e.read(1).astype("float64")[mask]
        ok = (chl > 0) & (nox > 0)
        rows.append(pd.DataFrame({"Year": y, "Chlorophyll": chl[ok], "NOx": nox[ok]}))
    return pd.concat(rows, ignore_index=True)


def ols_fit(x, y):
    """OLS of y on x: slope, intercept, 95% CI half-width of the slope, R2 and p of the slope."""
    m = sm.OLS(np.asarray(y, float), sm.add_constant(np.asarray(x, float))).fit(); ci = m.conf_int(0.05)
    return float(m.params[1]), float(m.params[0]), float((ci[1, 1] - ci[1, 0]) / 2), float(m.rsquared), float(m.pvalues[1])


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


def hinge_slopes(x, y, x0):
    """OLS on the hinge design with the breakpoint fixed: (slope1, CI), (slope2, CI), R2, and the p of both slopes."""
    X = np.column_stack([x, np.maximum(0.0, x - x0)])
    m = sm.OLS(y, sm.add_constant(X)).fit(); ci1 = m.conf_int(0.05)[1]
    t2 = m.t_test(np.array([[0.0, 1.0, 1.0]])); ci2 = np.asarray(t2.conf_int(0.05)).ravel()
    return ((float(m.params[1]), float((ci1[1] - ci1[0]) / 2)), (float(m.params[1] + m.params[2]), float((ci2[1] - ci2[0]) / 2)),
            float(m.rsquared), float(m.pvalues[1]), float(np.asarray(t2.pvalue).ravel()[0]))


def f2(v): return f"{v:.2f}"


def product_rows(label, chl_tpl):
    """The Table 1 rows (plus AOI 5 in 2013 only) of one product: key -> (n, slope ± CI, intercept, R2) and the check lines."""
    ref = chl_tpl.format(y=YEARS[0])
    data = {lab: extract_all(chl_tpl, region_mask(ref, gj)) for lab, gj in AOI_LIST}
    out, notes = {}, []
    pooled = pd.concat([data[lab] for lab, _ in AOI_LIST[:4]], ignore_index=True)
    s, i_, c, r2, p = ols_fit(pooled["NOx"], pooled["Chlorophyll"])
    out["Global (AOI 1-4)"] = (f"{len(pooled):,}", f"{f2(s)} ± {f2(c)}", f2(i_), f2(r2)); notes.append(f"global p = {p:.3g}")
    for lab, _ in AOI_LIST:
        d = data[lab]; x = d["NOx"].to_numpy(); y = d["Chlorophyll"].to_numpy()
        if lab == "AOI-1":
            pw = fit_hinge(x, y); x0 = pw["x0"]
            (s1, c1), (s2, c2), r2_total, p1, p2 = hinge_slopes(x, y, x0)
            i1 = pw["a"]; i2 = pw["a"] - (pw["slope2"] - pw["slope1"]) * x0
            m1 = x < x0; m2 = ~m1
            r2_seg = lambda m, a, b: 1 - ((y[m] - (a + b * x[m])) ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum()
            out["AOI-1_seg1"] = (f"{int(m1.sum()):,}", f"{f2(s1)} ± {f2(c1)}", f2(i1), f2(r2_seg(m1, i1, s1)))
            out["AOI-1_seg2"] = (f"{int(m2.sum()):,}", f"{f2(s2)} ± {f2(c2)}", f2(i2), f2(r2_seg(m2, i2, s2)))
            out["AOI-1"] = (f"{len(d):,}", f"Breakpoint*: x={x0:.3f}", "", f2(r2_total))
            notes.append(f"AOI-1 breakpoint {x0:.8g}, p1 = {p1:.3g}, p2 = {p2:.3g}")
        else:
            s, i_, c, r2, p = ols_fit(x, y)
            out[lab] = (f"{len(d):,}", f"{f2(s)} ± {f2(c)}", f2(i_), f2(r2)); notes.append(f"{lab} p = {p:.3g}")
            if lab == "AOI-5":
                d13 = d[d["Year"] == 2013]; s, i_, c, r2, p = ols_fit(d13["NOx"], d13["Chlorophyll"])
                out["AOI-5, 2013 only"] = (f"{len(d13):,}", f"{f2(s)} ± {f2(c)}", f2(i_), f2(r2)); notes.append(f"AOI-5 2013 p = {p:.3g}")
    print(f"[{label}] " + "; ".join(notes))
    return out, [f"=== {label} ==="] + ["  " + n for n in notes]


def main():
    """Fit both products, write Table_S19_aoi_fits_two_products.csv and checks.txt."""
    require([tpl.format(y=y) for _, tpl in PRODUCTS for y in YEARS] + [EMISSION.format(y=y) for y in YEARS] + [gj for _, gj in AOI_LIST])
    os.makedirs(OUT_DIR, exist_ok=True)
    results, checks = {}, []
    for label, tpl in PRODUCTS:
        results[label], notes = product_rows(label, tpl); checks += notes
    rows = [[key, model] + [v for lab, _ in PRODUCTS for v in results[lab][key]] for key, model in ROW_ORDER]
    T = pd.DataFrame(rows, columns=HEADER)
    p = os.path.join(OUT_DIR, "Table_S19_aoi_fits_two_products.csv"); T.to_csv(p, index=False, encoding="utf-8-sig"); print(f"[Saved] {p}")
    if os.path.exists(TABLE1_FITS):
        t1 = pd.read_csv(TABLE1_FITS, dtype=str).set_index("AOI"); bad = []
        for key in ("Global (AOI 1-4)", "AOI-1_seg1", "AOI-1_seg2", "AOI-2", "AOI-3", "AOI-4", "AOI-5"):
            mine = results["Aqua/MODIS"][key][1].split(" ±")[0]
            if key in t1.index and t1.loc[key, t1.columns[2]] != mine:
                bad.append(f"{key}: Table 1 {t1.loc[key, t1.columns[2]]} vs here {mine}")
        checks.append("Aqua/MODIS slopes vs Table 1: " + ("all agree" if not bad else "; ".join(bad)))
    else:
        checks.append(f"Aqua/MODIS slopes not compared with Table 1 ({TABLE1_FITS} not found)")
    with open(os.path.join(OUT_DIR, "checks.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(checks) + "\n")
    print("\n".join(checks))
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(T.to_string(index=False))
    print(f"[Done] Table S19 in {OUT_DIR}/")


if __name__ == "__main__":
    main()
