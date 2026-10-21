import os, io, sys
import numpy as np
import pandas as pd
import rasterio
from scipy.stats import fisher_exact
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.transforms as mtransforms

YEARS = ["2013", "2014", "2015", "2016", "2017"]
REF_YEAR = "2017"
PANEL_LETTER_X = -0.09
PANEL_LETTER_RISE = 17
WINDOWS = lambda y: os.path.join(f"surrogate_102_{y}", "windows", "windows_baseline_all.csv")
CHLOR = lambda y: os.path.join("data", "chlor", f"chlor_{y}.tif")
EMISSION = lambda y: os.path.join("data", "emission", f"NO_{y}_yearly.tif")
NMEAN_CSV = os.path.join("data", "Nmean_60m_data_updated.csv")
BASIN_MASK = os.path.join("data", "basinmask_01.msk")
LANE_SHP = os.path.join("data", "shipping_lane", "mygeodata", "Shipping_Lanes_v1.shp")
OUT_DIR = os.path.join("results", "Fig_S31_Table_S13_S14")
INT_DIR = os.path.join(OUT_DIR, "intermediate")

N_LIMIT = 2.0
ROUTE_PCTL = 50
ROUTE_FRAC, ROUTE_FRAC_ALT = 0.25, 0.10
Q_GRAD_ALT = 0.50
Q_VAR, Q_VAR_ALT = 0.50, 1 / 3
GRAD_BINS = [0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, np.inf]
VAR_BINS = [0, 0.005, 0.01, 0.015, 0.025, 0.04, 0.06, 0.1, np.inf]
R2_SCREEN, SLOPE_FLOOR, WINDOW = 0.6, 0.01, 11
XRES = YRES = 0.5
R_EARTH = 6_371_000
BASIN_ORDER = ["North Pacific", "South Atlantic", "North Atlantic", "South Pacific", "Indian", "Southern Ocean", "Other", "Unmapped"]
FONT_FAMILY = "Times New Roman"

def require(paths):
    """Stop with the list of missing input files."""
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit("[ERROR] missing input file(s):\n    " + "\n    ".join(missing))

def pixel_area_km2(lat_deg):
    lat = np.deg2rad(lat_deg); dlat = np.deg2rad(YRES); dlon = np.deg2rad(XRES)
    return R_EARTH**2 * dlon * (np.sin(lat + dlat / 2) - np.sin(lat - dlat / 2)) / 1e6

def read_field(path):
    with rasterio.open(path) as src:
        a = src.read(1).astype("float64"); nod = src.nodata
        if nod is not None and np.isfinite(nod): a[a == nod] = np.nan
        a[~(np.isfinite(a) & (a > 0))] = np.nan
        return a, src.transform

def map_windows_to_nmean(df, nmean_csv_path):
    """Nearest 1 degree center lookup of the WOA nitrogen value, as in Table S2 and Fig. 2A."""
    nmean_df = pd.read_csv(nmean_csv_path)
    lon_centers = np.sort(nmean_df["Longitude"].unique()); lat_centers = np.sort(nmean_df["Latitude"].unique())
    def nearest_center(vals, centers):
        idx = np.searchsorted(centers, vals)
        idx0 = np.clip(idx - 1, 0, len(centers) - 1); idx1 = np.clip(idx, 0, len(centers) - 1)
        pick0 = np.abs(vals - centers[idx0]) <= np.abs(vals - centers[idx1])
        return np.where(pick0, centers[idx0], centers[idx1])
    out = df.copy()
    out["lon_c"] = nearest_center(out["Longitude"].values, lon_centers)
    out["lat_c"] = nearest_center(out["Latitude"].values, lat_centers)
    nmean_idx = nmean_df.set_index(["Longitude", "Latitude"])
    keys = list(zip(out["lon_c"].values, out["lat_c"].values))
    out["N60"] = nmean_idx.reindex(keys)["mean_N_Depth_60"].values
    return out.drop(columns=["lon_c", "lat_c"])

def load_mask():
    with open(BASIN_MASK, encoding="utf-8", errors="replace") as fh:
        lines = [ln for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
    has_header = any(ch.isalpha() for ch in lines[0])
    m = pd.read_csv(io.StringIO("".join(lines)), header=0 if has_header else None)
    if not has_header: m.columns = ["Latitude", "Longitude"] + [f"Basin_{i}" for i in range(m.shape[1] - 2)]
    low = {c.lower().strip(): c for c in m.columns}
    lat_c = next((c for k, c in low.items() if k.startswith("lat")), m.columns[0])
    lon_c = next((c for k, c in low.items() if k.startswith("lon")), m.columns[1])
    code_c = next((c for k, c in low.items() if k.startswith("basin")), m.columns[2])
    d = pd.DataFrame({"mlat": m[lat_c].astype(float).round(2), "mlon": m[lon_c].astype(float).round(2),
                      "code": pd.to_numeric(m[code_c], errors="coerce")}).dropna()
    d["code"] = d["code"].astype(int); return d.drop_duplicates(["mlat", "mlon"])

def basin_name(code, lat):
    if code == 2: return "North Pacific" if lat >= 0 else "South Pacific"
    if code == 1: return "North Atlantic" if lat >= 0 else "South Atlantic"
    if code == 3: return "Indian"
    if code == 10: return "Southern Ocean"
    return "Other"

def assign_basin(lat, lon, mask):
    key = pd.DataFrame({"mlat": np.floor(lat) + 0.5, "mlon": np.floor(lon) + 0.5})
    key["mlon"] = np.where(key["mlon"] > 180, key["mlon"] - 360, key["mlon"])
    j = key.merge(mask, on=["mlat", "mlon"], how="left")
    out = np.array(["Unmapped"] * len(j), dtype=object); ok = j["code"].notna().to_numpy()
    out[ok] = [basin_name(int(c), la) for c, la in zip(j.loc[ok, "code"], np.asarray(lat)[ok])]
    return out

def window_blocks(A, ii, jj, k=WINDOW):
    """Values of the k x k window around each (i, j) as an (n, k*k) array, NaN outside the grid (no wraparound, as in the regression)."""
    half = k // 2; rows, cols = A.shape; n = len(ii)
    out = np.full((n, k * k), np.nan)
    for a, di in enumerate(range(-half, half + 1)):
        for b, dj in enumerate(range(-half, half + 1)):
            r = ii + di; c = jj + dj; ok = (r >= 0) & (r < rows) & (c >= 0) & (c < cols)
            vals = np.full(n, np.nan); vals[ok] = A[r[ok], c[ok]]; out[:, a * k + b] = vals
    return out

def emission_percentile(year, pctl):
    """Percentile of the emission raster over the cells with emission above zero (the rule of the top-decile column of Table S10, at any percentile)."""
    with rasterio.open(EMISSION(year)) as src:
        arr = src.read(1).astype(float)
    good = np.isfinite(arr) & (arr > 0)
    return float(np.percentile(arr[good], pctl))

def window_metrics(year, route_thr):
    """Per-window metrics for the windows entering the screen in one year."""
    w = pd.read_csv(WINDOWS(year))
    w = w[w["Slope"].abs() >= SLOPE_FLOOR].reset_index(drop=True)
    C, tf = read_field(CHLOR(year)); E, tf2 = read_field(EMISSION(year))
    assert C.shape == E.shape and tf == tf2
    col, row = (~tf) * (w["Longitude"].to_numpy(), w["Latitude"].to_numpy())
    ii, jj = np.floor(row).astype(int), np.floor(col).astype(int)
    both = np.isfinite(C) & np.isfinite(E)
    logE = np.where(both, np.log1p(E), np.nan); logC = np.where(both, np.log1p(C), np.nan)
    bE, bC = window_blocks(logE, ii, jj), window_blocks(logC, ii, jj)
    bEraw = window_blocks(np.where(both, E, np.nan), ii, jj)
    n_valid = np.isfinite(bE).sum(axis=1)
    w["n_route"] = (bEraw >= route_thr).sum(axis=1)
    w["route_frac"] = w["n_route"] / n_valid
    p50, p90 = np.nanpercentile(bE, [50, 90], axis=1)
    w["n_valid"] = n_valid
    w["grad"] = p90 - p50
    w["bvar"] = np.nanstd(bC, axis=1)
    w["positive"] = (w["R2"] >= R2_SCREEN) & (w["Slope"] > 0)
    w["retained"] = w["R2"] >= R2_SCREEN
    w["area"] = pixel_area_km2(w["Latitude"].to_numpy())
    w = map_windows_to_nmean(w, NMEAN_CSV)
    w["nlim"] = np.where(np.isfinite(w["N60"]), w["N60"] < N_LIMIT, np.nan)
    return w

def enrichment(w, cond, name, year):
    """Base rate, conditional rate and retained-to-all ratio of one condition (NaN where the condition is undefined)."""
    d = w[np.isfinite(cond.astype(float))].copy(); c = cond[np.isfinite(cond.astype(float))].astype(bool)
    pos = d["positive"].to_numpy(); area = d["area"].to_numpy()
    base_n, cond_n = c.mean(), c[pos].mean()
    base_a, cond_a = area[c].sum() / area.sum(), area[pos & c].sum() / area[pos].sum()
    a, b = int((c & pos).sum()), int((~c & pos).sum()); cc, dd = int((c & ~pos).sum()), int((~c & ~pos).sum())
    odds, p = fisher_exact([[a, b], [cc, dd]], alternative="greater")
    return dict(year=year, condition=name, windows=len(d), meeting=int(c.sum()), base_count=base_n, base_area=base_a,
                positive=int(pos.sum()), positive_meeting=a, cond_count=cond_n, cond_area=cond_a,
                enrich_count=cond_n / base_n, enrich_area=cond_a / base_a, odds_ratio=odds, p_nominal=p)

def f0(v): return f"{int(v):,}"
def f3(v): return f"{v:.3f}"
def f2(v): return f"{v:.2f}"
def f1(v): return f"{v:.1f}"

def main():
    require([WINDOWS(y) for y in YEARS] + [CHLOR(y) for y in YEARS] + [EMISSION(y) for y in YEARS] + [NMEAN_CSV, BASIN_MASK])
    os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(INT_DIR, exist_ok=True); checks = []
    route_thr = emission_percentile(REF_YEAR, ROUTE_PCTL)
    W = {}
    for y in YEARS:
        W[y] = window_metrics(y, route_thr); print(f"[{y}] {len(W[y]):,} windows entering the screen, {int(W[y]['positive'].sum()):,} retained with a positive slope")
    ref = W[REF_YEAR]
    thr = dict(route=route_thr, grad_alt=float(np.quantile(ref["grad"], Q_GRAD_ALT)),
               bvar=float(np.quantile(ref["bvar"], Q_VAR)), bvar_alt=float(np.quantile(ref["bvar"], Q_VAR_ALT)))
    checks.append(f"thresholds from {REF_YEAR}: route cell = emission >= {route_thr:.6g} (percentile {ROUTE_PCTL} of the ocean cells with emission); "
                  f"route present = route cells over at least {ROUTE_FRAC:.0%} of the window's valid cells (alternative {ROUTE_FRAC_ALT:.0%}); "
                  f"background variability < {thr['bvar']:.4f} (quantile {Q_VAR:.3f}; alternative {thr['bvar_alt']:.4f} at {Q_VAR_ALT:.3f}); "
                  f"nitrate < {N_LIMIT}; alternative route definition: emission gradient >= {thr['grad_alt']:.4f} (quantile {Q_GRAD_ALT:.3f})")
    checks.append(f"{REF_YEAR}: route present in {100 * (ref['route_frac'] >= ROUTE_FRAC).mean():.1f}% of windows and {100 * (ref.loc[ref['positive'], 'route_frac'] >= ROUTE_FRAC).mean():.1f}% of retained positive windows")

    def conds(w, frac=ROUTE_FRAC, v=thr["bvar"], grad_thr=None):
        nl = w["nlim"].to_numpy(dtype=float)
        rt = ((w["grad"].to_numpy() >= grad_thr) if grad_thr is not None else (w["route_frac"].to_numpy() >= frac)).astype(float)
        bv = (w["bvar"].to_numpy() < v).astype(float)
        allc = np.where(np.isnan(nl), np.nan, (nl > 0) & (rt > 0) & (bv > 0))
        return {"Nitrogen limitation": nl, "Route present": rt, "Low background variability": bv, "All three conditions": allc}

    rows = []
    for y in YEARS:
        w = W[y]; cs = conds(w)
        for name, c in cs.items(): rows.append(enrichment(w, c, name, y))
        w2 = w.copy()
        for name, c in cs.items(): w2[f"c_{name}"] = c
        w2.to_csv(os.path.join(INT_DIR, f"windows_conditions_{y}.csv"), index=False)
    long = pd.DataFrame(rows); long.to_csv(os.path.join(INT_DIR, "ratio_long.csv"), index=False)

    A = long[long["year"] == REF_YEAR]
    tA = pd.DataFrame({
        "Condition": A["condition"],
        "Windows entering the screen with data": A["windows"].map(f0),
        "Meeting the condition": A["meeting"].map(f0),
        "Fraction by count": A["base_count"].map(f3), "Fraction by area": A["base_area"].map(f3),
        "Retained windows with a positive slope": A["positive"].map(f0),
        "Of these, meeting the condition": A["positive_meeting"].map(f0),
        "Fraction by count ": A["cond_count"].map(f3), "Fraction by area ": A["cond_area"].map(f3),
        "Retained-to-all ratio by count": A["enrich_count"].map(f2), "Retained-to-all ratio by area": A["enrich_area"].map(f2),
        "Odds ratio (nominal one-sided p)": [f"{o:.2f} ({p:.1e})" if p > 0 else f"{o:.2f} (< 1e-300)" for o, p in zip(A["odds_ratio"], A["p_nominal"])]})
    tA.to_csv(os.path.join(OUT_DIR, "Table_S13A_conditions_2017.csv"), index=False, encoding="utf-8-sig")

    B = long.pivot(index="condition", columns="year", values=["enrich_count", "enrich_area"])
    tB = pd.DataFrame({"Condition": list(conds(ref).keys())})
    for y in YEARS:
        tB[y] = [f"{B.loc[c, ('enrich_count', y)]:.2f} ({B.loc[c, ('enrich_area', y)]:.2f})" for c in tB["Condition"]]
    tB.to_csv(os.path.join(OUT_DIR, "Table_S13C_ratio_by_year.csv"), index=False, encoding="utf-8-sig")

    def n_met(w):
        cs = conds(w); d = w[np.isfinite(w["nlim"])]
        n = (cs["Nitrogen limitation"] + cs["Route present"] + cs["Low background variability"])[np.isfinite(w["nlim"].to_numpy())]
        return d, n.astype(int)
    d17, n17 = n_met(ref); pos17 = d17["positive"].to_numpy().astype(bool)
    others = [n_met(W[y]) for y in YEARS if y != REF_YEAR]
    n_oth = np.concatenate([n for _, n in others]); pos_oth = np.concatenate([d["positive"].to_numpy().astype(bool) for d, _ in others])
    rowsB = []
    for k in range(4):
        sw, sp = (n17 == k).mean(), (n17[pos17] == k).mean(); swo, spo = (n_oth == k).mean(), (n_oth[pos_oth] == k).mean()
        rowsB.append([str(k), f1(100 * sw), f1(100 * sp), f2(sp / sw) if sw else "", f1(100 * swo), f1(100 * spo), f2(spo / swo) if swo else ""])
    pd.DataFrame(rowsB, columns=["Conditions met", f"Windows entering the screen, {REF_YEAR} (%)", f"Retained windows with a positive slope, {REF_YEAR} (%)", f"Relative rate, {REF_YEAR}",
                                 "Windows entering the screen, 2013 to 2016 (%)", "Retained windows with a positive slope, 2013 to 2016 (%)", "Relative rate, 2013 to 2016"]
                 ).to_csv(os.path.join(OUT_DIR, "Table_S13B_by_number_of_conditions.csv"), index=False, encoding="utf-8-sig")

    rowsC = []
    for y in YEARS:
        w = W[y]
        ct = conds(w, frac=ROUTE_FRAC_ALT); cg = conds(w, grad_thr=thr["grad_alt"]); cv = conds(w, v=thr["bvar_alt"])
        for lab, c in [(f"Route present, route cells over at least {ROUTE_FRAC_ALT:.0%} of the window", ct["Route present"]),
                       (f"All three, route present over at least {ROUTE_FRAC_ALT:.0%}", ct["All three conditions"]),
                       (f"Route defined by emission gradient at or above the {REF_YEAR} median", cg["Route present"]),
                       (f"All three, route defined by emission gradient", cg["All three conditions"]),
                       (f"Low background variability, threshold at quantile {Q_VAR_ALT:.2f}", cv["Low background variability"]),
                       (f"All three, variability threshold at quantile {Q_VAR_ALT:.2f}", cv["All three conditions"])]:
            rowsC.append(enrichment(w, c, lab, y))
    C_ = pd.DataFrame(rowsC); tC = pd.DataFrame({"Condition and threshold": C_["condition"].unique()})
    for y in YEARS:
        sub = C_[C_["year"] == y].set_index("condition")
        tC[y] = [f"{sub.loc[c, 'enrich_count']:.2f} ({sub.loc[c, 'enrich_area']:.2f})" for c in tC["Condition and threshold"]]
    tC.to_csv(os.path.join(OUT_DIR, "Table_S13D_sensitivity.csv"), index=False, encoding="utf-8-sig")

    rowsD = []; overall = ref["positive"].mean()
    for lab, col, bins in [("Emission gradient within the window", "grad", GRAD_BINS), ("Background variability of chlorophyll a", "bvar", VAR_BINS)]:
        cut = pd.cut(ref[col], bins, include_lowest=True, right=True)
        for iv, g in ref.groupby(cut, observed=True):
            lo, hi = iv.left, iv.right
            rng = f"{max(lo, 0):.3g} to {hi:.3g}" if np.isfinite(hi) else f"above {lo:.3g}"
            rowsD.append([lab, rng, f0(len(g)), f2(100 * g["positive"].mean()), f2(g["positive"].mean() / overall)])
    pd.DataFrame(rowsD, columns=["Quantity", "Range", "Windows entering the screen", "Retained with a positive slope (%)", "Relative to all windows"]
                 ).to_csv(os.path.join(OUT_DIR, "Table_S13E_rate_by_bin.csv"), index=False, encoding="utf-8-sig")

    mask = load_mask(); w = ref.copy(); cs = conds(w)
    for name, c in cs.items(): w[name] = c
    w["basin"] = assign_basin(w["Latitude"].to_numpy(), w["Longitude"].to_numpy(), mask)
    d = w[np.isfinite(w["nlim"])].copy()
    rows = []
    for b in [x for x in BASIN_ORDER if x != "Unmapped"] + ["Global ocean"]:
        g = d if b == "Global ocean" else d[d["basin"] == b]
        if not len(g): continue
        pos = g["positive"].to_numpy().astype(bool); fav = g["All three conditions"].to_numpy().astype(bool)
        base = fav.mean(); cond = fav[pos].mean() if pos.sum() else np.nan
        rows.append([b, f0(len(g)), f1(100 * g["Nitrogen limitation"].mean()), f1(100 * g["Route present"].mean()),
                     f1(100 * g["Low background variability"].mean()), f1(100 * base), f0(pos.sum()),
                     f1(100 * cond) if np.isfinite(cond) else "—", f2(cond / base) if np.isfinite(cond) and base > 0 else "—"])
    tS14 = pd.DataFrame(rows, columns=["Basin", "Windows entering the screen", "Nitrogen-limited windows (%)", "Windows with a route (%)",
                                       "Windows meeting the low-variability condition (%)", "Windows meeting all three conditions (%)",
                                       "Retained windows with a positive slope", "Of these, in windows meeting all three conditions (%)", "Retained-to-all ratio"])
    tS14.to_csv(os.path.join(OUT_DIR, "Table_S14_conditions_by_basin.csv"), index=False, encoding="utf-8-sig")

    nl = A[A["condition"] == "Nitrogen limitation"].iloc[0]
    checks.append(f"{REF_YEAR} nitrogen limitation: {nl['windows']:,} windows with a nitrate value, {nl['positive']:,} retained positive, {nl['positive_meeting']:,} of them limited; "
                  f"retained-to-all ratio {nl['enrich_count']:.3f} by count, {nl['enrich_area']:.3f} by area (Table S2: 115,348 / 4,070 / 2,966; 1.306 / 1.225)")
    for y in YEARS:
        s = long[(long["year"] == y) & (long["condition"] == "All three conditions")].iloc[0]
        checks.append(f"{y}: all three conditions met by {100 * s['base_count']:.1f}% of windows (by count) and {100 * s['cond_count']:.1f}% of retained positive windows; retained-to-all ratio {s['enrich_count']:.2f} ({s['enrich_area']:.2f} by area)")
    gsum = ref.groupby(assign_basin(ref["Latitude"].to_numpy(), ref["Longitude"].to_numpy(), mask))["grad"].median()
    checks.append("median emission gradient by basin (2017): " + "; ".join(f"{b} {gsum.get(b, np.nan):.3f}" for b in BASIN_ORDER))
    open(os.path.join(OUT_DIR, "checks.txt"), "w", encoding="utf-8").write("\n".join(checks) + "\n")
    print("\n".join(checks))

    draw_figure(W, thr, conds)
    print(f"[Done] Fig. S31, Tables S13 and S14 in {OUT_DIR}/")

def add_gridlines(ax, label_size=11, lon_step=60):
    """Labelled graticule, as in Fig."""
    import matplotlib.ticker as mticker
    kw = dict(linewidth=0.4, color="gray", alpha=0.55, linestyle=":",
              xlocs=mticker.FixedLocator(range(-180, 181, lon_step)), ylocs=mticker.FixedLocator(range(-90, 91, 30)))
    try:
        gl = ax.gridlines(draw_labels=True, **kw)
        gl.top_labels = False; gl.right_labels = False
        gl.xlabel_style = {"size": label_size}; gl.ylabel_style = {"size": label_size}
    except Exception as e:
        print(f"[warn] graticule labels unavailable ({e}); lines only"); ax.gridlines(draw_labels=False, **kw)

def draw_figure(W, thr, conds):
    if FONT_FAMILY:
        plt.rcParams.update({"font.family": "serif", "font.serif": [FONT_FAMILY, "Liberation Serif", "DejaVu Serif"],
                             "mathtext.fontset": "stix", "font.size": 13, "pdf.fonttype": 42})
    ref = W[REF_YEAR]; cs = conds(ref)
    n_met = np.nansum(np.stack([cs["Nitrogen limitation"], cs["Route present"], cs["Low background variability"]]), axis=0)
    n_met = np.where(np.isnan(cs["Nitrogen limitation"]), np.nan, n_met)
    C, tf = read_field(CHLOR(REF_YEAR)); shape = C.shape
    col, row = (~tf) * (ref["Longitude"].to_numpy(), ref["Latitude"].to_numpy()); ii, jj = np.floor(row).astype(int), np.floor(col).astype(int)
    grid = np.full(shape, np.nan); grid[ii, jj] = n_met
    xs = np.array([(tf * (j, 0))[0] for j in range(shape[1] + 1)]); ys = np.array([(tf * (0, i))[1] for i in range(shape[0] + 1)])
    try:
        import cartopy.crs as ccrs, cartopy.feature as cfeature
        proj, data_crs = ccrs.Robinson(), ccrs.PlateCarree()
    except Exception:
        ccrs = None; proj = None; data_crs = None
    try:
        import geopandas as gpd; lanes = gpd.read_file(LANE_SHP) if os.path.exists(LANE_SHP) else None
    except Exception:
        lanes = None
    from matplotlib.patches import Patch
    fig = plt.figure(figsize=(9, 9.2))
    cmap4 = mcolors.ListedColormap(["#f7f7f7", "#c6dbef", "#6baed6", "#08519c"]); norm4 = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4)
    panels = [("a", ref[ref["positive"]], f"Regions by the number of conditions met in {REF_YEAR}, with the positive-correlation area of {REF_YEAR}"),
              ("b", pd.concat([W[y][W[y]["positive"]] for y in YEARS if y != REF_YEAR]), f"Regions by the number of conditions met in {REF_YEAR}, with the positive-correlation area of 2013 to 2016 overlaid")]
    m = None
    for k, (letter, pts, title) in enumerate(panels):
        ax = fig.add_axes([0.03, 0.57 - 0.45 * k, 0.94, 0.38], projection=proj) if proj is not None else fig.add_axes([0.06, 0.57 - 0.45 * k, 0.90, 0.38])
        if proj is not None:
            ax.set_global(); ax.add_feature(cfeature.LAND, facecolor="lightgray"); ax.add_feature(cfeature.OCEAN, facecolor="white")
            ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
            add_gridlines(ax)
            kw = dict(transform=data_crs)
        else:
            ax.set_xlim(-180, 180); ax.set_ylim(-90, 90); kw = {}
        m = ax.pcolormesh(xs, ys, grid, cmap=cmap4, norm=norm4, shading="flat", **kw)
        if lanes is not None and proj is not None:
            ax.add_geometries(lanes.geometry, crs=data_crs, facecolor="none", edgecolor="#7b3294", linewidth=0.3, alpha=0.5, zorder=4)
        ax.scatter(pts["Longitude"], pts["Latitude"], s=1.2, color="#e66101", zorder=5, linewidths=0, **kw)
        ax.set_title(title, fontsize=13, loc="center")
        rise = mtransforms.ScaledTranslation(0, PANEL_LETTER_RISE / 72, fig.dpi_scale_trans)
        ax.text(PANEL_LETTER_X, 1.0, letter, transform=ax.transAxes + ax.titleOffsetTrans + rise, fontsize=14, fontweight="bold", ha="left", va="baseline")
    bar_h = 0.014
    cax = fig.add_axes([0.12, 0.065, 0.40, bar_h])
    cb = fig.colorbar(m, cax=cax, orientation="horizontal", ticks=[0, 1, 2, 3]); cb.set_label(f"Number of conditions met in {REF_YEAR}", fontsize=12)
    cb.outline.set_linewidth(0.5); cax.tick_params(length=2, width=0.5, labelsize=11)
    from matplotlib.patches import Rectangle
    lax = fig.add_axes([0.62, 0.065, 0.035, bar_h]); lax.set_xlim(0, 1); lax.set_ylim(0, 1); lax.axis("off")
    lax.add_patch(Rectangle((0, 0), 1, 1, facecolor="#e66101", edgecolor="none"))
    fig.text(0.665, 0.065 + bar_h / 2, "Positive-correlation area", fontsize=12, va="center", ha="left")
    fig.savefig(os.path.join(OUT_DIR, "Fig_S31_conditions_map.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "Fig_S31_conditions_map.pdf"), bbox_inches="tight"); plt.close(fig)
    print(f"[Saved] {os.path.join(OUT_DIR, 'Fig_S31_conditions_map.png')}")

if __name__ == "__main__":
    main()
