import calendar
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import rasterio
from scipy.stats import linregress, spearmanr
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAVE_CARTOPY = True
except Exception:
    HAVE_CARTOPY = False

YEAR = 2017
CHLOR_ANNUAL = f"data/chlor/chlor_{YEAR}.tif"
EMISSION_ANNUAL = f"data/emission/NO_{YEAR}_yearly.tif"
CHLOR_MONTHLY_DIR = "data/monthly_data/chlor_data_monthly"
NOX_MONTHLY_DIR = "data/monthly_data/nox_monthly"
OUT_DIR = os.path.join("results", "Table_S6_Fig_S6_to_S19")
FIRST_FIG = 6

SEASONS = {"Summer (Jun-Aug)": [(YEAR, 6), (YEAR, 7), (YEAR, 8)],
           "Winter (Dec-Feb)": [(YEAR, 1), (YEAR, 2), (YEAR, 12)]}
MIN_VALID_MONTHS_IN_SEASON = 2
EMISSION_SCALE = "annual_equivalent"
CHLOR_SENTINEL = 99999.0

R = 6_371_000
WINDOW_SIZE = 11
MIN_VALID = 70
R2_SCREEN = 0.6
SLOPE_ABS_MIN = 0.01

RESULTS_CSV = f"surrogate_102_{YEAR}/results_{YEAR}.csv"


FONT = "Times New Roman"
FONT_SIZE = 13
FORMATS = ("png", "pdf")
R_LOW = np.sqrt(R2_SCREEN)
R_COLORS = ["#0000cc", "#8080ff", "#b3b3ff", "#ffb3b3", "#ff8080", "#cc0000"]
LAND_COLOR = "white"
LANE_SHP = "data/shipping_lane/mygeodata/Shipping_Lanes_v1.shp"
LANE_COLOR = "#7b3294"
LANE_TYPE_FIELD = "Type"
LANE_STYLE = {"core": {"linewidth": 0.8, "alpha": 0.9}, "secondary": {"linewidth": 0.5, "alpha": 0.6},
              "minor": {"linewidth": 0.3, "alpha": 0.4}, "default": {"linewidth": 0.3, "alpha": 0.4}}
LANE_CLASS_ALIASES = {"major": "core", "main": "core", "middle": "secondary", "medium": "secondary", "small": "minor"}
LANE_LEGEND = [("core", "Core Shipping Route"), ("secondary", "Secondary Shipping Route"), ("minor", "Minor Shipping Route")]
N_SHP = "data/n_limited_shp/Nmean60m_all_from_csv.shp"
N_FIELD = "meanN60"
N_CELL = 1.0
N_STRETCH = "histeq_gamma"
N_GAMMA = 1.0
N_DARK = "#a6a6a6"
N_LABEL = "Mean nitrate concentration\nin the upper 60 m (µmol kg⁻¹)"

plt.rcParams.update({"font.family": FONT, "font.size": FONT_SIZE, "axes.titlesize": FONT_SIZE,
                     "legend.fontsize": FONT_SIZE, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": f"{FONT}:italic",
                     "mathtext.bf": f"{FONT}:bold"})
NOX = "NO$_{\\mathrm{x}}$"


def have_natural_earth():
    """True when cartopy can supply the land and coastline features."""
    if not HAVE_CARTOPY:
        print("[note] cartopy not available; drawing plain longitude-latitude frames"); return False
    try:
        list(cfeature.LAND.geometries()); list(cfeature.COASTLINE.geometries()); return True
    except Exception as e:
        print(f"[note] coastline data not available ({type(e).__name__}); land and coastlines skipped"); return False


def lane_class(value):
    v = str(value).strip().lower(); return LANE_CLASS_ALIASES.get(v, v)


def load_lane_shapes():
    """The shipping-route shapefile in WGS84, or None when it is absent."""
    if not os.path.exists(LANE_SHP):
        print(f"[warn] routes shapefile not found ({LANE_SHP}); maps drawn without routes"); return None
    lanes = gpd.read_file(LANE_SHP)
    if lanes.crs is not None and lanes.crs.to_epsg() != 4326:
        lanes = lanes.to_crs(4326)
    print(f"[lanes] {len(lanes)} features; classes {lanes[LANE_TYPE_FIELD].value_counts().to_dict() if LANE_TYPE_FIELD in lanes.columns else 'n/a'}")
    return lanes


def draw_lanes(ax, lanes, zorder):
    if LANE_TYPE_FIELD in lanes.columns:
        for cls, g in lanes.groupby(LANE_TYPE_FIELD):
            st = LANE_STYLE.get(lane_class(cls), LANE_STYLE["default"])
            ax.add_geometries(g.geometry, crs=ccrs.PlateCarree(), facecolor="none", edgecolor=LANE_COLOR, zorder=zorder, **st)
    else:
        ax.add_geometries(lanes.geometry, crs=ccrs.PlateCarree(), facecolor="none", edgecolor=LANE_COLOR, zorder=zorder, **LANE_STYLE["default"])


def r_class(r):
    """Index of the six r classes: three per sign, split at 0.8 and 0.9."""
    if r < 0:
        return 0 if r <= -0.9 else 1 if r <= -0.8 else 2
    return 5 if r >= 0.9 else 4 if r >= 0.8 else 3


def load_n_grid():
    """Rasterise the WOA23 nitrate point shapefile on a 1-degree grid; returns (grid, lon_edges, lat_edges, max)."""
    if not os.path.exists(N_SHP):
        sys.exit(f"[ABORT] nitrate shapefile not found: {N_SHP}")
    pts = gpd.read_file(N_SHP)
    if pts.crs is not None and pts.crs.to_epsg() != 4326:
        pts = pts.to_crs(4326)
    if N_FIELD not in pts.columns:
        sys.exit(f"[ABORT] {N_SHP}: field '{N_FIELD}' not found in {list(pts.columns)}")
    pts = pts[pts.geometry.notna() & pts[N_FIELD].notna()]
    lon = pts.geometry.x.to_numpy(float); lat = pts.geometry.y.to_numpy(float); val = pts[N_FIELD].to_numpy(float)
    ncols, nrows = int(round(360 / N_CELL)), int(round(180 / N_CELL))
    jj = np.clip(np.floor((lon + 180) / N_CELL).astype(int), 0, ncols - 1)
    ii = np.clip(np.floor((90 - lat) / N_CELL).astype(int), 0, nrows - 1)
    acc = np.zeros((nrows, ncols)); cnt = np.zeros((nrows, ncols), dtype=int)
    np.add.at(acc, (ii, jj), val); np.add.at(cnt, (ii, jj), 1)
    grid = np.full((nrows, ncols), np.nan); ok = cnt > 0; grid[ok] = acc[ok] / cnt[ok]
    if (cnt > 1).any():
        print(f"[N] note: {int((cnt > 1).sum())} cells hold more than one point; their mean is used")
    lon_e = -180 + np.arange(ncols + 1) * N_CELL; lat_e = 90 - np.arange(nrows + 1) * N_CELL
    print(f"[N] {N_SHP}: {len(pts):,} points, field '{N_FIELD}', {N_CELL:g}-degree grid, {int(ok.sum()):,} cells, max {np.nanmax(grid):.3f}")
    return grid, lon_e, lat_e, float(np.nanmax(grid))


def n_stretched(grid):
    """Map the nitrate field to [0, 1] for the grey ramp (histogram equalisation, gamma or linear)."""
    out = np.full(grid.shape, np.nan); ok = np.isfinite(grid)
    if N_STRETCH == "linear":
        out[ok] = grid[ok] / np.nanmax(grid)
    elif N_STRETCH == "gamma":
        out[ok] = (grid[ok] / np.nanmax(grid)) ** N_GAMMA
    else:
        v = np.sort(grid[ok]); cdf = np.arange(1, len(v) + 1) / len(v)
        out[ok] = np.interp(grid[ok], v, cdf) ** N_GAMMA
    return out


def draw_legend_strip(fig, n_cmap, n_max, lanes_present, fp):
    """Legend strip below the map: r blocks with rotated labels, the nitrate ramp and the three route classes."""
    lax = fig.add_axes([0.03, 0.055, 0.94, 0.155]); lax.set_xlim(0, 1); lax.set_ylim(0, 1); lax.axis("off")
    top = 0.92
    low = f"{R_LOW:.2f}"
    blocks = [(R_COLORS[3], f"{low} - 0.8"), (R_COLORS[4], "0.8 - 0.9"), (R_COLORS[5], "> 0.9"), None,
              (R_COLORS[2], f"-0.8 - -{low}"), (R_COLORS[1], "-0.9 - -0.8"), (R_COLORS[0], "< -0.9")]
    x, w, h = 0.04, 0.045, 0.17
    for b in blocks:
        if b is None:
            x += 0.035; continue
        color, label = b
        lax.add_patch(patches.Rectangle((x, top - h), w, h, facecolor=color, edgecolor="none"))
        lax.text(x + w / 2, top - h - 0.06, label, ha="right", va="top", rotation=45, rotation_mode="anchor", **fp)
        x += w
    xn = 0.50
    lax.text(xn, top, N_LABEL, ha="left", va="top", linespacing=1.3, **fp)
    bx, by, bw, bh = xn + 0.02, 0.06, 0.035, 0.42
    lax.imshow(np.linspace(0, 1, 64)[:, None], extent=(bx, bx + bw, by, by + bh), origin="lower", cmap=n_cmap,
               aspect="auto", zorder=2, vmin=0, vmax=1)
    lax.add_patch(patches.Rectangle((bx, by), bw, bh, facecolor="none", edgecolor="#999999", linewidth=0.4, zorder=3))
    lax.text(bx + bw + 0.008, by + bh, f"{n_max:.1f}", ha="left", va="center", **fp)
    lax.text(bx + bw + 0.008, by, "0", ha="left", va="center", **fp)
    if lanes_present:
        for i, (cls, label) in enumerate(LANE_LEGEND):
            yy = top - 0.04 - i * 0.24
            lax.plot([0.72, 0.765], [yy, yy], color=LANE_COLOR, **LANE_STYLE[cls])
            lax.text(0.775, yy, label, ha="left", va="center", **fp)


def draw_r_map(scr, title, rows, cols, n_layers, lanes, out_base, have_ne):
    """One window map: r classes of the screened windows (columns row, col, R) over the nitrate basemap and the routes."""
    n_grid, n_lon_e, n_lat_e, n_max = n_layers
    n_cmap = LinearSegmentedColormap.from_list("n60", ["#ffffff", N_DARK])
    field = np.full((rows, cols), np.nan)
    for r_, c_, rv in zip(scr["row"], scr["col"], scr["R"]):
        field[int(r_), int(c_)] = r_class(float(rv))
    lon_e = np.linspace(-180, 180, cols + 1); lat_e = np.linspace(90, -90, rows + 1)
    fp = {"family": FONT, "size": FONT_SIZE}
    fig = plt.figure(figsize=(12, 7.8))
    if HAVE_CARTOPY:
        ax = fig.add_axes([0.02, 0.235, 0.96, 0.755], projection=ccrs.Robinson()); ax.set_global(); kw = {"transform": ccrs.PlateCarree()}
    else:
        ax = fig.add_axes([0.04, 0.235, 0.92, 0.755]); ax.set_xlim(-180, 180); ax.set_ylim(-90, 90); ax.set_aspect("equal"); kw = {}
    ax.pcolormesh(n_lon_e, n_lat_e, np.ma.masked_invalid(n_stretched(n_grid)), cmap=n_cmap, vmin=0, vmax=1, shading="flat", zorder=1, **kw)
    if HAVE_CARTOPY and have_ne:
        ax.add_feature(cfeature.LAND, facecolor=LAND_COLOR, zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=5)
    if HAVE_CARTOPY and lanes is not None:
        draw_lanes(ax, lanes, zorder=3)
    cmap = ListedColormap(R_COLORS); norm = BoundaryNorm(np.arange(-0.5, 6.5, 1), cmap.N)
    ax.pcolormesh(lon_e, lat_e, np.ma.masked_invalid(field), cmap=cmap, norm=norm, shading="flat", zorder=4, **kw)
    if HAVE_CARTOPY:
        gl = ax.gridlines(draw_labels=True, xlocs=range(-180, 181, 60), ylocs=[-60, -30, 0, 30, 60],
                          linewidth=0.5, color="gray", alpha=0.5, linestyle="--", zorder=6)
        gl.top_labels = False; gl.xlabel_style = dict(fp); gl.ylabel_style = dict(fp)
    ax.set_title(title, fontsize=FONT_SIZE, fontfamily=FONT)
    draw_legend_strip(fig, n_cmap, n_max, lanes is not None, fp)
    for ext in FORMATS:
        fig.savefig(f"{out_base}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def pixel_area_m2(lat_deg, xres_deg, yres_deg):
    """Surface area of one grid cell at a latitude."""
    lat = np.deg2rad(lat_deg); dlat = np.deg2rad(yres_deg); dlon = np.deg2rad(xres_deg)
    return R**2 * dlon * (np.sin(lat + dlat/2) - np.sin(lat - dlat/2))


def perform_moving_window_regression(chl, em, tf, window_size=WINDOW_SIZE):
    """Regress log1p chlorophyll on log1p emission in every window, using the cells valid in both fields and pairing them cell by cell."""
    rows, cols = chl.shape
    half = window_size // 2
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        keep = np.isfinite(chl) & (chl > 0) & np.isfinite(em) & (em > 0)
        lc, le = np.log1p(chl), np.log1p(em)
    results = []
    for i in range(rows):
        r0, r1 = max(i - half, 0), min(i + half + 1, rows)
        for j in range(cols):
            c0, c1 = max(j - half, 0), min(j + half + 1, cols)
            k = keep[r0:r1, c0:c1].ravel()
            n = int(k.sum())
            if n < MIN_VALID:
                continue
            a_vals = lc[r0:r1, c0:c1].ravel()[k]
            b_vals = le[r0:r1, c0:c1].ravel()[k]
            if np.all(a_vals == a_vals[0]):
                continue
            if np.all(b_vals == b_vals[0]):
                continue
            slope, intercept, r_value, p_value, _ = linregress(b_vals, a_vals)
            lon, lat = tf * (j + 0.5, i + 0.5)
            results.append({"row": i, "col": j, "Longitude": lon, "Latitude": lat,
                            "Intercept": intercept, "Slope": slope,
                            "R": r_value, "R2": r_value**2,
                            "P_Value": p_value, "Count": n,
                            "floor_ok": bool(abs(slope) >= SLOPE_ABS_MIN)})
    return pd.DataFrame(results)


def seconds_in_month(y, m):
    return calendar.monthrange(y, m)[1] * 24 * 3600


def seconds_in_year(y):
    return (366 if calendar.isleap(y) else 365) * 24 * 3600


def chlor_path_for(y, m):
    return os.path.join(CHLOR_MONTHLY_DIR, f"chlor_{y}_{m:02d}.tiff")


def nox_path_for(y, m):
    return os.path.join(NOX_MONTHLY_DIR, f"NOxFlux_{y}_{m:02d}.tif")


def read_masked(path, sentinel=None):
    """Read band 1 as float64 with the declared nodata (and an optional sentinel) set to NaN."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float64")
        nodata, tf, shape, crs = src.nodata, src.transform, src.shape, src.crs
    n_nodata = 0
    if nodata is not None and np.isfinite(nodata):
        n_nodata = int((arr == nodata).sum()); arr[arr == nodata] = np.nan
    n_sent = 0
    if sentinel is not None:
        n_sent = int((arr == sentinel).sum()); arr[arr == sentinel] = np.nan
    return arr, tf, shape, crs, {"nodata": nodata, "n_nodata": n_nodata, "n_sentinel": n_sent}


def same_grid(tf_a, shape_a, tf_b, shape_b, label):
    """Both fields must sit on one grid."""
    if shape_a != shape_b:
        sys.exit(f"[ABORT] {label}: shapes differ {shape_a} vs {shape_b}")
    if tf_a == tf_b:
        return
    ident = rasterio.Affine.identity()
    if tf_a == ident or tf_b == ident:
        print(f"[note] {label}: one raster has no georeference; equal shapes, paired by index")
        return
    sys.exit(f"[ABORT] {label}: transforms differ\n  {tf_a}\n  {tf_b}")


def month_pair(y, m, ref_tf, ref_shape):
    """Chlorophyll (mg m-3) and emission (annual-equivalent g m-2 yr-1, or month total) of one month."""
    pc, pe = chlor_path_for(y, m), nox_path_for(y, m)
    for p in (pc, pe):
        if not os.path.exists(p):
            sys.exit(f"[ABORT] missing input {p}")
    chl, tfc, shc, _, ic = read_masked(pc, sentinel=CHLOR_SENTINEL)
    em, tfe, she, _, ie = read_masked(pe)
    same_grid(ref_tf, ref_shape, tfc, shc, f"{y}-{m:02d} chlorophyll vs annual grid")
    same_grid(ref_tf, ref_shape, tfe, she, f"{y}-{m:02d} emission vs annual grid")
    with np.errstate(invalid="ignore"):
        em = np.where(np.isfinite(em) & (em >= 0), em, np.nan)
    return chl, em, ic, ie


def emission_scale(y, m):
    return 1000.0 * (seconds_in_year(y) if EMISSION_SCALE == "annual_equivalent" else seconds_in_month(y, m))


def season_composite(months, ref_tf, ref_shape):
    """Seasonal composites, each field following its own annual convention."""
    H, W = ref_shape
    c_sum = np.zeros((H, W)); c_cnt = np.zeros((H, W), dtype=int)
    e_sum = np.zeros((H, W)); e_w = np.zeros((H, W)); e_cnt = np.zeros((H, W), dtype=int)
    for (y, m) in months:
        chl, em, _, _ = month_pair(y, m, ref_tf, ref_shape)
        sec = float(seconds_in_month(y, m))
        cv = np.isfinite(chl) & (chl > 0)
        c_sum[cv] += chl[cv]; c_cnt[cv] += 1
        ev = np.isfinite(em)
        e_sum[ev] += em[ev] * sec; e_w[ev] += sec; e_cnt[ev] += 1
    chl_out = np.full((H, W), np.nan); ok = c_cnt >= MIN_VALID_MONTHS_IN_SEASON
    chl_out[ok] = c_sum[ok] / c_cnt[ok]
    em_out = np.full((H, W), np.nan); ok = e_cnt >= MIN_VALID_MONTHS_IN_SEASON
    em_out[ok] = e_sum[ok] / e_w[ok] * 1000.0 * seconds_in_year(YEAR)
    return chl_out, em_out


def summarise_tier(label, months, df, chl, em, xres, yres):
    """Part A row of one tier."""
    ent = df[df["floor_ok"]]
    scr = ent[ent["R2"] >= R2_SCREEN]
    pos = scr[scr["Slope"] >= 0]; neg = scr[scr["Slope"] < 0]
    area = lambda d: float(pixel_area_m2(d["Latitude"].to_numpy(), xres, yres).sum() / 1e6) if len(d) else 0.0
    with np.errstate(invalid="ignore"):
        vc = int((np.isfinite(chl) & (chl > 0)).sum()); ve = int((np.isfinite(em) & (em > 0)).sum())
        vb = int((np.isfinite(chl) & (chl > 0) & np.isfinite(em) & (em > 0)).sum())
    return {"tier": label, "months": months,
            "fitted": len(df), "below_floor": int((~df["floor_ok"]).sum()), "entering": len(ent),
            "retained": len(scr), "positive": len(pos), "negative": len(neg),
            "screened_window_rate": len(scr) / len(ent) if len(ent) else np.nan,
            "positive_fraction": len(pos) / len(scr) if len(scr) else np.nan,
            "pos_area_km2": area(pos), "neg_area_km2": area(neg),
            "median_valid_cells_fitted": float(df["Count"].median()) if len(df) else np.nan,
            "median_R2_fitted": float(df["R2"].median()) if len(df) else np.nan,
            "median_R2_entering": float(ent["R2"].median()) if len(ent) else np.nan,
            "valid_chl_cells": vc, "valid_emission_cells": ve, "valid_both_cells": vb}


def overlap_row(label, df_a, df_t):
    """Part B row: overlap of a tier's retained set with the annual retained set on their joint fitted domain."""
    key = lambda d: set(zip(d["row"], d["col"]))
    dom = key(df_a) & key(df_t)
    scr_a = df_a[df_a["floor_ok"] & (df_a["R2"] >= R2_SCREEN)]
    scr_t = df_t[df_t["floor_ok"] & (df_t["R2"] >= R2_SCREEN)]
    sign_a = {k: np.sign(s) for k, s in zip(zip(scr_a["row"], scr_a["col"]), scr_a["Slope"])}
    sign_t = {k: np.sign(s) for k, s in zip(zip(scr_t["row"], scr_t["col"]), scr_t["Slope"])}
    ra = {k for k in sign_a if k in dom}; rt = {k for k in sign_t if k in dom}
    both = ra & rt; same = {k for k in both if sign_a[k] == sign_t[k]}
    ra_p = {k for k in ra if sign_a[k] > 0}; rt_p = {k for k in rt if sign_t[k] > 0}; both_p = ra_p & rt_p
    N = len(dom)

    def jac(s1, s2):
        u = len(s1 | s2); return len(s1 & s2) / u if u else np.nan

    def jac_exp(k1, k2):
        d = N * (k1 + k2) - k1 * k2; return k1 * k2 / d if d else np.nan

    j, je = jac(ra, rt), jac_exp(len(ra), len(rt))
    jp, jpe = jac(ra_p, rt_p), jac_exp(len(ra_p), len(rt_p))
    return {"tier": label, "joint_fitted_domain_N": N,
            "annual_retained_in_domain": len(ra), "tier_retained_in_domain": len(rt),
            "retained_in_both": len(both), "retained_in_both_same_sign": len(same),
            "fraction_of_tier_retained_also_annual": len(both) / len(rt) if len(rt) else np.nan,
            "fraction_of_annual_retained_also_tier": len(both) / len(ra) if len(ra) else np.nan,
            "jaccard_all": j, "jaccard_all_expected": je, "ratio_all": j / je if je else np.nan,
            "annual_positive_in_domain": len(ra_p), "tier_positive_in_domain": len(rt_p),
            "positive_in_both": len(both_p),
            "jaccard_positive": jp, "jaccard_positive_expected": jpe, "ratio_positive": jp / jpe if jpe else np.nan}


MONTH_NAMES = {f"{m:02d}": n for m, n in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August", "September",
     "October", "November", "December"], 1)}


def tier_name(t):
    """Display name of a tier."""
    if t == "Annual":
        return "Annual"
    if t.startswith("Summer"):
        return "Summer (Jun–Aug)"
    if t.startswith("Winter"):
        return "Winter (Jan, Feb, Dec)"
    y, m = t.split("-")
    return f"{MONTH_NAMES[m]} {y}"


def _n(x):
    return f"{int(round(x)):,}"


def _f(x, d):
    return f"{x:.{d}f}"


def final_parts(pa, pb, pc):
    """The three formatted parts of Table S6 as DataFrames."""
    a = pd.DataFrame([{"Aggregation": tier_name(r["tier"]),
                       "Fitted": _n(r["fitted"]), "Below slope floor": _n(r["below_floor"]),
                       "Entering the screen": _n(r["entering"]), "Retained": _n(r["retained"]),
                       "Positive": _n(r["positive"]), "Negative": _n(r["negative"]),
                       "Screened-window rate": _f(r["screened_window_rate"], 4),
                       "Positive-correlation area (10⁶ km²)": _f(r["pos_area_km2"] / 1e6, 2),
                       "Negative-correlation area (10⁶ km²)": _f(r["neg_area_km2"] / 1e6, 2),
                       "Median R² of fitted windows": _f(r["median_R2_fitted"], 3),
                       "Cells with chlorophyll a": _n(r["valid_chl_cells"])} for _, r in pa.iterrows()])
    b = pd.DataFrame([{"Aggregation": tier_name(r["tier"]),
                       "Joint fitted domain": _n(r["joint_fitted_domain_N"]),
                       "Annual retained": _n(r["annual_retained_in_domain"]),
                       "Retained in this field": _n(r["tier_retained_in_domain"]),
                       "Retained in both": _n(r["retained_in_both"]),
                       "Same sign": _n(r["retained_in_both_same_sign"]),
                       "Fraction also retained in the annual field": _f(r["fraction_of_tier_retained_also_annual"], 3),
                       "Jaccard, all": _f(r["jaccard_all"], 3), "Expected, all": _f(r["jaccard_all_expected"], 4),
                       "Ratio, all": _f(r["ratio_all"], 1),
                       "Jaccard, positive": _f(r["jaccard_positive"], 3),
                       "Expected, positive": _f(r["jaccard_positive_expected"], 4),
                       "Ratio, positive": _f(r["ratio_positive"], 1)} for _, r in pb.iterrows()])
    c = pd.DataFrame([{"Month": f"{MONTH_NAMES[r['month'].split('-')[1]]} {r['month'].split('-')[0]}",
                       "Share of annual emission (%)": _f(100 * r["share_of_annual_emission"], 1),
                       "Cells with emission": _n(r["cells_with_emission"]),
                       "Spearman correlation with the annual field": _f(r["spearman_vs_annual_log1p"], 4),
                       "Pearson correlation with the annual field": _f(r["pearson_vs_annual_log1p"], 4)}
                      for _, r in pc.iterrows()])
    return [("(A) Counts and areas by aggregation", a),
            ("(B) Overlap with the annual retained windows on the joint fitted domain", b),
            ("(C) Emission field month to month", c)]


def write_final(parts, out_csv, out_docx):
    """Write the paste-ready CSV (parts stacked) and the Word version."""
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as fh:
        for k, (title, df) in enumerate(parts):
            fh.write(title + "\n"); df.to_csv(fh, index=False)
            if k < len(parts) - 1:
                fh.write("\n")
    print(f"[Saved] {out_csv}")
    try:
        from docx import Document
        from docx.shared import Pt
    except ImportError:
        print("[note] python-docx not installed; the Word version was skipped (pip install python-docx)")
        return
    doc = Document()
    st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(9)
    doc.add_paragraph("Table S6. Retained windows of the chlorophyll a–NOx regression at annual, seasonal and monthly "
                      "aggregation in 2017 (caption of record in the SI file).")
    for title, df in parts:
        doc.add_paragraph(title)
        t = doc.add_table(rows=1, cols=len(df.columns)); t.style = "Table Grid"
        for j, col in enumerate(df.columns):
            t.rows[0].cells[j].text = col
        for _, r in df.iterrows():
            cells = t.add_row().cells
            for j, col in enumerate(df.columns):
                cells[j].text = str(r[col])
        doc.add_paragraph("")
    doc.save(out_docx); print(f"[Saved] {out_docx}")


def main():
    os.makedirs(os.path.join(OUT_DIR, "windows"), exist_ok=True)
    for p in (CHLOR_ANNUAL, EMISSION_ANNUAL, RESULTS_CSV):
        if not os.path.exists(p):
            sys.exit(f"[ABORT] missing input {p}")

    chl_a, tf, shape, crs, ic = read_masked(CHLOR_ANNUAL)
    em_a, tfe, she, _, ie = read_masked(EMISSION_ANNUAL)
    same_grid(tf, shape, tfe, she, "annual chlorophyll vs annual emission")
    with rasterio.open(CHLOR_ANNUAL) as src:
        xres, yres = src.res
    n_sent = int((chl_a == CHLOR_SENTINEL).sum())
    print(f"[input] annual chlorophyll: shape {shape}, res {xres} x {yres}, declared nodata {ic['nodata']} "
          f"({ic['n_nodata']} cells), cells equal to {CHLOR_SENTINEL:g}: {n_sent}")
    print(f"[input] annual emission: declared nodata {ie['nodata']} ({ie['n_nodata']} cells)")
    if n_sent > 0:
        sys.exit(f"[ABORT] the annual chlorophyll raster holds {n_sent} cells equal to the NEO sentinel; the surrogate "
                 "script treats them as valid values. Stop here and report this before anything else is run.")
    if EMISSION_SCALE not in ("annual_equivalent", "month_total"):
        sys.exit("[ABORT] EMISSION_SCALE must be 'annual_equivalent' or 'month_total'")

    tiers = []
    tiers.append(("Annual", f"Jan-Dec {YEAR}", chl_a, em_a))
    for lab, months in SEASONS.items():
        chl_s, em_s = season_composite(months, tf, shape)
        tiers.append((lab, ", ".join(f"{y}-{m:02d}" for y, m in months), chl_s, em_s))
    monthly_flux = {}
    for m in range(1, 13):
        chl_m, flux, icm, iem = month_pair(YEAR, m, tf, shape)
        monthly_flux[m] = flux
        with np.errstate(invalid="ignore"):
            em_m = flux * emission_scale(YEAR, m)
        tiers.append((f"{YEAR}-{m:02d}", f"{YEAR}-{m:02d}", chl_m, em_m))
        print(f"[input] {YEAR}-{m:02d}: chlorophyll nodata {icm['nodata']} ({icm['n_nodata']} cells), "
              f"sentinel cells {icm['n_sentinel']}; emission nodata {iem['nodata']} ({iem['n_nodata']} cells)")
    print(f"[plan] {len(tiers)} tiers; emission scale for sub-annual tiers: {EMISSION_SCALE}\n")

    rows_a, frames = [], {}
    for label, months, chl, em in tiers:
        t0 = time.time()
        df = perform_moving_window_regression(chl, em, tf)
        frames[label] = df
        row = summarise_tier(label, months, df, chl, em, xres, yres); rows_a.append(row)
        tag = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower()
        df.to_csv(os.path.join(OUT_DIR, "windows", f"windows_{tag}_all.csv"), index=False, encoding="utf-8-sig")
        print(f"[{label}] fitted {row['fitted']:,} | floor {row['below_floor']:,} | entering {row['entering']:,} | "
              f"retained {row['retained']:,} (pos {row['positive']:,} / neg {row['negative']:,}) | "
              f"rate {row['screened_window_rate']:.4f} | pos area {row['pos_area_km2']:,.0f} km2 | "
              f"median n {row['median_valid_cells_fitted']:.0f} | median R2 {row['median_R2_fitted']:.3f} | "
              f"{time.time() - t0:.0f} s", flush=True)
        if label == "Annual":
            got = {"fitted": row["fitted"], "below_floor": row["below_floor"], "entering": row["entering"],
                   "retained": row["retained"], "positive": row["positive"], "negative": row["negative"],
                   "pos_area_km2": round(row["pos_area_km2"])}
            ref = pd.read_csv(RESULTS_CSV); ref = ref[ref["scenario"] == "baseline"].iloc[-1]
            expect = {"entering": int(ref["n_regressed"]), "retained": int(ref["n_screened"]), "positive": int(ref["n_pos"]),
                      "negative": int(ref["n_neg"]), "pos_area_km2": int(round(ref["pos_area_km2"]))}
            bad = {k: (got[k], v) for k, v in expect.items() if got[k] != v}
            print("[Check] annual tier against the surrogate baseline: " + ("all checked keys OK" if not bad else f"MISMATCH {bad}"))
            if bad:
                sys.exit("[ABORT] the annual tier does not reproduce the surrogate baseline of 2017; nothing below is usable.")

    rows_b = [overlap_row(label, frames["Annual"], frames[label]) for label, _, _, _ in tiers if label != "Annual"]

    rows_c = []
    with np.errstate(invalid="ignore"):
        tot = np.zeros(shape); cnt = np.zeros(shape, dtype=int)
        for m, flux in monthly_flux.items():
            v = np.isfinite(flux); tot[v] += flux[v] * seconds_in_month(YEAR, m) * 1000.0; cnt[v] += 1
        ocean = np.isfinite(em_a) & (em_a > 0)
        for m, flux in monthly_flux.items():
            em_m = flux * 1000.0 * seconds_in_year(YEAR)
            ok = ocean & np.isfinite(em_m) & (em_m > 0)
            rho = spearmanr(np.log1p(em_m[ok]), np.log1p(em_a[ok])).correlation if ok.sum() > 2 else np.nan
            pr = np.corrcoef(np.log1p(em_m[ok]), np.log1p(em_a[ok]))[0, 1] if ok.sum() > 2 else np.nan
            share = float(np.nansum(flux[ocean] * seconds_in_month(YEAR, m) * 1000.0) / np.nansum(tot[ocean]))
            rows_c.append({"month": f"{YEAR}-{m:02d}", "share_of_annual_emission": share,
                           "cells_with_emission": int(ok.sum()),
                           "spearman_vs_annual_log1p": rho, "pearson_vs_annual_log1p": pr})
        full = ocean & (cnt == 12)
        ratio = tot[full] / em_a[full]
        chk = [{"check": "sum of the 12 monthly totals / annual emission file, median over ocean cells",
                "value": float(np.median(ratio))},
               {"check": "same ratio, 5th percentile", "value": float(np.percentile(ratio, 5))},
               {"check": "same ratio, 95th percentile", "value": float(np.percentile(ratio, 95))},
               {"check": "ocean cells with emission in all 12 months", "value": int(full.sum())}]
        c_sum = np.zeros(shape); c_cnt = np.zeros(shape, dtype=int)
        for m in range(1, 13):
            chl_m = tiers[3 + m - 1][2]
            v = np.isfinite(chl_m) & (chl_m > 0); c_sum[v] += chl_m[v]; c_cnt[v] += 1
        both = np.isfinite(chl_a) & (chl_a > 0) & (c_cnt > 0)
        mean_m = c_sum[both] / c_cnt[both]
        chk += [{"check": "annual chlorophyll / mean of valid monthly chlorophyll, median over cells", "value": float(np.median(chl_a[both] / mean_m))},
                {"check": "Pearson r, annual chlorophyll vs mean of valid monthly (log1p)", "value": float(np.corrcoef(np.log1p(chl_a[both]), np.log1p(mean_m))[0, 1])},
                {"check": "cells with annual chlorophyll but no valid month", "value": int((np.isfinite(chl_a) & (chl_a > 0) & (c_cnt == 0)).sum())},
                {"check": "cells with valid monthly chlorophyll but no annual value", "value": int(((c_cnt > 0) & ~(np.isfinite(chl_a) & (chl_a > 0))).sum())},
                {"check": "median number of valid months among cells with an annual value", "value": float(np.median(c_cnt[np.isfinite(chl_a) & (chl_a > 0)]))}]

    pa = pd.DataFrame(rows_a); pb = pd.DataFrame(rows_b); pc = pd.DataFrame(rows_c); pk = pd.DataFrame(chk)
    for name, d in (("Table_S6_partA.csv", pa), ("Table_S6_partB.csv", pb),
                    ("Table_S6_partC.csv", pc), ("Table_S6_data_checks.csv", pk)):
        d.to_csv(os.path.join(OUT_DIR, name), index=False, encoding="utf-8-sig")
        print(f"[Saved] {os.path.join(OUT_DIR, name)}")

    parts = final_parts(pa, pb, pc)
    write_final(parts, os.path.join(OUT_DIR, "Table_S6_final.csv"), os.path.join(OUT_DIR, "Table_S6_final.docx"))

    # Figs. S6 to S19
    n_layers = load_n_grid(); lanes = load_lane_shapes(); have_ne = have_natural_earth()
    os.makedirs(os.path.join(OUT_DIR, "figures"), exist_ok=True)
    for k, (label, months, chl, em) in enumerate(t for t in tiers if t[0] != "Annual"):
        df = frames[label]; scr = df[df["floor_ok"] & (df["R2"] >= R2_SCREEN)]
        kind = "Seasonal" if not label[0].isdigit() else "Monthly"
        name = f"{label.split(' ')[0]} {YEAR}" if kind == "Seasonal" else label
        stem = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower()
        out_base = os.path.join(OUT_DIR, "figures", f"Fig_S{FIRST_FIG + k}_chlor_nox_{stem}_r")
        draw_r_map(scr, f"{kind} Moving-window Chlorophyll\u2013{NOX} Correlation (Pearson r), {name}; R\u00b2 \u2265 {R2_SCREEN}",
                   shape[0], shape[1], n_layers, lanes, out_base, have_ne)
        print(f"[Saved] Fig. S{FIRST_FIG + k}: {out_base}.{FORMATS[0]} ({len(scr):,} retained windows)")

    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
    print("\n================ PART A: counts by tier ================")
    print(pa[["tier", "fitted", "below_floor", "entering", "retained", "positive", "negative",
              "screened_window_rate", "positive_fraction", "pos_area_km2", "neg_area_km2",
              "median_valid_cells_fitted", "median_R2_fitted", "valid_chl_cells", "valid_both_cells"]].to_string(index=False))
    print("\n================ PART B: overlap with the annual retained set ================")
    print(pb[["tier", "joint_fitted_domain_N", "annual_retained_in_domain", "tier_retained_in_domain",
              "retained_in_both", "retained_in_both_same_sign", "fraction_of_tier_retained_also_annual",
              "jaccard_all", "jaccard_all_expected", "ratio_all", "jaccard_positive", "jaccard_positive_expected",
              "ratio_positive"]].to_string(index=False))
    print("\n================ PART C: emission field month to month ================")
    print(pc.to_string(index=False))
    print("\n================ DATA CHECKS ================")
    print(pk.to_string(index=False))
    pd.set_option("display.width", 260)
    for title, df in parts:
        print(f"\n================ TABLE S6 {title} ================")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
