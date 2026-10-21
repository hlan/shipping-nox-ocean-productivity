import os
import sys
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from scipy.stats import linregress
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

YEARS = ["2013", "2014", "2015", "2016", "2017"]
FIG_NUMBERS = {"2013": 1, "2014": 2, "2015": 3, "2016": 4, "2017": 5}
CHLOR_PATH = "data/chlor/chlor_{year}.tif"
EMISSION_PATH = "data/emission/NO_{year}_yearly.tif"
RESULTS_CSV = "surrogate_102_{year}/results_{year}.csv"
OUT_DIR = os.path.join("results", "Fig_S1_to_S5")

WINDOW_SIZE = 11
MIN_VALID = 70
R2_SCREEN = 0.6
SLOPE_ABS_MIN = 0.01

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


def perform_moving_window_regression(chlor_raster_path, emission_raster_path, window_size=WINDOW_SIZE):
    """Regress log1p chlorophyll a on log1p emission in every window over the cells valid in both fields; windows with fewer
    than MIN_VALID cells, constant values or an absolute slope below SLOPE_ABS_MIN are dropped."""
    with rasterio.open(chlor_raster_path) as src_c, rasterio.open(emission_raster_path) as src_e:
        assert src_c.transform == src_e.transform and src_c.shape == src_e.shape, "Resolution or CRS mismatch"
        chl = src_c.read(1).astype("float64"); em = src_e.read(1).astype("float64")
        for arr, nod in ((chl, src_c.nodata), (em, src_e.nodata)):
            if nod is not None and np.isfinite(nod):
                arr[arr == nod] = np.nan
        tf = src_c.transform
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
            if np.all(a_vals == a_vals[0]) or np.all(b_vals == b_vals[0]):
                continue
            slope, intercept, r_value, p_value, _ = linregress(b_vals, a_vals)
            if abs(slope) < SLOPE_ABS_MIN:
                continue
            lon, lat = tf * (j + 0.5, i + 0.5)
            results.append({"row": i, "col": j, "Longitude": lon, "Latitude": lat, "Intercept": intercept, "Slope": slope,
                            "R": r_value, "R2": r_value**2, "P_Value": p_value, "Count": n})
    return pd.DataFrame(results)


def consistency(year, df_all, df_ret):
    """Compare the counts of one year with the surrogate results CSV; report only."""
    p = RESULTS_CSV.format(year=year)
    if not os.path.exists(p):
        print(f"[Consistency {year}] {p} not found, nothing to compare"); return
    res = pd.read_csv(p); row = res[(res["year"].astype(str) == str(year)) & (res["scenario"] == "baseline")]
    if row.empty:
        print(f"[Consistency {year}] no baseline row in {p}"); return
    r = row.iloc[-1]; n_pos = int((df_ret["Slope"] >= 0).sum())
    mark = lambda name, got, ref: f"{name} {got:,} -> " + ("ok" if got == ref else f"differs from {ref:,} by {got - ref:+,}")
    print(f"[Consistency {year}] " + "; ".join([mark("entering", len(df_all), int(r["n_regressed"])), mark("retained", len(df_ret), int(r["n_screened"])),
                                               mark("positive", n_pos, int(r["n_pos"]))]))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    n_layers = load_n_grid(); lanes = load_lane_shapes(); have_ne = have_natural_earth()
    for year in YEARS:
        chlor_path, emission_path = CHLOR_PATH.format(year=year), EMISSION_PATH.format(year=year)
        missing = [p for p in (chlor_path, emission_path) if not os.path.exists(p)]
        if missing:
            raise SystemExit("[ERROR] missing input file(s):\n    " + "\n    ".join(missing))
        print(f"Processing {year} ...")
        with rasterio.open(chlor_path) as src:
            rows, cols = src.shape
        df = perform_moving_window_regression(chlor_path, emission_path)
        ret = df[df["R2"] >= R2_SCREEN].reset_index(drop=True)
        consistency(year, df, ret)
        out_base = os.path.join(OUT_DIR, f"Fig_S{FIG_NUMBERS[year]}_chl_{year}_r")
        draw_r_map(ret, f"Annual Moving-window Chlorophyll\u2013{NOX} Correlation (Pearson r), {year}; R\u00b2 \u2265 {R2_SCREEN}",
                   rows, cols, n_layers, lanes, out_base, have_ne)
        print(f"  [Saved] Fig. S{FIG_NUMBERS[year]}: {out_base}.{FORMATS[0]} ({len(ret):,} retained windows)")
    print(f"[Done] Figs. S1 to S5 in {OUT_DIR}/")


if __name__ == "__main__":
    main()
