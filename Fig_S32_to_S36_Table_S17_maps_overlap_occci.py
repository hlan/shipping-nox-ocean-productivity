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
import cartopy.crs as ccrs
import cartopy.feature as cfeature

YEARS = ["2013", "2014", "2015", "2016", "2017"]
FIG_NUMBERS = {"2013": 32, "2014": 33, "2015": 34, "2016": 35, "2017": 36}
PRODUCTS = [("a", "occci", "OC-CCI", "data/chlor_oc_cci/chlor_{year}.tif", "surrogate_102_{year}_oc_cci/results_{year}.csv"),
            ("b", "modis", "Aqua/MODIS", "data/chlor/chlor_{year}.tif", "surrogate_102_{year}/results_{year}.csv")]
EMISSION_PATH = "data/emission/NO_{year}_yearly.tif"
OUT_DIR = os.path.join("results_cci", "Fig_S32_to_S36_Table_S17")
WIN_DIR = os.path.join(OUT_DIR, "windows")

WINDOW_SIZE = 11
MIN_VALID = 70
R2_SCREEN = 0.6
SLOPE_ABS_MIN = 0.01

FONT = "Times New Roman"
FONT_SIZE = 13
FORMATS = ("png", "pdf")
FIGSIZE = (16, 6.2)
PANEL_POS = [[0.005, 0.28, 0.47, 0.66], [0.525, 0.28, 0.47, 0.66]]
LEGEND_POS = [0.15, 0.03, 0.70, 0.19]
LEGEND_N_X = 0.48
LEGEND_ROUTE_X = 0.745
PANEL_LETTER_X = 0.0
INNER_LABELS = "touch"
GAP_PX = 2
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
S17_HEADER = ["Year", "Retained windows, Aqua/MODIS, total (positive slope)", "Retained windows, OC-CCI, total (positive slope)",
              "Windows retained by both products", "Jaccard index, all retained windows", "Positive-slope windows retained by both products",
              "Jaccard index, positive-slope windows", "Same slope sign among windows retained by both products (%)"]

plt.rcParams.update({"font.family": FONT, "font.size": FONT_SIZE, "axes.titlesize": FONT_SIZE,
                     "legend.fontsize": FONT_SIZE, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": f"{FONT}:italic",
                     "mathtext.bf": f"{FONT}:bold"})
NOX = "NO$_{\\mathrm{x}}$"


def have_natural_earth():
    """True when cartopy can supply the land and coastline features."""
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



def draw_legend_strip(fig, n_cmap, n_max, lanes_present, fp):
    """Legend strip below the two maps: r blocks with rotated labels, the nitrate ramp and the three route classes."""
    lax = fig.add_axes(LEGEND_POS); lax.set_xlim(0, 1); lax.set_ylim(0, 1); lax.axis("off")
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
    xn = LEGEND_N_X
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
            lax.plot([LEGEND_ROUTE_X, LEGEND_ROUTE_X + 0.045], [yy, yy], color=LANE_COLOR, **LANE_STYLE[cls])
            lax.text(LEGEND_ROUTE_X + 0.055, yy, label, ha="left", va="center", **fp)


def draw_panel(ax, scr, letter, title, rows, cols, n_layers, n_cmap, lanes, have_ne, fp):
    """One Robinson map of the retained windows (six r classes) over the nitrate basemap and the routes."""
    n_grid, n_lon_e, n_lat_e, _ = n_layers
    field = np.full((rows, cols), np.nan)
    for r_, c_, rv in zip(scr["row"], scr["col"], scr["R"]):
        field[int(r_), int(c_)] = r_class(float(rv))
    lon_e = np.linspace(-180, 180, cols + 1); lat_e = np.linspace(90, -90, rows + 1)
    ax.set_global()
    ax.pcolormesh(n_lon_e, n_lat_e, np.ma.masked_invalid(n_stretched(n_grid)), cmap=n_cmap, vmin=0, vmax=1,
                  shading="flat", zorder=1, transform=ccrs.PlateCarree())
    if have_ne:
        ax.add_feature(cfeature.LAND, facecolor=LAND_COLOR, zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=5)
    if lanes is not None:
        draw_lanes(ax, lanes, zorder=3)
    cmap = ListedColormap(R_COLORS); norm = BoundaryNorm(np.arange(-0.5, 6.5, 1), cmap.N)
    ax.pcolormesh(lon_e, lat_e, np.ma.masked_invalid(field), cmap=cmap, norm=norm, shading="flat", zorder=4,
                  transform=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, xlocs=range(-180, 181, 60), ylocs=[-60, -30, 0, 30, 60],
                      linewidth=0.5, color="gray", alpha=0.5, linestyle="--", zorder=6)
    gl.top_labels = False
    gl.xlabel_style = dict(fp); gl.ylabel_style = dict(fp)
    ax.set_title(title, fontsize=FONT_SIZE, fontfamily=FONT)
    ax.text(PANEL_LETTER_X, 1.0, letter, transform=ax.transAxes + ax.titleOffsetTrans, ha="left", va="baseline",
            fontsize=FONT_SIZE + 1, fontweight="bold", fontfamily=FONT)
    return gl


def set_side_labels(ax, gl, fp, left, right):
    """Latitude labels of one panel on the requested sides only; the 60-degree boundary labels are re-added by hand on the kept side."""
    if left and right:
        return
    gl.left_labels, gl.right_labels, gl.geo_labels = left, right, False
    lon, ha, dx = (180, "left", 5) if right else (-180, "right", -5)
    for lat in (60, -60):
        ax.annotate(f"60°{'N' if lat > 0 else 'S'}", xy=(lon, lat), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                    xytext=(dx, 0), textcoords="offset points", ha=ha, va="center", zorder=7, **fp)


def _label_artists(gl):
    arts = list(getattr(gl, "label_artists", []) or [])
    for name in ("left_label_artists", "right_label_artists", "geo_label_artists", "ylabel_artists"):
        for t in getattr(gl, name, []) or []:
            if t not in arts:
                arts.append(t)
    return arts


def close_panels(fig, a, b):
    """Move panel b left until the closest pair of inner latitude labels (a right side, b left side) is GAP_PX apart."""
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    (ax_a, gl_a), (ax_b, gl_b) = a, b
    lat_texts = {"60°N", "30°N", "0°", "30°S", "60°S"}
    xa_mid = ax_a.get_position().x0 + ax_a.get_position().width / 2
    xb_mid = ax_b.get_position().x0 + ax_b.get_position().width / 2
    right_a, left_b = [], []
    for t in _label_artists(gl_a):
        if t.get_text() in lat_texts and t.get_visible():
            bb = t.get_window_extent(ren); x = fig.transFigure.inverted().transform((bb.x1, 0))[0]
            if fig.transFigure.inverted().transform((bb.x0, 0))[0] > xa_mid:
                right_a.append((t.get_text(), x))
    for t in _label_artists(gl_b):
        if t.get_text() in lat_texts and t.get_visible():
            bb = t.get_window_extent(ren); x = fig.transFigure.inverted().transform((bb.x0, 0))[0]
            if fig.transFigure.inverted().transform((bb.x1, 0))[0] < xb_mid:
                left_b.append((t.get_text(), x))
    if not right_a or not left_b:
        print("[layout] inner labels not found; panels left at the provisional positions"); return
    gaps = []
    for txt, xr in right_a:
        for txt2, xl in left_b:
            if txt == txt2:
                gaps.append((xl - xr, txt))
    gap_now, binding = min(gaps)
    shift = gap_now - GAP_PX / fig.get_figwidth() / fig.dpi
    pos = ax_b.get_position()
    ax_b.set_position([pos.x0 - shift, pos.y0, pos.width, pos.height])
    print(f"[layout] panels moved together; binding labels: {binding}, gap {GAP_PX} px")


def draw_pair(panels, year, rows, cols, n_layers, lanes, out_base, have_ne):
    """Figure of one year: panel a (OC-CCI) and panel b (Aqua/MODIS) side by side, one legend strip below."""
    n_cmap = LinearSegmentedColormap.from_list("n60", ["#ffffff", N_DARK])
    fp = {"family": FONT, "size": FONT_SIZE}
    fig = plt.figure(figsize=FIGSIZE)
    if INNER_LABELS == "shared":
        positions = [[0.012, PANEL_POS[0][1], 0.475, PANEL_POS[0][3]], [0.513, PANEL_POS[1][1], 0.475, PANEL_POS[1][3]]]
        sides = [(True, True), (False, True)]
    elif INNER_LABELS == "none":
        gap = GAP_PX / (FIGSIZE[0] * 300)
        positions = [[0.012, PANEL_POS[0][1], 0.488 - gap / 2, PANEL_POS[0][3]], [0.5 + gap / 2, PANEL_POS[1][1], 0.488 - gap / 2, PANEL_POS[1][3]]]
        sides = [(True, False), (False, True)]
    else:
        positions = PANEL_POS
        sides = [(True, True), (True, True)]
    axes = []
    for (letter, label, df), pos, (left, right) in zip(panels, positions, sides):
        ax = fig.add_axes(pos, projection=ccrs.Robinson())
        title = f"Annual Chlorophyll\u2013{NOX} Correlation (Pearson r), {year}, {label}"
        gl = draw_panel(ax, df, letter, title, rows, cols, n_layers, n_cmap, lanes, have_ne, fp)
        set_side_labels(ax, gl, fp, left, right)
        axes.append((ax, gl))
    if INNER_LABELS == "touch":
        close_panels(fig, axes[0], axes[1])
    draw_legend_strip(fig, n_cmap, n_layers[3], lanes is not None, fp)
    for ext in FORMATS:
        fig.savefig(f"{out_base}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def consistency(year, label, results_csv, df_all, df_ret):
    """Compare the counts of one product and year with its surrogate results CSV; report only."""
    if not os.path.exists(results_csv):
        print(f"[Consistency {year} {label}] {results_csv} not found, nothing to compare"); return
    res = pd.read_csv(results_csv); row = res[(res["year"].astype(str) == str(year)) & (res["scenario"] == "baseline")]
    if row.empty:
        print(f"[Consistency {year} {label}] no baseline row in {results_csv}"); return
    r = row.iloc[-1]; n_pos = int((df_ret["Slope"] >= 0).sum())
    mark = lambda name, got, ref: f"{name} {got:,} -> " + ("ok" if got == ref else f"differs from {ref:,} by {got - ref:+,}")
    print(f"[Consistency {year} {label}] " + "; ".join([mark("entering", len(df_all), int(r["n_regressed"])), mark("retained", len(df_ret), int(r["n_screened"])),
                                                       mark("positive", n_pos, int(r["n_pos"]))]))


def keyed(df):
    """Retained windows indexed by their center cell."""
    d = df.copy(); d["key"] = list(zip(d["Latitude"].round(2), d["Longitude"].round(2)))
    return d.drop_duplicates("key").set_index("key")


def overlap_row(year, A, B):
    """One row of Table S17 (A: Aqua/MODIS, B: OC-CCI) and its check line."""
    both = A.index.intersection(B.index); union = A.index.union(B.index)
    Ap, Bp = A.index[A["Slope"] > 0], B.index[B["Slope"] > 0]
    both_p = Ap.intersection(Bp); union_p = Ap.union(Bp)
    same = ((A["Slope"].reindex(both) > 0).to_numpy() == (B["Slope"].reindex(both) > 0).to_numpy()).mean() * 100 if len(both) else float("nan")
    row = [year, f"{len(A):,} ({len(Ap):,})", f"{len(B):,} ({len(Bp):,})", f"{len(both):,}", f"{len(both) / len(union):.3f}",
           f"{len(both_p):,}", f"{len(both_p) / len(union_p):.3f}", f"{same:.1f}"]
    check = (f"{year}: MODIS-only {len(A.index.difference(B.index)):,}, OC-CCI-only {len(B.index.difference(A.index)):,}, "
             f"both {len(both):,} of union {len(union):,}; sign conflicts among both: {int(round((100 - same) / 100 * len(both)))}")
    return row, check


def main():
    """Regress both products for every year, draw Figs. S32 to S36, write the retained windows and Table S17 with checks.txt."""
    needed = [EMISSION_PATH.format(year=y) for y in YEARS] + [tpl.format(year=y) for _, _, _, tpl, _ in PRODUCTS for y in YEARS]
    missing = [p for p in needed if not os.path.exists(p)]
    if missing:
        raise SystemExit("[ERROR] missing input file(s):\n    " + "\n    ".join(missing))
    os.makedirs(WIN_DIR, exist_ok=True)
    n_layers = load_n_grid(); lanes = load_lane_shapes(); have_ne = have_natural_earth()
    rows_s17, checks = [], []
    for year in YEARS:
        emission_path = EMISSION_PATH.format(year=year)
        with rasterio.open(emission_path) as src:
            rows, cols = src.shape
        panels, retained = [], {}
        for letter, key, label, chl_tpl, res_tpl in PRODUCTS:
            df = perform_moving_window_regression(chl_tpl.format(year=year), emission_path)
            ret = df[df["R2"] >= R2_SCREEN].reset_index(drop=True)
            n_pos = int((ret["Slope"] > 0).sum())
            print(f"[{year}] {label:11s} retained {len(ret):,} (positive {n_pos:,}, negative {len(ret) - n_pos:,})")
            consistency(year, label, res_tpl.format(year=year), df, ret)
            ret.drop(columns=["row", "col"]).to_csv(os.path.join(WIN_DIR, f"windows_{key}_{year}_retained.csv"), index=False)
            panels.append((letter, label, ret)); retained[key] = ret
        out_base = os.path.join(OUT_DIR, f"Fig_S{FIG_NUMBERS[year]}_chl_{year}_r")
        draw_pair(panels, year, rows, cols, n_layers, lanes, out_base, have_ne)
        print(f"  [Saved] Fig. S{FIG_NUMBERS[year]}: {out_base}.{FORMATS[0]}")
        row, check = overlap_row(year, keyed(retained["modis"]), keyed(retained["occci"]))
        rows_s17.append(row); checks.append(check)
    p = os.path.join(OUT_DIR, "Table_S17_product_overlap.csv")
    pd.DataFrame(rows_s17, columns=S17_HEADER).to_csv(p, index=False, encoding="utf-8-sig"); print(f"[Saved] {p}")
    with open(os.path.join(OUT_DIR, "checks.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(checks) + "\n")
    print("\n".join(checks))
    print(f"[Done] Figs. S32 to S36 and Table S17 in {OUT_DIR}/")


if __name__ == "__main__":
    main()
