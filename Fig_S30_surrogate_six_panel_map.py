import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.lines import Line2D
import geopandas as gpd

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

CHLOR_DIR = "data/chlor"
OUT_ROOT = "results/Fig_S30"
FIG_DIR = os.path.join(OUT_ROOT, "figure")
STEM = "Fig_S30_surrogate_six_panel_map"

YEAR = "2017"
WINDOWS_DIR = os.path.join(f"surrogate_102_{YEAR}", "windows")
SHOW_GRID = True
PANEL_TITLE_SIZE = 14
FIGSIZE = (11.5, 11)
WSPACE = 0.0

# Panels a to f: the window tag of the generator and the panel title. The five surrogates are chosen for what they show
# (Supplementary Note 1.2.5): the closest surrogate (10° W, along the North Pacific corridor axis), a shift across the corridors
# (15° S), the field that exceeds the real field in the North Atlantic (20° W 10° N), a far zonal shift (60° W) and the rotation.
PANELS = [
    ("baseline", "real field"),
    ("lon-10", "10° westward shift"),
    ("lat-15", "15° southward shift"),
    ("lon-20_lat+10", "shift of 20° west and 10° north"),
    ("lon-60", "60° westward shift"),
    ("rot180", "180° rotation"),
]

POS_COLOR = "#e66101"
NEG_COLOR = "#2166ac"

SHOW_LANES = True
LANE_SHP = "data/shipping_lane/mygeodata/Shipping_Lanes_v1.shp"
LANE_COLOR = "#7b3294"
LANE_TYPE_FIELD = "Type"
LANE_STYLE = {
    "core":      {"linewidth": 0.8,  "alpha": 0.9},
    "secondary": {"linewidth": 0.5,  "alpha": 0.6},
    "minor":     {"linewidth": 0.3,  "alpha": 0.4},
    "default":   {"linewidth": 0.3,  "alpha": 0.4},
}
LANE_CLASS_ALIASES = {"major": "core", "main": "core", "middle": "secondary",
                      "medium": "secondary", "small": "minor"}


def lane_class(value):
    """Map a shapefile class name to its style key."""
    v = str(value).strip().lower()
    return LANE_CLASS_ALIASES.get(v, v)

SHOW_BASINS = True
BASIN_MASK_PATH = "data/basinmask_01.msk"
BASIN_ID_NAMES = {1: "N Pacific", 2: "S Pacific", 3: "N Atlantic",
                  4: "S Atlantic", 5: "Indian", 6: "Southern Ocean"}
BASIN_LINE = {"edgecolor": "black", "linewidth": 1.0, "linestyle": (0, (5, 2.5))}
BASIN_LABEL_COLOR = "#1a1a1a"
BASIN_LABELS_AT = {
    "N Pacific": (-150, 42), "S Pacific": (-118, -45),
    "N Atlantic": (-30, 41), "S Atlantic": (-15, -42),
    "Indian": (80, -20), "Southern Ocean": (20, -62),
}


def _add_base_features(ax, coast_lw=1.0, borders=True):
    """Add land, ocean and coastline layers."""
    feats = [(cfeature.LAND, {'facecolor': 'lightgray'}),
             (cfeature.OCEAN, {'facecolor': 'white'}),
             (cfeature.COASTLINE, {'linewidth': coast_lw})]
    if borders:
        feats.append((cfeature.BORDERS, {'linestyle': ':'}))
    for feat, kw in feats:
        try:
            next(iter(feat.geometries()), None)
            ax.add_feature(feat, **kw)
        except Exception as e:
            name = getattr(feat, 'name', str(feat))
            print(f"[warn] base feature '{name}' skipped: {e}")


def _add_gridlines(ax, labels=True, label_size=7, lon_step=30):
    """Add the labelled graticule."""
    import matplotlib.ticker as mticker
    kw = dict(linewidth=0.4, color="gray", alpha=0.55, linestyle=":",
              xlocs=mticker.FixedLocator(range(-180, 181, lon_step)),
              ylocs=mticker.FixedLocator(range(-90, 91, 30)))
    try:
        gl = ax.gridlines(draw_labels=labels, **kw)
        if labels:
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {"size": label_size}
            gl.ylabel_style = {"size": label_size}
    except Exception as e:
        print(f"[warn] graticule labels unavailable ({e}); lines only")
        ax.gridlines(draw_labels=False, **kw)


def load_basin_ids():
    """Load the WOA23 basin-ID grid: 1 to 6 for the six basins of the tables, 0 for other ocean cells, -1 for cells absent from
    the mask (land)."""
    if not SHOW_BASINS:
        return None
    try:
        mask = pd.read_csv(BASIN_MASK_PATH, comment="#")
        mask = mask[["Latitude", "Longitude", "Basin_0m"]].dropna()
        code = mask["Basin_0m"].astype(int).to_numpy()
    except Exception as e:
        print(f"[warn] basin overlay skipped ({e}) - check BASIN_MASK_PATH")
        return None
    lat = mask["Latitude"].to_numpy()
    lon = mask["Longitude"].to_numpy()
    lat_c = np.sort(np.unique(lat)); lon_c = np.sort(np.unique(lon))
    dlat = float(np.min(np.diff(lat_c))); dlon = float(np.min(np.diff(lon_c)))
    top, left = lat_c.max(), lon_c.min()
    ids = np.full((len(lat_c), len(lon_c)), -1, dtype=int)
    ii = np.round((top - lat) / dlat).astype(int)
    jj = np.round((lon - left) / dlon).astype(int)
    out = np.zeros_like(code)
    out[(code == 2) & (lat >= 0)] = 1
    out[(code == 2) & (lat < 0)] = 2
    out[(code == 1) & (lat >= 0)] = 3
    out[(code == 1) & (lat < 0)] = 4
    out[code == 3] = 5
    out[code == 10] = 6
    ids[ii, jj] = out
    from rasterio.transform import from_origin
    tfm = from_origin(left - dlon / 2, top + dlat / 2, dlon, dlat)
    print(f"[basins] {len(mask)} mask cells on a {dlon:g}-deg grid")
    return ids, tfm


def basin_boundaries(ids, tfm):
    """The basin boundaries as merged polylines on the cell edges of the mask. An edge is a boundary where two ocean cells belong
    to different basins; cells absent from the mask (land) add no line, so the coasts are not traced. Each boundary is drawn
    once, which keeps the dashes even where basins meet."""
    from shapely.geometry import MultiLineString
    from shapely.ops import linemerge
    nrows_b, ncols_b = ids.shape
    x_edge = tfm.c + np.arange(ncols_b + 1) * tfm.a
    y_edge = tfm.f + np.arange(nrows_b + 1) * tfm.e
    ocean = ids >= 0
    segs = []
    d = (ids[:, :-1] != ids[:, 1:]) & ocean[:, :-1] & ocean[:, 1:]
    for i, j in zip(*np.nonzero(d)):
        segs.append(((x_edge[j + 1], y_edge[i]), (x_edge[j + 1], y_edge[i + 1])))
    d = (ids[:-1, :] != ids[1:, :]) & ocean[:-1, :] & ocean[1:, :]
    for i, j in zip(*np.nonzero(d)):
        segs.append(((x_edge[j], y_edge[i + 1]), (x_edge[j + 1], y_edge[i + 1])))
    merged = linemerge(MultiLineString(segs))
    return list(merged.geoms) if hasattr(merged, "geoms") else [merged]


def draw_basin_overlay(ax, ids, tfm, with_labels=False, label_size=6.5):
    """Outline and label the basins. The outlines are drawn above the routes so that they stay visible."""
    ax.add_geometries(basin_boundaries(ids, tfm), crs=ccrs.PlateCarree(), facecolor="none", zorder=6, **BASIN_LINE)
    for name in BASIN_ID_NAMES.values():
        if with_labels and name in BASIN_LABELS_AT:
            lo, la = BASIN_LABELS_AT[name]
            ax.text(lo, la, name.upper(), fontsize=label_size, color=BASIN_LABEL_COLOR,
                    fontweight="bold", ha="center", va="center", zorder=7,
                    transform=ccrs.PlateCarree(),
                    path_effects=[path_effects.withStroke(
                        linewidth=2.2, foreground="white")])


def load_lane_shapes():
    """Read the shipping-route shapefile."""
    if not SHOW_LANES:
        return None
    try:
        lanes = gpd.read_file(LANE_SHP)
        if lanes.crs is not None and lanes.crs.to_epsg() != 4326:
            lanes = lanes.to_crs(4326)
        print(f"[lanes] {len(lanes)} lane features from {LANE_SHP}")
        print(f"[lanes] columns: {list(lanes.columns)}")
        if LANE_TYPE_FIELD in lanes.columns:
            print(f"[lanes] class counts: "
                  f"{lanes[LANE_TYPE_FIELD].value_counts().to_dict()}")
        else:
            print(f"[lanes] field '{LANE_TYPE_FIELD}' not found - all lanes "
                  f"drawn with the default style")
        return lanes
    except Exception as e:
        print(f"[warn] lane overlay skipped ({e}) - check LANE_SHP")
        return None


def draw_lanes(ax, lane_gdf):
    """Draw the routes in their three classes."""
    if LANE_TYPE_FIELD in lane_gdf.columns:
        for cls, g in lane_gdf.groupby(LANE_TYPE_FIELD):
            st = LANE_STYLE.get(lane_class(cls), LANE_STYLE["default"])
            ax.add_geometries(g.geometry, crs=ccrs.PlateCarree(),
                              facecolor="none", edgecolor=LANE_COLOR,
                              zorder=5, **st)
    else:
        ax.add_geometries(lane_gdf.geometry, crs=ccrs.PlateCarree(),
                          facecolor="none", edgecolor=LANE_COLOR,
                          zorder=5, **LANE_STYLE["default"])


OVERLAY_LEGEND = [
    Line2D([0], [0], color=LANE_COLOR, lw=1.8, alpha=0.9, label='core shipping routes'),
    Line2D([0], [0], color=LANE_COLOR, lw=1.1, alpha=0.6, label='secondary shipping routes'),
    Line2D([0], [0], color=LANE_COLOR, lw=0.6, alpha=0.4, label='minor shipping routes'),
    Line2D([0], [0], color="black", ls=(0, (5, 2.5)), lw=1.4, label='basin boundaries'),
]


def visualize_six_panel_sign(windows_by_tag, fig_dir, stem, transform, shape,
                             basin_layer=None, lane_gdf=None, show_grid=True):
    """Draw the six-panel window-sign figure."""
    nrows_r, ncols_r = shape
    inv = ~transform
    xs = np.array([(transform * (j, 0))[0] for j in range(ncols_r + 1)])
    ys = np.array([(transform * (0, i))[1] for i in range(nrows_r + 1)])
    cmap = mcolors.ListedColormap([NEG_COLOR, POS_COLOR])
    norm = mcolors.BoundaryNorm([-1.5, 0.0, 1.5], cmap.N)

    fig = plt.figure(figsize=FIGSIZE)
    letters = "abcdef"
    for k, (tag, title) in enumerate(PANELS):
        df = windows_by_tag[tag]
        grid = np.full(shape, np.nan)
        if len(df):
            cr = [inv * (lo, la) for lo, la in zip(df["Longitude"], df["Latitude"])]
            jj = np.array([int(c) for c, r in cr])
            ii = np.array([int(r) for c, r in cr])
            ok = (ii >= 0) & (ii < nrows_r) & (jj >= 0) & (jj < ncols_r)
            sign = np.where(df["Slope"].to_numpy() >= 0, 1.0, -1.0)
            grid[ii[ok], jj[ok]] = sign[ok]

        ax = fig.add_subplot(3, 2, k + 1, projection=ccrs.Robinson())
        ax.set_global()
        _add_base_features(ax, coast_lw=0.4, borders=False)
        if show_grid:
            _add_gridlines(ax, labels=True, label_size=12, lon_step=60)
        ax.pcolormesh(xs, ys, grid, transform=ccrs.PlateCarree(),
                      cmap=cmap, norm=norm, shading="flat")
        if lane_gdf is not None:
            draw_lanes(ax, lane_gdf)
        if basin_layer is not None:
            draw_basin_overlay(ax, *basin_layer, with_labels=True, label_size=10)
        ax.set_title(f"$\\bf{{{letters[k]}}}$ {title}", fontsize=PANEL_TITLE_SIZE)

    handles = [
        Line2D([0], [0], marker='s', color=POS_COLOR, linestyle='None',
               markersize=11, label='positive slope (window passes the screening criteria)'),
        Line2D([0], [0], marker='s', color=NEG_COLOR, linestyle='None',
               markersize=11, label='negative slope (window passes the screening criteria)'),
    ]
    if lane_gdf is not None:
        handles += OVERLAY_LEGEND[0:3]
    if basin_layer is not None:
        handles.append(OVERLAY_LEGEND[3])
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=13)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.96,
                        bottom=0.11 if show_grid else 0.07,
                        hspace=0.30 if show_grid else 0.26, wspace=WSPACE)
    out_png, out_pdf = os.path.join(fig_dir, stem + ".png"), os.path.join(fig_dir, stem + ".pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_png}")
    print(f"[Saved] {out_pdf}")


def main():
    """Read the screened windows of the six fields and draw the figure."""
    os.makedirs(FIG_DIR, exist_ok=True)
    chlor_path = os.path.join(CHLOR_DIR, f"chlor_{YEAR}.tif")
    window_files = {tag: os.path.join(WINDOWS_DIR, f"windows_{tag}.csv") for tag, _ in PANELS}
    needed = [chlor_path, LANE_SHP, BASIN_MASK_PATH] + list(window_files.values())
    missing = [q for q in needed if not os.path.exists(q)]
    if missing:
        raise SystemExit("[ERROR] missing input file(s):\n    " + "\n    ".join(missing))
    with rasterio.open(chlor_path) as src:
        raster_transform = src.transform
        raster_shape = src.shape

    basin_layer = load_basin_ids()
    lane_gdf = load_lane_shapes()

    windows_by_tag = {}
    for tag, _ in PANELS:
        df = pd.read_csv(window_files[tag])
        print(f"[{tag}] {len(df):,} screened windows from {window_files[tag]}")
        windows_by_tag[tag] = df

    visualize_six_panel_sign(windows_by_tag, FIG_DIR, STEM, raster_transform, raster_shape,
                             basin_layer=basin_layer, lane_gdf=lane_gdf, show_grid=SHOW_GRID)


if __name__ == "__main__":
    main()
