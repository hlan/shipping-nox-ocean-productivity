import os
import io
import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

PRODUCT = ""
SDIR = lambda year: f"surrogate_102_{year}{PRODUCT}"
CHL_DIR = "data/chlor" if PRODUCT == "" else "data/chlor_oc_cci"
OUT_DIR = os.path.join("results" if PRODUCT == "" else "results_cci", "Table_S7_to_S12_Fig_S28_S29")
FIG_S28 = os.path.join(OUT_DIR, "Fig_S28_surrogate_null_distribution")
FIG_S29 = os.path.join(OUT_DIR, "Fig_S29_surrogate_decay_curve")

WINDOW_WIDTH_DEG = 5.5
FONT_FAMILY = "Times New Roman"
DIR_COLORS = {"zonal E": "#1f77b4", "zonal W": "#d62728", "meridional": "#2ca02c", "diagonal": "#7f7f7f"}
DIST_BINS = [0.0, 15.0, 25.0, 40.0, 61.0]
DIST_COLORS = ["#08306b", "#2171b5", "#6baed6", "#c6dbef"]
ROT_COLOR = "#f1c232"
LABEL_ABOVE_PCT = 90.0
PCT_BINS = np.arange(20, 105.1, 2.5)
plt.rcParams.update({"font.family": FONT_FAMILY, "mathtext.fontset": "stix", "font.size": 13, "axes.titlesize": 13,
                     "axes.labelsize": 13, "pdf.fonttype": 42, "ps.fonttype": 42})

YEARS = ["2017", "2016", "2015", "2014", "2013"]
YEAR = "2017"
WINDOW, MIN_CELLS = 11, 70
R2_SCREEN = 0.6
SLOPE_FLOOR = 0.01
XRES = YRES = 0.5
R_EARTH = 6_371_000
PAIR_COVERAGE = 0.75
RATIO_MAX = 1.05
N_LIMIT = 2.0
EMI_DIR = "data/emission"
EMISSION = os.path.join(EMI_DIR, f"NO_{YEAR}_yearly.tif")
BASIN_MASK = "data/basinmask_01.msk"
N_CSV = "data/Nmean_60m_data_updated.csv"
BINS = [(0, 15), (15, 25), (25, 40), (40, 61)]

FAMILIES = ["Zonal westward", "Zonal eastward", "Meridional northward", "Meridional southward", "Diagonal", "Rotation"]
BASIN_ORDER = ["North Pacific", "South Atlantic", "North Atlantic", "South Pacific", "Indian", "Southern Ocean", "Other", "Unmapped"]
BASIN_DEFINITION = {
    "North Pacific": "mask code 2, north of the equator", "South Pacific": "mask code 2, south of the equator",
    "North Atlantic": "mask code 1, north of the equator", "South Atlantic": "mask code 1, south of the equator",
    "Indian": "mask code 3", "Southern Ocean": "mask code 10",
    "Other": "all other mask codes", "Unmapped": "cells absent from the mask"}


def f0(x):  return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{int(round(x)):,}"
def f1(x):  return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.1f}"
def f2(x):  return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.2f}"
def f4(x):  return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.4f}"


def pixel_area_km2(lat_deg):
    lat = np.deg2rad(lat_deg); dlat = np.deg2rad(YRES); dlon = np.deg2rad(XRES)
    return R_EARTH**2 * dlon * (np.sin(lat + dlat / 2) - np.sin(lat - dlat / 2)) / 1e6


def family_of(dlon, dlat, mode):
    if mode == "rot180": return "Rotation"
    if dlat == 0: return "Zonal eastward" if dlon > 0 else "Zonal westward"
    if dlon == 0: return "Meridional northward" if dlat > 0 else "Meridional southward"
    return "Diagonal"


def shift_short(tag, dlon, dlat, mode):
    """'20° W, 10° S' style."""
    if mode == "rot180": return "rotation"
    parts = []
    if dlon: parts.append(f"{abs(dlon)}° {'E' if dlon > 0 else 'W'}")
    if dlat: parts.append(f"{abs(dlat)}° {'N' if dlat > 0 else 'S'}")
    return ", ".join(parts)


def shift_long(tag, dlon, dlat, mode):
    """'15° west and 10° north' style."""
    if mode == "rot180": return "rotation"
    parts = []
    if dlon: parts.append(f"{abs(dlon)}° {'east' if dlon > 0 else 'west'}")
    if dlat: parts.append(f"{abs(dlat)}° {'north' if dlat > 0 else 'south'}")
    return " and ".join(parts)


def require(paths):
    """Stop with the list of missing input files."""
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit("[ERROR] missing input file(s):\n    " + "\n    ".join(missing))


def write(out_dir, name, header, rows):
    p = os.path.join(out_dir, name)
    pd.DataFrame(rows, columns=header).to_csv(p, index=False)
    print(f"[Saved] {p}")


def transpose(header, rows, first="Quantity"):
    """Rows as quantities and columns as the first field of each row (years), for the tall layout of a wide table."""
    return [first] + [r[0] for r in rows], [[header[k]] + [r[k] for r in rows] for k in range(1, len(header))]


def load_mask():
    """The WOA basin mask as (mlat, mlon, code) on 1-degree centers."""
    with open(BASIN_MASK, encoding="utf-8", errors="replace") as fh:
        lines = [ln for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
    has_header = any(ch.isalpha() for ch in lines[0])
    m = pd.read_csv(io.StringIO("".join(lines)), header=0 if has_header else None)
    if not has_header:
        m.columns = ["Latitude", "Longitude"] + [f"Basin_{i}" for i in range(m.shape[1] - 2)]
    low = {c.lower().strip(): c for c in m.columns}
    lat_c = next((c for k, c in low.items() if k.startswith("lat")), m.columns[0])
    lon_c = next((c for k, c in low.items() if k.startswith("lon")), m.columns[1])
    code_c = next((c for k, c in low.items() if k.startswith("basin")), m.columns[2])
    d = pd.DataFrame({"mlat": m[lat_c].astype(float).round(2), "mlon": m[lon_c].astype(float).round(2),
                      "code": pd.to_numeric(m[code_c], errors="coerce")}).dropna()
    d["code"] = d["code"].astype(int)
    return d.drop_duplicates(["mlat", "mlon"])


def basin_name(code, lat):
    if code == 2: return "North Pacific" if lat >= 0 else "South Pacific"
    if code == 1: return "North Atlantic" if lat >= 0 else "South Atlantic"
    if code == 3: return "Indian"
    if code == 10: return "Southern Ocean"
    return "Other"


def assign_basin(lat, lon, mask):
    """Basin name of every (lat, lon) from the nearest 1-degree mask cell."""
    key = pd.DataFrame({"mlat": np.floor(lat) + 0.5, "mlon": np.floor(lon) + 0.5})
    key["mlon"] = np.where(key["mlon"] > 180, key["mlon"] - 360, key["mlon"])
    j = key.merge(mask, on=["mlat", "mlon"], how="left")
    out = np.array(["Unmapped"] * len(j), dtype=object)
    ok = j["code"].notna().to_numpy()
    out[ok] = [basin_name(int(c), la) for c, la in zip(j.loc[ok, "code"], np.asarray(lat)[ok])]
    return out


def read_mask(path):
    """Validity mask (finite and above zero) of a raster and its transform."""
    with rasterio.open(path) as src:
        a = src.read(1).astype("float64"); nod = src.nodata
        if nod is not None and np.isfinite(nod): a[a == nod] = np.nan
        return np.isfinite(a) & (a > 0), src.transform


def box_count(mask, k=WINDOW):
    """Number of True cells in the k x k window centred on each cell, windows clipped at the grid edges."""
    m = mask.astype("int32"); half = k // 2
    cs = np.cumsum(np.cumsum(np.pad(m, ((1, 0), (1, 0))), axis=0), axis=1)
    rows, cols = m.shape; i = np.arange(rows); j = np.arange(cols)
    r0 = np.clip(i - half, 0, rows); r1 = np.clip(i + half + 1, 0, rows); c0 = np.clip(j - half, 0, cols); c1 = np.clip(j + half + 1, 0, cols)
    return cs[np.ix_(r1, c1)] - cs[np.ix_(r0, c1)] - cs[np.ix_(r1, c0)] + cs[np.ix_(r0, c0)]


def shifted(E, dlon, dlat, mode, tf):
    """The same roll or rotation as the generator; a positive latitude shift moves the field north."""
    if mode == "rot180": return np.rot90(E, 2)
    dlat_px = int(round(dlat / YRES)); row_px = -dlat_px if tf.e < 0 else dlat_px
    return np.roll(E, (row_px, int(round(dlon / XRES))), axis=(0, 1))


def entering_windows(wdir, tf):
    """Real field's windows entering the screen with grid indices, screen flags and cell area."""
    w = pd.read_csv(os.path.join(wdir, "windows_baseline_all.csv"))
    w = w[w["Slope"].abs() >= SLOPE_FLOOR].reset_index(drop=True)
    col, row = (~tf) * (w["Longitude"].to_numpy(), w["Latitude"].to_numpy())
    w["i"] = np.floor(row).astype(int); w["j"] = np.floor(col).astype(int)
    w["screened"] = w["R2"] >= R2_SCREEN; w["pos"] = w["screened"] & (w["Slope"] >= 0)
    w["area"] = pixel_area_km2(w["Latitude"].to_numpy())
    return w


def paired_domains(year, wdir, scen, chl_dir):
    """Real field's entering windows and, for each surrogate, the flags of those windows that lie in its window set D_s."""
    C, tf = read_mask(os.path.join(chl_dir, f"chlor_{year}.tif")); E, tf2 = read_mask(os.path.join(EMI_DIR, f"NO_{year}_yearly.tif"))
    assert C.shape == E.shape and tf == tf2, "chlorophyll and emission rasters differ in grid"
    w = entering_windows(wdir, tf)
    inside = box_count(C & E)[w["i"], w["j"]] >= MIN_CELLS
    inD, n_fit = {}, {}
    for _, s in scen.iterrows():
        Ds = box_count(C & shifted(E, s["dlon"], s["dlat"], s["mode"], tf)) >= MIN_CELLS
        inD[s["scenario"]] = Ds[w["i"], w["j"]]; n_fit[s["scenario"]] = int(Ds.sum())
    return w, inD, n_fit, float(inside.mean())


def paired_global(w, inD, sur):
    """Real field's rates, positive area, negative count and negative area on each D_s, and the paired percentages."""
    sur = sur.copy(); rr, rp, ra, rn, rna = [], [], [], [], []
    neg = w["screened"] & (w["Slope"] < 0)
    for tag in sur["scenario"]:
        d = inD[tag]; n_ent = int(d.sum()); n_scr = int((d & w["screened"]).sum()); n_posd = int((d & w["pos"]).sum())
        rr.append(n_scr / n_ent if n_ent else np.nan); rp.append(n_posd / n_ent if n_ent else np.nan); ra.append(float(w.loc[d & w["pos"], "area"].sum()))
        rn.append(int((d & neg).sum())); rna.append(float(w.loc[d & neg, "area"].sum()))
    sur["ppr"] = sur["n_pos"] / sur["n_regressed"]
    sur["rate_real_p"] = rr; sur["prate_real_p"] = rp; sur["area_real_p"] = ra; sur["nneg_real_p"] = rn; sur["negarea_real_p"] = rna
    sur["rate_pct"] = 100 * sur["fpr"] / sur["rate_real_p"]; sur["area_pct"] = 100 * sur["pos_area_km2"] / sur["area_real_p"]
    sur["prate_pct"] = 100 * sur["ppr"] / sur["prate_real_p"]
    sur["neg_n_pct"] = 100 * sur["n_neg"] / sur["nneg_real_p"].replace(0, np.nan)
    sur["neg_area_pct"] = 100 * sur["neg_area_km2"] / sur["negarea_real_p"].replace(0, np.nan)
    return sur


def screen(df):
    m = df["R2"] >= R2_SCREEN
    if "Slope" in df.columns: m &= df["Slope"].abs() >= SLOPE_FLOOR
    return df[m]


def surrogate_neg_area(wdir, res):
    """Negative-correlation area (km2) of every field of one year from its screened-window file, checked against n_neg."""
    out = {}
    for tag, n_neg in zip(res["scenario"], res["n_neg"]):
        w = screen(pd.read_csv(os.path.join(wdir, f"windows_{tag}.csv")))
        m = w["Slope"].to_numpy() < 0
        assert int(m.sum()) == int(n_neg), f"{wdir}/windows_{tag}.csv: {int(m.sum())} negative windows, results CSV says {n_neg}"
        out[tag] = float(pixel_area_km2(w["Latitude"].to_numpy()[m]).sum())
    return out


def load_year(year, sdir, chl_dir, checks):
    """Results table, surrogates with the paired statistics, real-field windows and D_s flags of one year."""
    wdir = os.path.join(sdir(year), "windows")
    res = pd.read_csv(os.path.join(sdir(year), f"results_{year}.csv")).drop_duplicates("scenario", keep="last")
    res["family"] = [family_of(a, b, c) for a, b, c in zip(res["dlon"], res["dlat"], res["mode"])]
    res.loc[res["scenario"] == "baseline", "family"] = "Real field"
    neg_area = surrogate_neg_area(wdir, res)
    res["neg_area_km2"] = [neg_area[t] for t in res["scenario"]]
    sur = res[res["scenario"] != "baseline"].copy()
    w, inD, n_fit, inside = paired_domains(year, wdir, sur, chl_dir)
    checks.append(f"paired {year}: real-field entering windows inside the mask-based set {100 * inside:.2f}% (expected 100)")
    assert inside > 0.999, f"{year}: the real field's windows do not lie in the mask-based window set; the chlorophyll folder or the surrogate folder is the wrong product"
    ratio = np.array([n_fit[t] / res.set_index("scenario").loc[t, "n_regressed"] for t in sur["scenario"]])
    checks.append(f"paired {year}: fitted windows from the masks / n_regressed reported = {ratio.min():.3f} to {ratio.max():.3f} "
                  f"(slope floor removes 1 to 4%; values outside 1.00 to {RATIO_MAX} mean the shift convention differs from the generator)")
    assert 1.0 <= ratio.min() and ratio.max() <= RATIO_MAX, f"{year}: D_s does not match the generator's shift convention; see checks"
    sur = paired_global(w, inD, sur)
    return res, sur, w, inD


def yearly_stats(res, s):
    """Paired ranks of the real field on the positive-slope rate, the positive area and the negative windows of one year."""
    b = res[res["scenario"] == "baseline"].iloc[0]
    rk_r = int((s["prate_pct"] > 100).sum()) + 1; rk_a = int((s["area_pct"] > 100).sum()) + 1
    j90 = int(((s["prate_pct"] >= 90) & (s["area_pct"] >= 90)).sum())
    rk_nn = int((s["n_neg"] > s["nneg_real_p"]).sum()) + 1; rk_na = int((s["neg_area_km2"] > s["negarea_real_p"]).sum()) + 1
    return dict(b=b, s=s, rank_rate=rk_r, rank_area=rk_a, joint90=j90, rank_neg_n=rk_nn, rank_neg_area=rk_na,
                rank_rate_all=int((s["rate_pct"] > 100).sum()) + 1,
                higher_frac=int((s["frac_pos"] > b["frac_pos"]).sum()), n=len(s))


def basin_areas(w17, inD, wdir, mask, tags):
    """Positive and negative areas of every field by basin, with the real field's positive area on D_s and the coverage."""
    wbas = assign_basin(w17["Latitude"].to_numpy(), w17["Longitude"].to_numpy(), mask)
    wpos = w17["pos"].to_numpy(); warea = w17["area"].to_numpy()
    rows = []
    for tag in tags:
        w = screen(pd.read_csv(os.path.join(wdir, f"windows_{tag}.csv")))
        lat, lon, sl = w["Latitude"].to_numpy(), w["Longitude"].to_numpy(), w["Slope"].to_numpy()
        area = pixel_area_km2(lat); bas = assign_basin(lat, lon, mask)
        d = inD[tag] if tag != "baseline" else np.ones(len(w17), bool)
        for b in BASIN_ORDER:
            m = bas == b; mr = wbas == b
            rows.append({"scenario": tag, "basin": b, "pos_area": float(area[m & (sl >= 0)].sum()), "neg_area": float(area[m & (sl < 0)].sum()),
                         "n_pos": int((m & (sl >= 0)).sum()), "n_neg": int((m & (sl < 0)).sum()),
                         "real_p": float(warea[d & wpos & mr].sum()), "cov": float((d & mr).sum() / mr.sum()) if mr.sum() else np.nan})
    return pd.DataFrame(rows)


def region_paired(g):
    """Eligible surrogates of one region (coverage at or above PAIR_COVERAGE), their paired percentages and the rank of the real field."""
    e = g[g["cov"] >= PAIR_COVERAGE].copy()
    e["pct"] = np.where(e["real_p"] > 0, 100 * e["value"] / e["real_p"].replace(0, np.nan), np.inf)
    e["above"] = e["value"] > e["real_p"]
    return e, int(e["above"].sum()) + 1


def pct_shift(e, tag, meta):
    pct = e.loc[tag, "pct"]
    return f"{shift_short(tag, meta.loc[tag, 'dlon'], meta.loc[tag, 'dlat'], meta.loc[tag, 'mode'])} ({f1(pct) if np.isfinite(pct) else 'real field 0'})"


def closest_of(pct):
    """Index of the largest finite paired percentage; None if there is none."""
    fin = pct[np.isfinite(pct)]
    return fin.idxmax() if len(fin) else None


def above_list(e, meta, n_show=5):
    """Fields above the real field on common windows, highest first; at most n_show listed."""
    ab = e[e["above"]].sort_values("pct", ascending=False)
    if not len(ab): return "none"
    s = "; ".join(pct_shift(e, t, meta) for t in ab.index[:n_show])
    return s if len(ab) <= n_show else f"{s}; and {len(ab) - n_show} more"


def direction_rows(area_df, key_col, key_order, value_col, meta, label_of):
    """Strongest eligible surrogate of each direction family per region, paired (% of the real field on the same windows)."""
    out = []
    for k in key_order:
        g = area_df[area_df[key_col] == k].set_index("scenario").drop("baseline").rename(columns={value_col: "value"})
        e, _ = region_paired(g)
        r_ = [label_of(k)]
        for fam in FAMILIES:
            members = [t for t in e.index if meta.loc[t, "family"] == fam]
            n_all = int((meta.loc[g.index, "family"] == fam).sum())
            if not members: r_.append(f"excluded ({n_all} of {n_all})" if n_all else ""); continue
            best = closest_of(e.loc[members, "pct"])
            if best is None: r_.append(f"real field 0 ({len(members)} fields)"); continue
            pct = f1(e.loc[best, "pct"])
            excl = f", {n_all - len(members)} of {n_all} excluded" if n_all > len(members) else ""
            r_.append(f"{pct}{excl}" if fam == "Rotation" else f"{pct} ({shift_short(best, meta.loc[best, 'dlon'], meta.loc[best, 'dlat'], meta.loc[best, 'mode'])}{excl})")
        out.append(r_)
    return out


def direction_header(first):
    return [first] + [f"{f} (% of real field on common windows, shift)" if f != "Rotation" else "Rotation (% of real field on common windows)" for f in FAMILIES]


def nitrate_and_emission(mask, emission_path):
    """Nitrate cells with their basin, emission cells with their basin and the global top-decile threshold."""
    n = pd.read_csv(N_CSV); val = n["mean_N_Depth_60"].to_numpy(); okN = np.isfinite(val)
    nlat, nlon = n["Latitude"].to_numpy()[okN], n["Longitude"].to_numpy()[okN]; nval = val[okN]
    nbas = assign_basin(nlat, nlon, mask)
    with rasterio.open(emission_path) as src:
        arr = src.read(1).astype(float); tfm = src.transform
    nr, nc = arr.shape; jj, ii = np.meshgrid(np.arange(nc), np.arange(nr))
    elon = (tfm.c + (jj + 0.5) * tfm.a).ravel(); elat = (tfm.f + (ii + 0.5) * tfm.e).ravel(); ev = arr.ravel()
    good = np.isfinite(ev); p90 = np.percentile(ev[good & (ev > 0)], 90)
    return dict(nlat=nlat, nlon=nlon, nval=nval, nbas=nbas, elat=elat[good], elon=elon[good], evg=ev[good], ebas=assign_basin(elat[good], elon[good], mask), p90=p90)


def dist_labels(shifted):
    """Label of each displacement range from the distances it contains."""
    cats = pd.cut(shifted["shift_dist_deg"], DIST_BINS)
    out = []
    for iv in cats.cat.categories:
        d = shifted.loc[cats == iv, "shift_dist_deg"]
        out.append(f"{d.min():g}\u00b0" if d.min() == d.max() else f"{d.min():.1f}\u00b0 to {d.max():.1f}\u00b0".replace(".0\u00b0", "\u00b0"))
    return out


def classify_direction(row):
    if row["mode"] == "rot180":
        return "rot180"
    if row["dlat"] == 0:
        return "zonal E" if row["dlon"] > 0 else "zonal W"
    if row["dlon"] == 0:
        return "meridional"
    return "diagonal"


def shift_label(row):
    """Short name of a shift: 10° W, 10° W 10° N, rotation."""
    if row["mode"] == "rot180":
        return "rotation"
    parts = []
    if row["dlon"] != 0:
        parts.append(f"{abs(int(row['dlon']))}\u00b0 {'E' if row['dlon'] > 0 else 'W'}")
    if row["dlat"] != 0:
        parts.append(f"{abs(int(row['dlat']))}\u00b0 {'N' if row['dlat'] > 0 else 'S'}")
    return " ".join(parts)


def panel_title(ax, letter, text):
    """Bold panel letter followed by the title text."""
    ax.set_title(r"$\bf{" + letter + r"}$ " + text, loc="left")


def decay_panel(ax, sur, ycol, ylabel, short_name, letter, title):
    sur = sur.copy()
    sur["direction"] = sur.apply(classify_direction, axis=1)
    shifted = sur[sur["direction"] != "rot180"]
    rot = sur[sur["direction"] == "rot180"]

    for d, g in shifted.groupby("direction"):
        ax.scatter(g["shift_dist_deg"], g[ycol], s=28, alpha=0.75, label=d, color=DIR_COLORS[d])

    grp = shifted.groupby(pd.cut(shifted["shift_dist_deg"], DIST_BINS), observed=True)
    bm = grp.agg(d=("shift_dist_deg", "mean"), y=(ycol, "mean")).dropna()
    ax.plot(bm["d"], bm["y"], "-", color="k", lw=1.6, zorder=4, label="mean of the range (shifted fields)")

    if len(rot):
        ax.axhline(rot.iloc[0][ycol], color=ROT_COLOR, ls="--", lw=1.6, label="180\u00b0-rotated field")

    ax.scatter([0], [100.0], marker="*", s=320, color="red", zorder=5, label="real field")

    ax.axvline(WINDOW_WIDTH_DEG, color="k", ls=":", lw=1)
    ax.text(WINDOW_WIDTH_DEG + 0.9, ax.get_ylim()[1] * 0.97,
            "one window width (5.5\u00b0)", fontsize=12, va="top")

    rho = spearmanr(shifted["shift_dist_deg"], shifted[ycol]).statistic
    ax.text(0.02, 0.03,
            f"Spearman \u03c1(distance, {short_name}) = {rho:.2f}\n"
            f"({len(shifted)} shifted fields)",
            transform=ax.transAxes, fontsize=12, va="bottom", ha="left")

    ax.set_xlabel("Displacement distance (\u00b0)")
    ax.set_ylabel(ylabel)
    panel_title(ax, letter, title)
    ax.legend(frameon=True, fontsize=12)
    return rho


def null_panel(ax, sur, ycol, xlabel, letter, title):
    sur = sur.copy()
    shifted = sur[sur["mode"] != "rot180"]; rot = sur[sur["mode"] == "rot180"]
    groups = [shifted.loc[pd.cut(shifted["shift_dist_deg"], DIST_BINS) == iv, ycol].to_numpy()
              for iv in pd.cut(shifted["shift_dist_deg"], DIST_BINS).cat.categories]
    groups.append(rot[ycol].to_numpy())
    ax.hist(groups, bins=PCT_BINS, stacked=True, color=DIST_COLORS + [ROT_COLOR],
            edgecolor="white", label=dist_labels(shifted) + ["rotation"])
    ax.axvline(100.0, color="red", lw=2.5)
    rank = int((sur[ycol] > 100.0).sum()) + 1
    n = len(sur) + 1
    ax.text(100.0 - 0.012 * (ax.get_xlim()[1] - ax.get_xlim()[0]), ax.get_ylim()[1] * 0.95,
            f"real field\nrank {rank}/{n}\np = {rank/n:.4f}",
            color="red", fontsize=12, va="top", ha="right")
    close = sur[sur[ycol] >= LABEL_ABOVE_PCT].sort_values(ycol, ascending=False)
    lines = [f"fields above {LABEL_ABOVE_PCT:.0f}%:"] + [f"{shift_label(r)} ({r[ycol]:.1f})" for _, r in close.iterrows()]
    ax.text(100.0 - 0.012 * (ax.get_xlim()[1] - ax.get_xlim()[0]), ax.get_ylim()[1] * 0.70, "\n".join(lines),
            fontsize=11, va="top", ha="right", color="0.2")
    ax.set_xticks([20, 40, 60, 80, 90, 100])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of surrogate fields")
    panel_title(ax, letter, title)
    ax.legend(frameon=True, fontsize=11, title="displacement distance", title_fontsize=11, loc="upper left")
    return rank, n


def save(fig, base):
    fig.tight_layout()
    fig.savefig(base + ".png", dpi=300)
    fig.savefig(base + ".pdf")
    plt.close(fig)


def main():
    """Build the paired comparison for every year and write Tables S7 to S12, Figs. S28 and S29, the paired files and checks.txt."""
    require([EMISSION, BASIN_MASK, N_CSV]
            + [os.path.join(SDIR(y), f"results_{y}.csv") for y in YEARS]
            + [os.path.join(SDIR(y), "windows", "windows_baseline_all.csv") for y in YEARS]
            + [os.path.join(CHL_DIR, f"chlor_{y}.tif") for y in YEARS] + [os.path.join(EMI_DIR, f"NO_{y}_yearly.tif") for y in YEARS])
    for y in YEARS:
        res0 = pd.read_csv(os.path.join(SDIR(y), f"results_{y}.csv"))
        require([os.path.join(SDIR(y), "windows", f"windows_{t}.csv") for t in res0["scenario"].astype(str)])
    os.makedirs(OUT_DIR, exist_ok=True)
    checks = []

    yres, ysur = {}, {}
    for year in YEARS:
        res_y, sur_y, w_y, inD_y = load_year(year, SDIR, CHL_DIR, checks)
        sur_y.to_csv(os.path.join(OUT_DIR, f"paired_{year}_global.csv"), index=False)
        print(f"[Saved] {os.path.join(OUT_DIR, f'paired_{year}_global.csv')}")
        yres[year], ysur[year] = res_y, sur_y
        if year == YEAR:
            w17, inD = w_y, inD_y
    res, sur = yres[YEAR], ysur[YEAR]
    base = res[res["scenario"] == "baseline"].iloc[0]; meta = res.set_index("scenario")
    n_family = len(sur) + 1
    family_tags = ["baseline"] + list(sur["scenario"])
    checks.append(f"family: {len(sur)} surrogate fields plus the real field = {n_family} fields")
    sh = sur[sur["mode"] == "shift"]; rot = sur[sur["mode"] == "rot180"].iloc[0]
    Em, tfm = read_mask(EMISSION); lat_rows = np.array([(tfm * (0, i + 0.5))[1] for i in range(Em.shape[0])])
    cen = lambda M: float((M.sum(axis=1) * lat_rows).sum() / M.sum())
    checks.append(f"direction: emission centroid latitude {cen(Em):.2f} -> {cen(shifted(Em, 0, 15, 'shift', tfm)):.2f} under dlat = +15 "
                  f"(row 0 at latitude {lat_rows[0]:.2f}); it must move north, as positive dlat is labelled northward in Table S7")

    # Table S7
    rows = [["Real field", "1", "0", "0", "0"]]
    for fam, lonrange, latrange in [("Zonal, eastward", "+10° to +60° in steps of 5°", "0"),
                                    ("Zonal, westward", "−10° to −60° in steps of 5°", "0"),
                                    ("Meridional, northward", "0", "+10°, +15°, +20°"),
                                    ("Meridional, southward", "0", "−10°, −15°, −20°"),
                                    ("Diagonal", "±10°, ±15°, ±20°, ±30°, ±40°, ±50°", "±10°, ±15°, ±20°")]:
        g = sur[sur["family"] == fam.replace(", ", " ")]
        rows.append([fam, str(len(g)), lonrange, latrange, f"{g['shift_dist_deg'].min():.1f}° to {g['shift_dist_deg'].max():.1f}°".replace(".0°", "°")])
    rows.append(["Rotation by 180°", "1", "not defined", "not defined", "not defined"])
    rows.append(["All fields", str(n_family), "", "", ""])
    write(OUT_DIR, "Table_S7_family_composition.csv", ["Family", "Members", "Longitude shift", "Latitude shift", "Displacement distance"], rows)

    # Table S8 (rows are the quantities, columns the years)
    header = ["Year", "Windows entering screen, real field", "Retained windows, real field, total (positive slope)",
              "Retained windows, surrogates (minimum to maximum)", "Positive-slope window rate, real field", "Surrogate mean rate",
              "Highest surrogate rate", "Positive-correlation area, real field (10⁶ km²)", "Surrogate mean area (10⁶ km²)",
              "Highest surrogate area (10⁶ km²)", "Negative-correlation area, real field (10⁶ km²) (surrogate mean)",
              "Closest surrogate on common windows, rate (% of real field)",
              "Closest surrogate on common windows, area (% of real field)", "Surrogates reaching 90% of the real field on both statistics",
              "Surrogates with a higher positive-slope fraction", "Surrogates in family", "Rank of real field, both statistics",
              "Rank of real field on negative windows, count / area", "Empirical p"]
    ystat = {year: yearly_stats(yres[year], ysur[year]) for year in YEARS}; st17 = ystat[YEAR]
    rows = []
    for year in YEARS:
        st = ystat[year]; b, s, n = st["b"], st["s"], st["n"]
        rank = str(st["rank_rate"]) if st["rank_rate"] == st["rank_area"] else f"{st['rank_rate']} / {st['rank_area']}"
        rank_neg = f"{st['rank_neg_n']} / {st['rank_neg_area']}"
        rows.append([str(year), f0(b["n_regressed"]), f"{f0(b['n_screened'])} ({f0(b['n_pos'])})",
                     f"{f0(s['n_screened'].min())} to {f0(s['n_screened'].max())}", f4(b["n_pos"] / b["n_regressed"]), f4(s["ppr"].mean()), f4(s["ppr"].max()),
                     f2(b["pos_area_km2"] / 1e6), f2(s["pos_area_km2"].mean() / 1e6), f2(s["pos_area_km2"].max() / 1e6),
                     f"{f2(b['neg_area_km2'] / 1e6)} ({f2(s['neg_area_km2'].mean() / 1e6)})",
                     f1(s["prate_pct"].max()), f1(s["area_pct"].max()), str(st["joint90"]), str(st["higher_frac"]), str(n), rank, rank_neg,
                     f"{st['rank_rate'] / (n + 1):.4f}"])
    write(OUT_DIR, "Table_S8_global_results_by_year.csv", *transpose(header, rows))

    # Table S9
    rows = [["0, real field", "1", f"{f4(base['n_pos'] / base['n_regressed'])} (100)", f"{f2(base['pos_area_km2'] / 1e6)} (100)"]]
    for lo, hi in BINS:
        g = sh[(sh["shift_dist_deg"] > lo) & (sh["shift_dist_deg"] <= hi)]
        lab = (f"{g['shift_dist_deg'].min():.1f}°" if g["shift_dist_deg"].nunique() == 1
               else f"{g['shift_dist_deg'].min():.1f}° to {g['shift_dist_deg'].max():.1f}°").replace(".0°", "°")
        rows.append([lab, str(len(g)), f"{f4(g['ppr'].mean())} ({f1(g['prate_pct'].mean())})",
                     f"{f2(g['pos_area_km2'].mean() / 1e6)} ({f1(g['area_pct'].mean())})"])
    rows.append(["Rotation by 180°", "1", f"{f4(rot['ppr'])} ({f1(rot['prate_pct'])})",
                 f"{f2(rot['pos_area_km2'] / 1e6)} ({f1(rot['area_pct'])})"])
    rho_r = spearmanr(sh["shift_dist_deg"], sh["prate_pct"]).correlation; rho_a = spearmanr(sh["shift_dist_deg"], sh["area_pct"]).correlation
    rows.append([f"Spearman rank correlation between displacement distance and the statistic over the {len(sh)} shifted fields of the family",
                 str(len(sh)), f"{rho_r:.2f}".replace("-", "−"), f"{rho_a:.2f}".replace("-", "−")])
    write(OUT_DIR, "Table_S9_decay_with_distance.csv", ["Displacement", "Fields", "Mean positive-slope window rate (% of real field)",
                                                        "Mean positive-correlation area, 10⁶ km² (% of real field)"], rows)

    # basins
    mask = load_mask()
    basin_area = basin_areas(w17, inD, os.path.join(SDIR(YEAR), "windows"), mask, family_tags)
    basin_area.to_csv(os.path.join(OUT_DIR, "paired_2017_basins.csv"), index=False)
    gb = basin_area[basin_area["scenario"] == "baseline"].set_index("basin")
    checks.append(f"basins sum of baseline positive area {gb['pos_area'].sum():,.0f} km2 vs results CSV {base['pos_area_km2']:,.0f} km2 "
                  + ("OK" if abs(gb["pos_area"].sum() - base["pos_area_km2"]) < 1 else "MISMATCH"))
    ne = nitrate_and_emission(mask, EMISSION)

    # Table S10
    rows = []
    for b in BASIN_ORDER:
        r_ = gb.loc[b]
        mN = ne["nbas"] == b; mE = ne["ebas"] == b
        nl = 100 * (ne["nval"][mN] < N_LIMIT).sum() / mN.sum() if mN.sum() else np.nan
        td = 100 * (ne["evg"][mE] >= ne["p90"]).sum() / mE.sum() if mE.sum() and b != "Unmapped" else np.nan
        rows.append([b, BASIN_DEFINITION[b], f"{f0(r_['n_pos'] + r_['n_neg'])} ({f0(r_['n_pos'])} / {f0(r_['n_neg'])})",
                     f0(r_["pos_area"]), f0(r_["neg_area"]), f1(nl) if b != "Unmapped" else "", f1(td) if b != "Unmapped" else ""])
    tot = gb.sum()
    rows.append(["Global ocean", "sum of the 8 rows", f"{f0(tot['n_pos'] + tot['n_neg'])} ({f0(tot['n_pos'])} / {f0(tot['n_neg'])})",
                 f0(tot["pos_area"]), f0(tot["neg_area"]), "", ""])
    write(OUT_DIR, "Table_S10_basin_definitions_baseline.csv", ["Basin", "Definition", "Retained windows, total (positive slope / negative slope)",
                                                                "Positive-correlation area (km²)", "Negative-correlation area (km²)",
                                                                "Nitrogen-limited cells at 60 m (%)", "Cells in the global top emission decile (%)"], rows)

    # Table S11
    rows = []
    glob_base = gb["pos_area"].sum()
    for b in BASIN_ORDER:
        g = basin_area[basin_area["basin"] == b].set_index("scenario")
        bb = g.loc["baseline", "pos_area"]; s_ = g.drop("baseline").rename(columns={"pos_area": "value"})
        e, rank = region_paired(s_)
        best = closest_of(e["pct"]) if len(e) else None
        closest = shift_long(best, meta.loc[best, "dlon"], meta.loc[best, "dlat"], meta.loc[best, "mode"]) + f" ({f1(e.loc[best, 'pct'])})" if best is not None else ""
        rows.append([b, f0(bb), f1(100 * bb / glob_base), f0(s_["value"].mean()), str(len(e)), closest,
                     above_list(e, meta), f"{rank} of {len(e) + 1}", f"{rank / (len(e) + 1):.3f}"])
    bt = sur.loc[sur["area_pct"].idxmax(), "scenario"]
    rows.append(["Global ocean", f0(glob_base), "100", f0(sur["pos_area_km2"].mean()), str(len(sur)),
                 shift_long(bt, meta.loc[bt, "dlon"], meta.loc[bt, "dlat"], meta.loc[bt, "mode"]) + f" ({f1(sur['area_pct'].max())})",
                 "none" if st17["rank_area"] == 1 else "", f"{st17['rank_area']} of {n_family}", f"{st17['rank_area'] / n_family:.4f}"])
    write(OUT_DIR, "Table_S11_basin_test.csv", ["Basin", "Positive-correlation area, real field (km²)", "Fraction of global (%)", "Surrogate mean (km²)",
                                                f"Surrogates compared (coverage of the real field's windows ≥ {PAIR_COVERAGE:.0%})",
                                                "Closest surrogate on common windows (% of real field)", "Surrogates exceeding the real field on common windows (% of real field)",
                                                "Rank of real field", "Empirical p"], rows)

    # Table S12
    write(OUT_DIR, "Table_S12_basin_by_direction.csv", direction_header("Basin"),
          direction_rows(basin_area, "basin", BASIN_ORDER, "pos_area", meta, lambda k: k))

    # Figs. S28 and S29

    fig, axes = plt.subplots(2, 1, figsize=(9, 10))
    r1, n = null_panel(axes[0], sur, "prate_pct", "Positive-slope window rate (% of the real field on the common windows)", "a",
                       "Positive-slope window rate by displacement distance (2017)")
    r2, _ = null_panel(axes[1], sur, "area_pct", "Positive-correlation area (% of the real field on the common windows)", "b",
                       "Positive-correlation area by displacement distance (2017)")
    save(fig, FIG_S28)
    print(f"[Saved] {FIG_S28}.png / .pdf   (real field rank {r1}/{n} on rate, {r2}/{n} on area)")

    fig, axes = plt.subplots(2, 1, figsize=(9, 11))
    rho1 = decay_panel(axes[0], sur, "prate_pct", "Positive-slope window rate\n(% of the real field on the common windows)",
                       "rate", "a", "Positive-slope window rate against displacement distance (2017)")
    rho2 = decay_panel(axes[1], sur, "area_pct", "Positive-correlation area\n(% of the real field on the common windows)",
                       "area", "b", "Positive-correlation area against displacement distance (2017)")
    save(fig, FIG_S29)
    print(f"[Saved] {FIG_S29}.png / .pdf   (Spearman rho = {rho1:.3f} rate, {rho2:.3f} area)")

    # checks
    for year in YEARS:
        st, b_, s_ = ystat[year], ystat[year]["b"], ystat[year]["s"]
        checks.append(f"{year} rank of real field in the family of {n_family} (paired): positive-slope rate {st['rank_rate']}, area {st['rank_area']}; "
                      f"all-sign rate {st['rank_rate_all']}; "
                      f"unpaired: positive-slope rate {int((s_['ppr'] > b_['n_pos'] / b_['n_regressed']).sum()) + 1}, area {int((s_['pos_area_km2'] > b_['pos_area_km2']).sum()) + 1}; "
                      f"negative windows (paired): count {st['rank_neg_n']}, area {st['rank_neg_area']}; "
                      f"unpaired: count {int((s_['n_neg'] > b_['n_neg']).sum()) + 1}, area {int((s_['neg_area_km2'] > b_['neg_area_km2']).sum()) + 1}")
        top = s_.sort_values("prate_pct", ascending=False).head(3)
        checks.append(f"{year} closest on the positive-slope rate, paired: " + "; ".join(f"{r.scenario} rate {r.prate_pct:.1f}% area {r.area_pct:.1f}%" for r in top.itertuples()))
    cl = sur.sort_values("area_pct", ascending=False).head(3)
    checks.append("closest surrogates on the area, paired: " + "; ".join(f"{r.scenario} rate {r.prate_pct:.1f}% area {r.area_pct:.1f}%" for r in cl.itertuples())
                  + f"; mean surrogate area paired {sur['area_pct'].mean():.1f}%")
    with open(os.path.join(OUT_DIR, "checks.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(checks) + "\n")
    print("\n".join(checks))
    print(f"[Done] Tables S7 to S12 and Figs. S28, S29 in {OUT_DIR}/; family of {n_family} fields, positive-slope rate and area")


if __name__ == "__main__":
    main()
