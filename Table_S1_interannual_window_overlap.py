import csv
import os
import time
import numpy as np
import pandas as pd
import rasterio
from scipy.stats import linregress

R = 6_371_000
YEARS = ["2013", "2014", "2015", "2016", "2017"]
CHLOR_PATH = {y: os.path.join("data", "chlor", f"chlor_{y}.tif") for y in YEARS}
EMISSION_PATH = {y: os.path.join("data", "emission", f"NO_{y}_yearly.tif") for y in YEARS}
RESULTS_CSV = {y: os.path.join(f"surrogate_102_{y}", f"results_{y}.csv") for y in YEARS}

OUT_DIR = os.path.join("results", "Table_S1")
BLOCK1_PATH = os.path.join(OUT_DIR, "Table_S1_block1_yearly.csv")
BLOCK2_PATH = os.path.join(OUT_DIR, "Table_S1_block2_intersection.csv")
BLOCK3_PATH = os.path.join(OUT_DIR, "Table_S1_block3_adjacent_overlap.csv")
BLOCK4_PATH = os.path.join(OUT_DIR, "Table_S1_block4_turnover.csv")
RAW_PATH = os.path.join(OUT_DIR, "Table_S1_raw_values.csv")
WINDOWS_PATH = os.path.join(OUT_DIR, "Table_S1_windows_five_years.csv")
FINAL_PATH = os.path.join(OUT_DIR, "Table_S1_final.csv")

WINDOW_SIZE = 11
MIN_VALID = 70
R2_THRESH = 0.6
SLOPE_ABS_MIN = 0.01


ADJACENT_PAIRS = [("2013", "2014"), ("2014", "2015"), ("2015", "2016"), ("2016", "2017")]
KEY = ["Longitude", "Latitude"]


def pixel_area_m2(lat_deg, xres_deg, yres_deg):
    """Surface area of one grid cell at a latitude."""
    lat = np.deg2rad(lat_deg); dlat = np.deg2rad(yres_deg); dlon = np.deg2rad(xres_deg)
    return R**2 * dlon * (np.sin(lat + dlat/2) - np.sin(lat - dlat/2))

def regress_all_windows(chlor_raster_path, emission_raster_path, window_size=WINDOW_SIZE):
    """Every window regressed in log1p space from the cells with a valid value (finite and > 0) in both rasters, paired cell by cell; the slope floor is recorded as a flag instead of dropping the window."""
    with rasterio.open(chlor_raster_path) as src_c, rasterio.open(emission_raster_path) as src_e:
        assert src_c.transform == src_e.transform and src_c.shape == src_e.shape, "Resolution or CRS mismatch"
        chl, em, tf = src_c.read(1), src_e.read(1), src_c.transform
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
            results.append({"Longitude": lon, "Latitude": lat,
                            "Intercept": intercept, "Slope": slope,
                            "R": r_value, "R2": r_value**2,
                            "P_Value": p_value, "Count": n,
                            "below_slope_floor": abs(slope) < SLOPE_ABS_MIN})
    return pd.DataFrame(results)


def require(paths):
    """Stop with the list of missing input files."""
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit("[ERROR] missing input file(s):\n    " + "\n    ".join(missing))


def assert_one_grid(paths):
    """Stop unless every raster shares one transform and shape; return (xres, yres)."""
    ref_tf = ref_shape = None
    for p in paths:
        with rasterio.open(p) as src:
            if ref_tf is None:
                ref_tf, ref_shape, res = src.transform, src.shape, src.res
            else:
                assert src.transform == ref_tf, f"transform differs: {p}"
                assert src.shape == ref_shape, f"shape differs: {p}"
    return res


def area_1e6km2(lat_values, xres, yres):
    """Summed center-cell area in 10^6 km^2."""
    lat = np.asarray(lat_values, dtype="float64")
    return float(pixel_area_m2(lat, xres, yres).sum() / 1e12) if lat.size else 0.0


def consistency_year(year, entering, retained, retained_pos, retained_neg, pos_area_1e6):
    """Compare the yearly counts and the positive area with the surrogate results CSV; report only."""
    if not os.path.exists(RESULTS_CSV[year]):
        print(f"[Consistency {year}] entering {entering:,}; retained {retained:,}; positive {retained_pos:,} "
              f"(results CSV not found, nothing to compare)")
        return
    res = pd.read_csv(RESULTS_CSV[year])
    row = res[(res["year"].astype(str) == str(year)) & (res["scenario"] == "baseline")]
    if row.empty:
        print(f"[Consistency {year}] no baseline row in the results CSV")
        return
    r = row.iloc[-1]
    def mark(name, got, ref):
        return f"{name} {got:,} -> " + ("ok" if got == ref else f"differs from {ref:,} by {got - ref:+,}")
    area_ref = r["pos_area_km2"] / 1e6
    area_line = f"positive area {pos_area_1e6:.2f}e6 km2 -> " + (
        "ok" if abs(pos_area_1e6 - area_ref) < 0.005 else f"differs by {pos_area_1e6 - area_ref:+.3f}e6 km2")
    print(f"[Consistency {year}] " + "; ".join([
        mark("entering", entering, int(r["n_regressed"])),
        mark("retained", retained, int(r["n_screened"])),
        mark("positive", retained_pos, int(r["n_pos"])),
        mark("negative", retained_neg, int(r["n_neg"])), area_line]))


def fmt_frac(x):
    """Fractions to three decimals; smaller values in scientific notation."""
    return f"{x:.3f}" if x >= 5e-4 else f"{x:.3g}"


SUPERSCRIPT = str.maketrans("-0123456789", "\u207b\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079")

def thou(n):
    """Counts with thousands separators."""
    return f"{int(n):,}"


def sci2(x):
    """Two significant figures, typeset as m.m x 10^e with a superscript exponent."""
    if x <= 0:
        return "0"
    e = int(np.floor(np.log10(x)))
    mant = x / 10 ** e
    if round(mant, 1) >= 10:
        mant, e = mant / 10, e + 1
    return f"{mant:.1f} \u00d7 10{str(e).translate(SUPERSCRIPT)}"


def write_final_table(path, m, area_common, rows1, inter_rows, pair_rows, turn_rows):
    """Write the assembled table, parts A to C, ready to paste into the SI."""
    out = [["A. Yearly window counts, whole ocean and common domain", "", "", "", "", ""],
           ["Year", "Windows fitted", "Windows retained", "Retained in the common domain",
            "Retained positive in the common domain", "Retained negative in the common domain"]]
    for r in rows1:
        out.append([r["year"], thou(r["windows_fitted"]), thou(r["windows_retained"]),
                    thou(r["retained_in_common_domain"]), thou(r["retained_positive_in_common_domain"]),
                    thou(r["retained_negative_in_common_domain"])])
    out += [[""] * 6, ["B. Windows retained in all five years", "", "", "", "", "", "", "", ""],
            ["Subset", "Windows", "Fraction of the common domain", "Area (10\u2076 km\u00b2)",
             "Fraction of the common-domain area", "Expected fraction by count", "Expected fraction by area",
             "Observed to expected by count", "Observed to expected by area"]]
    for label, n, a, exp_c, exp_a in inter_rows:
        out.append([label[0].upper() + label[1:], thou(n), f"{n / m:.4f}", f"{a:.2f}", f"{a / area_common:.4f}",
                    sci2(exp_c), sci2(exp_a),
                    sci2((n / m) / exp_c) if exp_c > 0 else "", sci2((a / area_common) / exp_a) if exp_a > 0 else ""])
    out += [[""] * 6, ["C. Adjacent-year overlap of the retained sets", "", "", "", "", "", "", ""],
            ["Adjacent years", "Retained in both years", "Jaccard index, retained sets",
             "Same, for independent sets of the same sizes", "Observed to expected",
             "Jaccard index, retained positive sets", "Same, for independent sets of the same sizes", "Observed to expected"]]
    for pair, both, jr, jr_i, jp, jp_i in pair_rows:
        out.append([pair.replace("-", "\u2013"), thou(both), f"{jr:.3f}", f"{jr_i:.3f}", f"{jr / jr_i:.1f}",
                    f"{jp:.3f}", f"{jp_i:.3f}", f"{jp / jp_i:.1f}"])
    out += [[""] * 6, ["D. Windows retained in one year, followed into the next year", "", "", "", "", "", "", "", ""],
            ["Adjacent years", "Retained in the first year", "Retained again in the second year (fraction)",
             "Median R\u00b2 in the second year, all", "Not retained again", "Median (IQR) R\u00b2 in the second year, not retained again",
             "Same sign of slope (fraction)", "Reversed sign (fraction)", "Below slope floor (fraction)"]]
    for pair, first, again, med_all, leave, q, same, rev, floor in turn_rows:
        out.append([pair.replace("-", "\u2013") if pair != "All pairs" else pair, thou(first), f"{again / first:.3f}",
                    f"{med_all:.3f}", thou(leave), f"{q[1]:.3f} ({q[0]:.3f}\u2013{q[2]:.3f})",
                    f"{same / leave:.3f}", f"{rev / leave:.3f}", f"{floor / leave:.3f}"])
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        csv.writer(fh).writerows(out)


def main():
    """Regress the five years, check the counts, and write the three table blocks."""
    inputs = [CHLOR_PATH[y] for y in YEARS] + [EMISSION_PATH[y] for y in YEARS]
    require(inputs)
    os.makedirs(OUT_DIR, exist_ok=True)
    xres, yres = assert_one_grid(inputs)
    print(f"[Grid] one shared grid, {xres} x {yres} degrees per pixel")

    frames, keysets = {}, {}
    for y in YEARS:
        print(f"[Regress {y}] {CHLOR_PATH[y]} on {EMISSION_PATH[y]}")
        t0 = time.time()
        df = regress_all_windows(CHLOR_PATH[y], EMISSION_PATH[y])
        print(f"[Time {y}] {time.time() - t0:.1f} s for {len(df):,} windows")
        floor = df["below_slope_floor"].to_numpy(dtype=bool)
        r2 = df["R2"].to_numpy(dtype="float64")
        slope = df["Slope"].to_numpy(dtype="float64")
        retained = ~floor & (r2 >= R2_THRESH)
        pos, neg = retained & (slope > 0), retained & (slope < 0)
        pos_area = area_1e6km2(df.loc[pos, "Latitude"], xres, yres)
        consistency_year(y, int((~floor).sum()), int(retained.sum()),
                         int(pos.sum()), int(neg.sum()), pos_area)
        f = df[KEY].round(6).copy()
        f[f"Slope_{y}"], f[f"R2_{y}"] = slope, r2
        f[f"retained_{y}"], f[f"pos_{y}"], f[f"neg_{y}"] = retained.astype("int8"), pos.astype("int8"), neg.astype("int8")
        frames[y] = f
        keys = list(zip(f["Longitude"], f["Latitude"]))
        keysets[y] = {"all": {k for k, r in zip(keys, retained) if r},
                      "pos": {k for k, r in zip(keys, pos) if r},
                      "neg": {k for k, r in zip(keys, neg) if r}}
        print(f"[Record {y}] retained negative, global: {int(neg.sum()):,} "
              f"(= {int(retained.sum()):,} - {int(pos.sum()):,})")

    wide = frames[YEARS[0]]
    for y in YEARS[1:]:
        wide = wide.merge(frames[y], on=KEY, how="outer")
    for y in YEARS:
        for c in (f"retained_{y}", f"pos_{y}", f"neg_{y}"):
            wide[c] = wide[c].fillna(0).astype(bool)
        wide[f"fitted_{y}"] = wide[f"Slope_{y}"].notna()
    wide["in_common_domain"] = np.logical_and.reduce([wide[f"fitted_{y}"] for y in YEARS])
    wide.to_csv(WINDOWS_PATH, index=False, encoding="utf-8-sig")

    common = wide["in_common_domain"].to_numpy()
    m = int(common.sum())
    area_common = area_1e6km2(wide.loc[common, "Latitude"], xres, yres)
    lat_all = wide["Latitude"].to_numpy(dtype="float64")

    rows1, rate_cnt, rate_area = [], {}, {}
    for y in YEARS:
        ret_c = wide[f"retained_{y}"].to_numpy() & common
        pos_c = wide[f"pos_{y}"].to_numpy() & common
        neg_c = wide[f"neg_{y}"].to_numpy() & common
        rows1.append({"year": y,
                      "windows_fitted": len(frames[y]),
                      "windows_retained": int(wide[f"retained_{y}"].sum()),
                      "retained_in_common_domain": int(ret_c.sum()),
                      "retained_positive_in_common_domain": int(pos_c.sum()),
                      "retained_negative_in_common_domain": int(neg_c.sum())})
        rate_cnt[y] = {"all": ret_c.sum() / m, "pos": pos_c.sum() / m, "neg": neg_c.sum() / m}
        rate_area[y] = {"all": area_1e6km2(lat_all[ret_c], xres, yres) / area_common,
                        "pos": area_1e6km2(lat_all[pos_c], xres, yres) / area_common,
                        "neg": area_1e6km2(lat_all[neg_c], xres, yres) / area_common}
        print(f"[Factor {y}] count rates on the common domain: "
              f"all {rate_cnt[y]['all']:.6f}, pos {rate_cnt[y]['pos']:.6f}, neg {rate_cnt[y]['neg']:.6f}; "
              f"area rates: all {rate_area[y]['all']:.6f}, pos {rate_area[y]['pos']:.6f}, neg {rate_area[y]['neg']:.6f}")
    inter = {"all": np.logical_and.reduce([wide[f"retained_{y}"].to_numpy() for y in YEARS]),
             "pos": np.logical_and.reduce([wide[f"pos_{y}"].to_numpy() for y in YEARS]),
             "neg": np.logical_and.reduce([wide[f"neg_{y}"].to_numpy() for y in YEARS])}
    label = {"all": "retained in all five years", "pos": "retained with a positive slope in all five years",
             "neg": "retained with a negative slope in all five years"}
    rows2, inter_rows, raw = [], [], {}
    for k in ("all", "pos", "neg"):
        n = int(inter[k].sum())
        a = area_1e6km2(lat_all[inter[k]], xres, yres)
        exp_c = float(np.prod([rate_cnt[y][k] for y in YEARS]))
        exp_a = float(np.prod([rate_area[y][k] for y in YEARS]))
        inter_rows.append((label[k], n, a, exp_c, exp_a))
        rows2.append({"subset": label[k], "n_windows": n,
                      "fraction_of_common_domain": fmt_frac(n / m),
                      "area_1e6_km2": f"{a:.2f}",
                      "fraction_of_common_area": fmt_frac(a / area_common),
                      "expected_fraction_count": f"{exp_c:.3g}",
                      "ratio_count": f"{(n / m) / exp_c:.3g}" if exp_c > 0 else "",
                      "expected_fraction_area": f"{exp_a:.3g}",
                      "ratio_area": f"{(a / area_common) / exp_a:.3g}" if exp_a > 0 else ""})
        raw.update({f"intersection_{k}_n": n, f"intersection_{k}_area_1e6km2": a,
                    f"expected_fraction_count_{k}": exp_c, f"expected_fraction_area_{k}": exp_a})
    block2 = pd.DataFrame(rows2)
    block2.to_csv(BLOCK2_PATH, index=False, encoding="utf-8-sig")

    rows1.append({"year": "All five years", "windows_fitted": m, "windows_retained": int(inter["all"].sum()),
                  "retained_in_common_domain": int(inter["all"].sum()),
                  "retained_positive_in_common_domain": int(inter["pos"].sum()),
                  "retained_negative_in_common_domain": int(inter["neg"].sum())})
    block1 = pd.DataFrame(rows1)
    block1.to_csv(BLOCK1_PATH, index=False, encoding="utf-8-sig")
    mixed = int(inter["all"].sum() - inter["pos"].sum() - inter["neg"].sum())

    slope_w = {y: wide[f"Slope_{y}"].to_numpy(dtype="float64") for y in YEARS}
    r2_w = {y: wide[f"R2_{y}"].to_numpy(dtype="float64") for y in YEARS}

    def jaccard_independent(ka, kb):
        """Jaccard index of two sets of sizes ka and kb placed independently on the m windows."""
        i = ka * kb / m
        return i / (ka + kb - i)

    rows3, pair_rows = [], []
    for a, b in ADJACENT_PAIRS:
        ra, rb = wide[f"retained_{a}"].to_numpy() & common, wide[f"retained_{b}"].to_numpy() & common
        pa, pb = wide[f"pos_{a}"].to_numpy() & common, wide[f"pos_{b}"].to_numpy() & common
        both = int((ra & rb).sum())
        jr, jp = (ra & rb).sum() / (ra | rb).sum(), (pa & pb).sum() / (pa | pb).sum()
        jr_i, jp_i = jaccard_independent(ra.sum(), rb.sum()), jaccard_independent(pa.sum(), pb.sum())
        print(f"[Union {a}-{b}] retained union {int((ra | rb).sum()):,}; positive union {int((pa | pb).sum()):,}; "
              f"independent-set Jaccard {jr_i:.4f} / {jp_i:.4f}")
        pair_rows.append((f"{a}-{b}", both, jr, jr_i, jp, jp_i))
        rows3.append({"pair": f"{a}-{b}", "windows_retained_in_both": both,
                      "jaccard_retained": f"{jr:.3f}", "jaccard_retained_independent": f"{jr_i:.3f}",
                      "ratio_retained": f"{jr / jr_i:.1f}",
                      "jaccard_retained_positive": f"{jp:.3f}", "jaccard_retained_positive_independent": f"{jp_i:.3f}",
                      "ratio_retained_positive": f"{jp / jp_i:.1f}"})
    block3 = pd.DataFrame(rows3)
    block3.to_csv(BLOCK3_PATH, index=False, encoding="utf-8-sig")

    rows4, turn_rows, pool = [], [], {"first": 0, "again": 0, "leave": 0, "same": 0, "rev": 0, "floor": 0,
                                       "r2_all": [], "r2_leave": []}
    for a, b in ADJACENT_PAIRS:
        first = wide[f"retained_{a}"].to_numpy() & common
        again = first & wide[f"retained_{b}"].to_numpy()
        leave = first & ~wide[f"retained_{b}"].to_numpy()
        floor_b = leave & (np.abs(slope_w[b]) < SLOPE_ABS_MIN)
        same = leave & ~floor_b & (slope_w[a] * slope_w[b] > 0)
        rev = leave & ~floor_b & (slope_w[a] * slope_w[b] < 0)
        if int(same.sum() + rev.sum() + floor_b.sum()) != int(leave.sum()):
            raise SystemExit("[ERROR] leaving windows are not fully classified")
        r2_all, r2_leave = r2_w[b][first], r2_w[b][leave]
        q = np.percentile(r2_leave, [25, 50, 75])
        turn_rows.append((f"{a}-{b}", int(first.sum()), int(again.sum()), float(np.median(r2_all)),
                          int(leave.sum()), q, int(same.sum()), int(rev.sum()), int(floor_b.sum())))
        rows4.append({"pair": f"{a}-{b}", "retained_first_year": int(first.sum()), "retained_again": int(again.sum()),
                      "median_r2_second_year_all": f"{np.median(r2_all):.4f}", "leaving": int(leave.sum()),
                      "leaving_q25_r2_second_year": f"{q[0]:.4f}", "leaving_median_r2_second_year": f"{q[1]:.4f}",
                      "leaving_q75_r2_second_year": f"{q[2]:.4f}", "leaving_same_sign": int(same.sum()),
                      "leaving_reversed_sign": int(rev.sum()), "leaving_below_slope_floor": int(floor_b.sum())})
        for k, v in (("first", first.sum()), ("again", again.sum()), ("leave", leave.sum()),
                     ("same", same.sum()), ("rev", rev.sum()), ("floor", floor_b.sum())):
            pool[k] += int(v)
        pool["r2_all"].append(r2_all); pool["r2_leave"].append(r2_leave)
    r2_all_p, r2_leave_p = np.concatenate(pool["r2_all"]), np.concatenate(pool["r2_leave"])
    qp = np.percentile(r2_leave_p, [25, 50, 75])
    turn_rows.append(("All pairs", pool["first"], pool["again"], float(np.median(r2_all_p)), pool["leave"], qp,
                      pool["same"], pool["rev"], pool["floor"]))
    rows4.append({"pair": "all", "retained_first_year": pool["first"], "retained_again": pool["again"],
                  "median_r2_second_year_all": f"{np.median(r2_all_p):.4f}", "leaving": pool["leave"],
                  "leaving_q25_r2_second_year": f"{qp[0]:.4f}", "leaving_median_r2_second_year": f"{qp[1]:.4f}",
                  "leaving_q75_r2_second_year": f"{qp[2]:.4f}", "leaving_same_sign": pool["same"],
                  "leaving_reversed_sign": pool["rev"], "leaving_below_slope_floor": pool["floor"]})
    block4 = pd.DataFrame(rows4)
    block4.to_csv(BLOCK4_PATH, index=False, encoding="utf-8-sig")

    common_keys = set(zip(wide.loc[common, "Longitude"], wide.loc[common, "Latitude"]))
    for k in ("all", "pos", "neg"):
        alt = len(set.intersection(*[keysets[y][k] for y in YEARS]) & common_keys)
        hit = alt == int(inter[k].sum())
        print(f"[XCheck] intersection {k}: merged-table {int(inter[k].sum()):,} vs set-based {alt:,} -> {'ok' if hit else 'MISS'}")
        if not hit:
            raise SystemExit("[ERROR] the two intersection paths disagree")
    if mixed < 0 or int(inter["all"].sum()) < int(inter["pos"].sum()) + int(inter["neg"].sum()):
        raise SystemExit("[ERROR] intersection accounting is inconsistent")

    raw.update({"common_domain_windows": m, "common_domain_area_1e6km2": area_common,
                "mixed_sign_windows": mixed,
                "turnover_pooled_retained_first_year": pool["first"], "turnover_pooled_retained_again": pool["again"],
                "turnover_pooled_leaving": pool["leave"], "turnover_pooled_leaving_same_sign": pool["same"],
                "turnover_pooled_leaving_reversed_sign": pool["rev"], "turnover_pooled_leaving_below_floor": pool["floor"],
                "turnover_pooled_median_r2_second_year_all": float(np.median(r2_all_p)),
                "turnover_pooled_leaving_median_r2_second_year": float(qp[1])})
    for r in rows1:
        for c in list(r)[1:]:
            raw[f"{c}_{r['year'].replace(' ', '_').lower()}"] = r[c]
    pd.DataFrame(sorted(raw.items()), columns=["quantity", "value"]).to_csv(RAW_PATH, index=False, encoding="utf-8-sig")
    write_final_table(FINAL_PATH, m, area_common, rows1, inter_rows, pair_rows, turn_rows)

    for df, name in ((block1, "block 1"), (block2, "block 2"), (block3, "block 3"), (block4, "block 4")):
        print(f"\n----- {name} -----")
        print(df.to_string(index=False))
    print(f"\n[Caption] common domain: {m:,} windows, {area_common:.2f} x 1e6 km2")
    print(f"[Caption] retained in all five years with mixed signs: {mixed:,} "
          f"(= {int(inter['all'].sum()):,} - {int(inter['pos'].sum()):,} - {int(inter['neg'].sum()):,})")
    print(f"[Sentence] Of the {m:,} windows fitted in all five years, {int(inter['all'].sum()):,} "
          f"({100 * inter['all'].sum() / m:.1f}%) are retained in all five years and {int(inter['pos'].sum()):,} "
          f"({100 * inter['pos'].sum() / m:.1f}%) are retained with a positive slope in all five years, "
          f"against {100 * float(np.prod([rate_cnt[y]['all'] for y in YEARS])):.2g}% and "
          f"{100 * float(np.prod([rate_cnt[y]['pos'] for y in YEARS])):.2g}% expected if yearly retention were independent.")
    print(f"[Saved] {BLOCK1_PATH}\n[Saved] {BLOCK2_PATH}\n[Saved] {BLOCK3_PATH}\n[Saved] {BLOCK4_PATH}\n[Saved] {RAW_PATH}\n[Saved] {WINDOWS_PATH}\n[Saved] {FINAL_PATH}")
    print(f"[Done] Table S1 in {OUT_DIR}/")


if __name__ == "__main__":
    main()
