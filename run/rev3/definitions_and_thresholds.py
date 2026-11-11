"""Referee items M6 and Mock-M5: do 3.7 and 3.8 survive the analysis choices?

Two mock referees asked the same thing from different angles. The paper calls a
repression domain by collapsing contiguous pEF-hit tiles -- the fourth of four
definitions in circulation -- and then calls a CD43 hit at one fixed threshold.
Both are defensible; neither was shown to be inconsequential.

  M6 (Q2)        three domain definitions:
                   A  contiguous Hit_pEF runs              (the paper, 3,001)
                   B  contiguous runs hit at pEF AND PGK   (DelRosso fig. caption)
                   C  contiguous Hit_PGK runs              (DelRosso body, ~3,900)
  Mock-M5 #4     does the bifunctional association hold at other hit thresholds?

Structural note that falls out of doing it, and that the referee's circularity
worry is really about: under definition B every domain is a PGK hit by
construction, so the robustness axis of 3.7 is not estimable there at all. That
is reported as "not estimable", not as a null.

Everything here is CPU and runs in under a minute. The script refuses to print
a sensitivity table until it has reproduced the published numbers under
definition A from master.csv, so a silent change in the rebuild cannot be
mistaken for a change in the conclusion.

Run:  ~/miniconda3/envs/machine_deep_learning/bin/python definitions_and_thresholds.py
(that env has statsmodels and sklearn; `protein` has neither, it has matplotlib)
"""
import json
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2, fisher_exact
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = "/home/dmin/ai&stem_cell"
THR = 0.6296596656141653          # CD43 hit call: 2 SD over the control median
OUT = {}


def build_domains(m, hit_mask, label):
    """Collapse contiguous runs of hit tiles, per protein, into domains."""
    d = m.loc[hit_mask & m.Avg_pEF.notna() if label != "C"
              else hit_mask & m.Avg_PGK.notna()].copy()
    d = d.sort_values(["protein", "tile_num"])
    brk = (d.protein != d.protein.shift()) | (d.tile_num != d.tile_num.shift() + 1)
    d["dom"] = brk.cumsum()
    rows = []
    for dom, g in d.groupby("dom", sort=False):
        rows.append(dict(
            protein=g.protein.iloc[0], n_tiles=len(g),
            first_tile=int(g.tile_num.iloc[0]), last_tile=int(g.tile_num.iloc[-1]),
            aa_start=int(g.tile_start.min()), aa_end=int(g.tile_end.max()),
            max_pEF=float(np.nanmax(g.Avg_pEF)) if g.Avg_pEF.notna().any() else np.nan,
            pgk_hit=bool((g.Hit_PGK == "Hit").any()),
            pef_hit=bool((g.Hit_pEF == "Hit").any()),
            mincmv_hit=bool((g.Hit_minCMV == "Hit").any()),
            max_minCMV=float(np.nanmax(g.Avg_minCMV)) if g.Avg_minCMV.notna().any() else np.nan,
            cd43_max=float(np.nanmax(g.cd43)) if g.cd43.notna().any() else np.nan,
            cd43_first=float(g.cd43.iloc[0]),
        ))
    return pd.DataFrame(rows)


def fisher_block(d, exposure, outcome, alt):
    k1 = d[outcome][d[exposure] == 1]
    k0 = d[outcome][d[exposure] == 0]
    if len(k1) == 0 or len(k0) == 0:
        return None
    orv, p = fisher_exact([[int(k1.sum()), len(k1) - int(k1.sum())],
                           [int(k0.sum()), len(k0) - int(k0.sum())]],
                          alternative=alt)
    return dict(k1=int(k1.sum()), n1=len(k1), p1=100 * k1.mean(),
                k0=int(k0.sum()), n0=len(k0), p0=100 * k0.mean(),
                OR=float(orv), p=float(p))


def adjusted(d, outcome, exposure="robust"):
    """OR for the exposure after strength and length, plus the GEE by protein."""
    try:
        b = smf.logit(f"{outcome} ~ max_pEF + logn", d).fit(disp=0)
        f = smf.logit(f"{outcome} ~ {exposure} + max_pEF + logn", d).fit(disp=0)
        g = smf.gee(f"{outcome} ~ {exposure} + max_pEF + logn", "protein", d,
                    family=sm.families.Binomial()).fit()
        lr = 2 * (f.llf - b.llf)
        a2 = roc_auc_score(d[outcome], b.predict())
        a3 = roc_auc_score(d[outcome], f.predict())
        return dict(OR_adj=float(np.exp(f.params[exposure])),
                    lrt_p=float(chi2.sf(lr, 1)),
                    OR_gee=float(np.exp(g.params[exposure])),
                    gee_p=float(g.pvalues[exposure]),
                    auc_base=float(a2), auc_plus=float(a3), d_auc=float(a3 - a2))
    except Exception as e:                       # separation under a definition
        return dict(error=str(e)[:80])


# ------------------------------------------------------------------ gate
def reproduce(m):
    """Definition A rebuilt from master.csv must match the published table."""
    print("=" * 74)
    print("GATE: rebuild definition A from master.csv and match the paper")
    print("=" * 74)
    a = build_domains(m, m.Hit_pEF == "Hit", "A")
    pub = pd.read_csv(f"{ROOT}/run/tycko/domains_cd43.csv")
    ok = True

    def chk(lab, got, want, tol=0):
        nonlocal ok
        good = abs(got - want) <= tol
        ok &= good
        print(f"  [{'ok ' if good else 'FAIL'}] {lab}: {got} vs published {want}")

    chk("domain count", len(a), len(pub))
    chk("PGK-hit domains", int(a.pgk_hit.sum()), int(pub.pgk_hit.sum()))
    chk("minCMV-hit domains", int(a.mincmv_hit.sum()), int(pub.mincmv_hit.sum()))

    # The domain boundaries agree exactly; the per-tile CD43 join in the
    # published table does not. Four domains have no CD43 there but do in
    # master.csv, and 47 more had their max taken over fewer tiles -- in every
    # large case the published max is the LOWER one, which is the signature of
    # missing tiles rather than of a different statistic. The fixed-tile
    # outcome the paper headlines is untouched by this; the max outcome is not.
    j = pub.merge(a, on=["protein", "first_tile"], suffixes=("_pub", "_me"))
    chk("domains matched on (protein, first_tile)", len(j), len(pub))
    recovered = j[j.cd43_max_pub.isna() & j.cd43_max_me.notna()]
    print(f"  [note] {len(recovered)} domain(s) have CD43 in master.csv but not "
          f"in the published table: {', '.join(recovered.protein)}")
    both = j[j.cd43_max_pub.notna() & j.cd43_max_me.notna()]
    for c in ("cd43_first", "cd43_max"):
        dis = int(((both[f"{c}_pub"] > THR) != (both[f"{c}_me"] > THR)).sum())
        print(f"  [note] {c}: {dis} hit-call disagreement(s) over {len(both)} domains")
        if c == "cd43_first":
            chk("fixed-tile hit calls agree", dis, 0)

    # On exactly the published subset, the published 3.8 must come back.
    sub = a.merge(pub.loc[pub.cd43_max.notna(), ["protein", "first_tile"]],
                  on=["protein", "first_tile"])
    sub = sub[sub.cd43_max.notna()].copy()
    sub["hit_fix"] = (sub.cd43_first > THR).astype(int)
    sub["bifun"] = sub.mincmv_hit.astype(int)
    r = fisher_block(sub, "bifun", "hit_fix", "less")
    print(f"  3.8 on the published subset: bifunctional {r['k1']}/{r['n1']}  "
          f"others {r['k0']}/{r['n0']}  OR {r['OR']:.3f}")
    chk("3.8 bifunctional hits", r["k1"], 2)
    chk("3.8 bifunctional n", r["n1"], 77)
    chk("3.8 background hits", r["k0"], 895)
    chk("3.8 background n", r["n0"], 2911)
    if not ok:
        sys.exit("rebuild does not reproduce the paper; not running sensitivity")

    # Full rebuild, for reference: the 4 recovered domains added.
    d = a[a.cd43_max.notna()].copy()
    d["hit_fix"] = (d.cd43_first > THR).astype(int)
    d["bifun"] = d.mincmv_hit.astype(int)
    r = fisher_block(d, "bifun", "hit_fix", "less")
    print(f"  3.8 on the full rebuild   : bifunctional {r['k1']}/{r['n1']} = "
          f"{r['p1']:.2f}%  others {r['k0']}/{r['n0']} = {r['p0']:.1f}%  "
          f"OR {r['OR']:.3f}  p {r['p']:.2g}")
    OUT["rebuild"] = dict(published_subset_ok=True, recovered=len(recovered),
                          full=r)
    print("  -> reproduces the paper on its own subset; sensitivity uses the "
          "fuller rebuild\n")
    return a


# ---------------------------------------------------------- M6: definitions
def definitions(m):
    print("=" * 74)
    print("M6: the same two tests under all three domain definitions")
    print("=" * 74)
    defs = {
        "A  pEF hits (bài này)": (m.Hit_pEF == "Hit", "A"),
        "B  hit ở cả pEF và PGK": ((m.Hit_pEF == "Hit") & (m.Hit_PGK == "Hit"), "B"),
        "C  PGK hits": (m.Hit_PGK == "Hit", "C"),
    }
    res = {}
    for name, (mask, lab) in defs.items():
        d = build_domains(m, mask, lab)
        n_all = len(d)
        d = d[d.cd43_max.notna()].copy()
        d["hit_max"] = (d.cd43_max > THR).astype(int)
        d["hit_fix"] = (d.cd43_first > THR).astype(int)
        d["bifun"] = d.mincmv_hit.astype(int)
        d["logn"] = np.log(d.n_tiles)
        # Under B every domain is a PGK hit, so the robustness axis has no
        # contrast; under C domains come from PGK, so the mirror axis is pEF.
        if lab == "A":
            d["robust"] = d.pgk_hit.astype(int)
            axis = "pgk_hit"
        elif lab == "C":
            d["robust"] = d.pef_hit.astype(int)
            axis = "pef_hit (trục gương)"
        else:
            axis = "không ước lượng được (mọi domain đều hit PGK)"

        print(f"\n--- {name}: {n_all:,} domain, {len(d):,} có điểm CD43 ---")
        print(f"    trục bền = {axis}")
        if lab != "B":
            for out in ("hit_fix", "hit_max"):
                r = fisher_block(d, "robust", out, "greater")
                a = adjusted(d, out)
                print(f"    3.7 {out}: bền {r['k1']}/{r['n1']} = {r['p1']:.1f}% vs "
                      f"{r['k0']}/{r['n0']} = {r['p0']:.1f}%  OR {r['OR']:.3f} "
                      f"p {r['p']:.2g}")
                if "error" not in a:
                    print(f"         adj OR {a['OR_adj']:.3f} (LRT p {a['lrt_p']:.3g}), "
                          f"GEE OR {a['OR_gee']:.3f} (p {a['gee_p']:.3g}), "
                          f"ΔAUC {a['d_auc']:+.4f}")
                res[f"{lab}|3.7|{out}"] = dict(**r, **a)
        else:
            print(f"    3.7 không ước lượng được: {int(d.pgk_hit.sum())}/{len(d)} "
                  f"domain là PGK hit theo định nghĩa")
        for out in ("hit_fix", "hit_max"):
            r = fisher_block(d, "bifun", out, "less")
            if r is None:
                continue
            print(f"    3.8 {out}: lưỡng chức năng {r['k1']}/{r['n1']} = "
                  f"{r['p1']:.2f}% vs {r['k0']}/{r['n0']} = {r['p0']:.1f}%  "
                  f"OR {r['OR']:.3f}  p {r['p']:.2g}")
            res[f"{lab}|3.8|{out}"] = r
    OUT["definitions"] = res


# --------------------------------------------------- Mock-M5 #4: thresholds
def thresholds(a):
    print("\n" + "=" * 74)
    print("Mock-M5: does 3.8 hold at other hit thresholds?")
    print("=" * 74)
    d = a[a.cd43_max.notna()].copy()
    d["bifun"] = d.mincmv_hit.astype(int)

    print("\n  (i) varying the CD43 outcome threshold "
          f"(paper uses {THR:.4f}):")
    grid = [0.0, 0.2, 0.4, THR, 0.8, 1.0, 1.5, 2.0]
    rows = []
    for t in grid:
        d["hit"] = (d.cd43_first > t).astype(int)
        r = fisher_block(d, "bifun", "hit", "less")
        star = "  <- bài dùng" if t == THR else ""
        print(f"    ngưỡng {t:5.3f}: nền {r['p0']:5.1f}%  lưỡng chức năng "
              f"{r['k1']:3d}/{r['n1']} = {r['p1']:5.2f}%  OR {r['OR']:.3f}  "
              f"p {r['p']:.2g}{star}")
        rows.append(dict(thr=t, **r))
    OUT["cd43_threshold"] = rows

    print("\n  (ii) varying the minCMV exposure threshold "
          "(paper uses the published Hit_minCMV call):")
    d["hit"] = (d.cd43_first > THR).astype(int)
    rows = []
    r = fisher_block(d, "bifun", "hit", "less")
    print(f"    nhãn công bố : n = {r['n1']:3d}  lưỡng chức năng {r['p1']:5.2f}%  "
          f"OR {r['OR']:.3f}  p {r['p']:.2g}")
    rows.append(dict(rule="published", **r))
    for q in (0.90, 0.95, 0.975, 0.99):
        cut = np.nanquantile(d.max_minCMV, q)
        d["bi_q"] = (d.max_minCMV > cut).astype(int)
        r = fisher_block(d, "bi_q", "hit", "less")
        print(f"    Avg_minCMV > phân vị {q:.3f} ({cut:5.2f}): n = {r['n1']:3d}  "
              f"lưỡng chức năng {r['p1']:5.2f}%  OR {r['OR']:.3f}  p {r['p']:.2g}")
        rows.append(dict(rule=f"q{q}", cut=float(cut), **r))
    OUT["mincmv_threshold"] = rows


def pfam_intervals():
    """Absolute aa coordinates of every Pfam hit, so a domain called under any
    definition can inherit the group by overlap instead of by domain_id."""
    a = pd.read_csv(f"{ROOT}/run/pfam_domains_annotated.csv")
    a = a[a.pfam_group.notna() & a.ali_from.notna()].copy()
    a["abs_from"] = a.aa_start + a.ali_from - 1
    a["abs_to"] = a.aa_start + a.ali_to - 1
    return a[["protein", "abs_from", "abs_to", "pfam_group"]]


def assign_groups(dom, iv):
    """One group per domain: the Pfam hit with the largest aa overlap."""
    j = dom.reset_index(drop=True).reset_index().merge(iv, on="protein", how="left")
    ov = (np.minimum(j.aa_end, j.abs_to) - np.maximum(j.aa_start, j.abs_from) + 1)
    j["ov"] = ov.clip(lower=0).fillna(0)
    j = j[j.ov > 0].sort_values("ov", ascending=False).drop_duplicates("index")
    out = dom.reset_index(drop=True).copy()
    out["pfam_group"] = out.index.map(
        j.set_index("index").pfam_group.to_dict())
    return out


def section36(m):
    """M6 for 3.6: does the domain-group ordering depend on the definition?"""
    print("\n" + "=" * 74)
    print("M6 for 3.6: nhóm miền dưới ba định nghĩa")
    print("=" * 74)
    iv = pfam_intervals()
    defs = {"A": m.Hit_pEF == "Hit",
            "B": (m.Hit_pEF == "Hit") & (m.Hit_PGK == "Hit"),
            "C": m.Hit_PGK == "Hit"}
    tabs = {}
    for lab, mask in defs.items():
        d = build_domains(m, mask, lab)
        d = assign_groups(d, iv)
        d = d[d.cd43_max.notna()].copy()
        d["hit_fix"] = (d.cd43_first > THR).astype(int)
        t = (d[d.pfam_group.notna()].groupby("pfam_group")
             .agg(n=("hit_fix", "size"), cd43=("hit_fix", "mean")))
        tabs[lab] = t
        print(f"\n  định nghĩa {lab}: {int(t.n.sum())} domain có nhãn nhóm, "
              f"{len(t)} nhóm")
    common = sorted(set.intersection(*[set(t.index[t.n >= 5]) for t in tabs.values()]))
    print(f"\n  {len(common)} nhóm có n >= 5 ở cả ba định nghĩa:")
    print(f"    {'nhóm':<16}" + "".join(f"{l:>18}" for l in "ABC"))
    for gname in common:
        row = "".join(f"{tabs[l].cd43[gname]*100:>10.1f}% (n={int(tabs[l].n[gname]):>3})"
                      for l in "ABC")
        print(f"    {gname:<16}{row}")
    from scipy.stats import spearmanr
    print("\n  tương quan thứ hạng của tỷ lệ hit CD43 theo nhóm:")
    for x, y in (("A", "B"), ("A", "C"), ("B", "C")):
        r = spearmanr([tabs[x].cd43[g] for g in common],
                      [tabs[y].cd43[g] for g in common])[0]
        print(f"    {x} vs {y}: rho = {r:+.3f}")
    OUT["section36"] = {l: tabs[l].to_dict() for l in tabs}


if __name__ == "__main__":
    m = pd.read_csv(f"{ROOT}/run/rev2/master.csv")
    a = reproduce(m)
    definitions(m)
    section36(m)
    thresholds(a)
    json.dump(OUT, open(f"{ROOT}/run/rev3/definitions_and_thresholds.json", "w"),
              indent=1)
    print(f"\nwrote {ROOT}/run/rev3/definitions_and_thresholds.json")
    print("DONE_DEFINITIONS")
