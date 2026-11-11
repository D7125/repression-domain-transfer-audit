"""
Stage 1: unsupervised clustering of confirmed repression domains.

No neural network is trained here. HDBSCAN is a density-based clustering
algorithm; it runs on CPU in minutes. The only input it needs is one vector
per domain, which you already have as mean-pooled ESM embeddings.

What it does, in order:
  1. collapse overlapping hit tiles into contiguous domains
  2. build one vector per domain (mean of its tiles' embeddings)
  3. cluster
  4. measure internal sequence identity per cluster  <- separates Pfam families
     from convergent clusters, which is the whole point
  5. test motif enrichment per cluster, Fisher exact + Benjamini-Hochberg
  6. measure promoter robustness per cluster (pEF hit that is also a PGK hit)
  7. run a shuffled-label negative control

Usage
-----
  # with your existing mean-pooled ESM vectors (aligned row-for-row to data.csv)
  python cluster_domains.py --csv data.csv --emb esm2_640.npy --out clusters

  # no ESM yet? this runs today on sequence composition alone, so you can
  # check the pipeline and the enrichment statistics before paying for a GPU
  python cluster_domains.py --csv data.csv --emb none --out clusters

data.csv must come from prepare_data.py, and must carry Hit_pEF and Hit_PGK.
Re-run prepare_data.py if your copy predates those columns.

Outputs
-------
  <out>_domains.csv    one row per domain: cluster id, sequence, motifs, scores
  <out>_clusters.csv   one row per cluster: size, identity, robustness, strength
  <out>_enrichment.csv one row per (cluster, feature): odds ratio, p, FDR
"""

import argparse
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

# --------------------------------------------------------------------------
# Motifs. Each is a short linear sequence that recruits a named co-repressor.
# Sources: DelRosso et al. 2023 and the ELM database. Regex, not fuzzy match --
# a real motif scan would use a PWM, so treat these as a conservative floor.
# --------------------------------------------------------------------------
MOTIFS = {
    # Patterns as PRINTED IN THE PAPER, not reinvented here.
    "TLE_WRPW":     r"WRPW",            # Groucho/TLE. Was WRP[WY]: unjustified widening.
    "HP1_PxVxL":    r"P.V.L",           # unchanged -- this one was already right
    "CtBP":         r"P.[DENS][LM].",   # was P[LM][DNS][LS][ST]: wrong residue ORDER
    "CtBP_refined": r"PL[DN]L.",        # the paper's tightened variant
    "SUMO_fwd":     r"[VILMF]K.E",      # psi-K-x-E; psi set calibrated below
    "SIM":          r"[DE].{0,3}[VILMF].[VILMF][VILMF].{0,3}[DE]",  # SUMO-interacting
    # Not in the paper. Kept, but named so nobody mistakes it for a published
    # motif when it shows up in an enrichment table.
    "SUMO_rev_unpublished": r"E.K[VILMF]",
}

# What the paper reports on its own non-hit sequences. These are the acceptance
# tests: if our rate on OUR non-hit tiles is far off, the regex is wrong rather
# than the data.
PAPER_BACKGROUND = {
    "TLE_WRPW":   0.000,
    "HP1_PxVxL":  0.002,
    "CtBP":       0.002,
    "SUMO_fwd":   0.123,
}
# Published counts among repressing tiles, for the same purpose.
PAPER_TILE_COUNTS = {
    "coregulator-binding (TLE | HP1 | CtBP)": 36,
    "SIM": 113,
}
# Candidate hydrophobic sets for the SUMO psi position. The paper's 12.3%
# background picks the set; a set that fires on 30% of non-hits is too wide.
SUMO_PSI_SETS = ["VILMF", "VILMFP", "VILM", "VIL", "AVILMFP", "LIVMFYWC"]

# Composition features, tested the same way. The paper reports that repression
# domains are MORE structured than activation domains, so these are not padding.
def composition_features(seq):
    n = max(len(seq), 1)
    charge = (seq.count("K") + seq.count("R") - seq.count("D") - seq.count("E")) / n
    return {
        "frac_Pro":      seq.count("P") / n,
        "frac_aromatic": sum(seq.count(a) for a in "FWY") / n,
        "frac_acidic":   sum(seq.count(a) for a in "DE") / n,
        "frac_basic":    sum(seq.count(a) for a in "KR") / n,
        "net_charge":    charge,
        "frac_Cys":      seq.count("C") / n,   # zinc-finger proxy
    }


# --------------------------------------------------------------------------
def build_domains(df):
    """Collapse overlapping hit tiles into contiguous repression domains.

    Tiles step 10 aa, so consecutive tile_num overlap by 70 aa. Clustering raw
    tiles would just rediscover that overlap, which is why this runs first.
    """
    hits = df[(df.Hit_pEF == "Hit") & df.Avg_pEF.notna()].copy()
    hits["tile_num"] = pd.to_numeric(hits.tile_num)
    recs = []
    for gene, sub in hits.groupby("protein"):
        sub = sub.sort_values("tile_num")
        t = sub.tile_num.values
        cuts = np.where(np.diff(t) > 1)[0] + 1
        bounds = np.concatenate([[0], cuts, [len(sub)]])
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            grp = sub.iloc[lo:hi]
            seq = merge_tiles(grp)
            recs.append({
                "protein": gene,
                "domain_id": f"{gene}_{int(grp.tile_num.iloc[0])}",
                "n_tiles": len(grp),
                "first_tile": int(grp.tile_num.iloc[0]),
                "last_tile": int(grp.tile_num.iloc[-1]),
                # REAL residue coordinates. The old code put tile_num in the
                # columns named tile_start/tile_end, so they could not be
                # intersected with UniProt domain annotations.
                "aa_start": int(grp.tile_start.min()),
                "aa_end": int(grp.tile_end.max()),
                "aa_len": int(grp.tile_end.max() - grp.tile_start.min() + 1),
                "row_idx": grp.index.tolist(),
                # data.csv row of the strongest tile, for --domain-pool strongest
                "peak_row": int(grp.Avg_pEF.idxmax()),
                "sequence": seq,
                "max_pEF": float(grp.Avg_pEF.max()),
                "mean_pEF": float(grp.Avg_pEF.mean()),
                "mean_FLAG": float(grp.Avg_FLAG.mean(skipna=True))
                             if grp.Avg_FLAG.notna().any() else np.nan,
                "pgk_hit": bool((grp.Hit_PGK == "Hit").any()),
                "mincmv_hit": bool((grp.Hit_minCMV == "Hit").any()),
            })
    return pd.DataFrame(recs)


def merge_tiles(grp):
    """Stitch overlapping tiles back into the full domain sequence.

    The old version appended a fixed cur[-10:], assuming every tile steps by
    exactly 10 aa. It does not: near protein C-termini the final tile is
    shifted by 4, and dropping unscored tiles leaves steps of 20. A fixed -10
    then silently duplicates or drops residues. Stitch by real coordinates.
    """
    seqs = grp.sequence.tolist()
    starts = grp.tile_start.values
    out = seqs[0]
    for i in range(1, len(seqs)):
        shift = int(starts[i] - starts[i - 1])        # new residues this tile adds
        cur = seqs[i]
        out += cur[-shift:] if 0 < shift < len(cur) else cur
    return out


# --------------------------------------------------------------------------
def kmer_identity(a, b, k=5):
    """5-mer Jaccard similarity: a dependency-free proxy for sequence identity.

    NOT the same as alignment identity. It is here so the script runs anywhere.
    For anything you publish, replace this with MMseqs2:
        mmseqs easy-cluster domains.fasta clustered tmp --min-seq-id 0.3
    """
    A = {a[i:i+k] for i in range(len(a) - k + 1)}
    B = {b[i:i+k] for i in range(len(b) - k + 1)}
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def cluster_identity(seqs, rng, max_pairs=300):
    """Median pairwise similarity inside one cluster, on a random pair sample."""
    n = len(seqs)
    if n < 2:
        return np.nan
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(pairs) > max_pairs:
        pairs = [pairs[i] for i in rng.choice(len(pairs), max_pairs, replace=False)]
    return float(np.median([kmer_identity(seqs[i], seqs[j]) for i, j in pairs]))


# --------------------------------------------------------------------------
def bh_fdr(p):
    """Benjamini-Hochberg. Without this, 300 tests at p<0.05 give ~15 false hits."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def wilson(k, n, z=1.96):
    """95% CI for a proportion. A robustness fraction of 0.00 on 10 domains is
    compatible with the 27% background; on 60 domains it is not. Without the
    interval the cluster table cannot be read at all."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def motif_background(df):
    """Motif rates in repressing vs non-hit tiles, at TILE level.

    Two separate fixes from the patch doc live here:
      * level: domain-level rates are not comparable with the paper's
        tile-level background, because a long multi-tile domain matches more
        often for purely mechanical reasons. So this measures tiles.
      * own background: the paper's numbers are for cross-checking, not for
        substituting. Fisher 2x2 per motif on OUR non-hit set, BH corrected.

    "non-hit" = every scored tile not called Hit (96,682), which is the
    definition that reproduces the documented 1:14.7 class imbalance.
    """
    hit = df[df.Hit_pEF == "Hit"]
    non = df[df.Hit_pEF != "Hit"]
    print(f"\n  tile-level sets: {len(hit):,} repressing tiles vs "
          f"{len(non):,} non-hit tiles  (1:{len(non)/max(len(hit),1):.1f})")
    strict = df[df.Hit_pEF == "Not Hit"]
    print(f"  (a stricter 'Not Hit'-only background would be {len(strict):,} "
          f"tiles; reported alongside because the choice moves the rates)")

    rows = []
    for name, pat in MOTIFS.items():
        h = hit.sequence.str.contains(pat, regex=True)
        nh = non.sequence.str.contains(pat, regex=True)
        st = strict.sequence.str.contains(pat, regex=True)
        a, b = int(h.sum()), int((~h).sum())
        c, d = int(nh.sum()), int((~nh).sum())
        odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")
        rows.append({
            "motif": name, "pattern": pat,
            "n_hit_tiles": a, "frac_hit": a / max(a + b, 1),
            "n_nonhit_tiles": c, "frac_nonhit": c / max(c + d, 1),
            "frac_nonhit_strict": float(st.mean()) if len(strict) else np.nan,
            "paper_nonhit": PAPER_BACKGROUND.get(name, np.nan),
            "odds_ratio": odds, "p": p,
        })
    out = pd.DataFrame(rows)
    out["fdr"] = bh_fdr(out.p.values)
    out["enrichment"] = out.frac_hit / out.frac_nonhit.replace(0, np.nan)
    return out


def calibrate_sumo(df):
    """Pick the hydrophobic set for the SUMO psi position using the paper's
    12.3% non-hit background as the target. Reported, not silently applied."""
    non = df[df.Hit_pEF != "Hit"].sequence
    print("\n  SUMO psi-set calibration (target: 12.3% of non-hit tiles)")
    best, best_gap = None, np.inf
    for s in SUMO_PSI_SETS:
        rate = float(non.str.contains(f"[{s}]K.E", regex=True).mean())
        gap = abs(rate - 0.123)
        mark = ""
        if gap < best_gap:
            best, best_gap, mark = s, gap, ""
        print(f"    [{s:<8}]K.E -> {rate*100:5.2f}% of non-hit tiles"
              f"   (off by {gap*100:+.2f} pts)")
    print(f"    closest to the published background: [{best}]K.E")
    return best


def calibrate_sim(df):
    """The paper reports 113 repressing tiles containing a SIM. Our regex is a
    guess, so this prints the count it actually produces -- the number to tune
    against before the SIM column is used for anything."""
    hit = df[df.Hit_pEF == "Hit"].sequence
    n = int(hit.str.contains(MOTIFS["SIM"], regex=True).sum())
    print(f"\n  SIM regex check: {n:,} of {len(hit):,} repressing tiles match")
    print(f"    paper reports {PAPER_TILE_COUNTS['SIM']}. "
          f"ratio = {n / PAPER_TILE_COUNTS['SIM']:.1f}x")
    if n > 3 * PAPER_TILE_COUNTS["SIM"]:
        print("    -> far too permissive. Treat SIM as UNCALIBRATED; do not")
        print("       report SIM enrichment as a finding from this run.")
    coreg = hit.str.contains(MOTIFS['TLE_WRPW'], regex=True) | \
        hit.str.contains(MOTIFS['HP1_PxVxL'], regex=True) | \
        hit.str.contains(MOTIFS['CtBP'], regex=True)
    print(f"  coregulator-binding (TLE|HP1|CtBP): {int(coreg.sum()):,} tiles; "
          f"paper reports {PAPER_TILE_COUNTS['coregulator-binding (TLE | HP1 | CtBP)']}")
    print("    (the paper's 36 counts only tiles it mutagenised, so ours is")
    print("     expected to be larger; the check is order of magnitude)")
    return n


def enrichment_table(dom, feat_cols, label_col="cluster"):
    """Fisher exact per (cluster, binary feature), one cluster vs all others."""
    rows = []
    for cl in sorted(c for c in dom[label_col].unique() if c >= 0):
        inside = dom[label_col] == cl
        for f in feat_cols:
            a = int((inside & dom[f]).sum())
            b = int((inside & ~dom[f]).sum())
            c = int((~inside & dom[f]).sum())
            d = int((~inside & ~dom[f]).sum())
            if a + b == 0 or a + c == 0:
                continue
            odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")
            # A raw OR of inf just means b=0, which happens constantly on
            # clusters of 15-40 domains and tells you nothing about effect
            # size. Haldane-Anscombe keeps it finite and the CI shows how
            # little the point estimate is worth.
            A, B, C, D = a + .5, b + .5, c + .5, d + .5
            or_h = (A * D) / (B * C)
            se = np.sqrt(1 / A + 1 / B + 1 / C + 1 / D)
            rows.append({"cluster": cl, "feature": f, "n_in_cluster": a,
                         "cluster_size": a + b, "frac_in": a / (a + b),
                         "frac_out": c / max(c + d, 1),
                         "a": a, "b": b, "c": c, "d": d,
                         "odds_ratio": odds,
                         "or_haldane": or_h,
                         "or_ci_lo": float(np.exp(np.log(or_h) - 1.96 * se)),
                         "or_ci_hi": float(np.exp(np.log(or_h) + 1.96 * se)),
                         "p": p})
    out = pd.DataFrame(rows)
    if len(out):
        out["fdr"] = bh_fdr(out.p.values)
        out = out.sort_values("fdr")
    return out


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="data.csv from prepare_data.py")
    ap.add_argument("--emb", required=True,
                    help="mean-pooled ESM .npy aligned to data.csv rows, "
                         "or 'none' to use sequence composition instead")
    ap.add_argument("--out", default="clusters")
    ap.add_argument("--min-cluster-size", type=int, default=15)
    ap.add_argument("--min-tiles", type=int, default=1,
                    help="drop domains supported by fewer than this many tiles. "
                         "53%% of domains rest on a single 80-aa window; use 3 "
                         "for the well-supported core")
    ap.add_argument("--method", default="leaf", choices=["leaf", "eom", "kmeans"],
                    help="leaf = HDBSCAN fine-grained (default); eom = HDBSCAN "
                         "default, often one giant blob; kmeans = forces every "
                         "domain into one of --k clusters")
    ap.add_argument("--k", type=int, default=20, help="clusters, kmeans only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--domain-pool", default="mean",
                    choices=["mean", "max", "strongest"],
                    help="pooling level B, tiles -> domain. 'strongest' takes "
                         "the single highest-Avg_pEF tile, which is the rule "
                         "the paper uses for scoring a repression domain.")
    ap.add_argument("--mmseqs", default=None,
                    help="MMseqs2 easy-cluster TSV for DOMAINS (rep<TAB>member). "
                         "Replaces the 5-mer Jaccard proxy, which returns a "
                         "background of 0.0000 here and so cannot tell a "
                         "homology family from a convergent cluster.")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    df = pd.read_csv(args.csv)
    need = {"protein", "sequence", "Avg_pEF", "Hit_pEF", "Hit_PGK", "tile_num"}
    missing = need - set(df.columns)
    if missing:
        sys.exit(f"data.csv is missing {sorted(missing)} -- re-run prepare_data.py")

    # ---------------- tile-level motif validation (runs first) -------------
    # This has to precede clustering: if the regexes are wrong, every
    # downstream enrichment table is wrong too.
    print("=" * 72)
    print("MOTIF VALIDATION AT TILE LEVEL  (own background, paper as check)")
    print("=" * 72)
    calibrate_sumo(df)
    calibrate_sim(df)
    mb = motif_background(df)
    print("\n  motif          hit%   nonhit%  (strict%)  paper%   enrich   FDR")
    for _, r in mb.iterrows():
        paper = "   -  " if r.paper_nonhit != r.paper_nonhit else f"{r.paper_nonhit*100:5.1f}%"
        enr_s = "  inf " if r.enrichment != r.enrichment else f"{r.enrichment:5.1f}x"
        print(f"  {r.motif:<22} {r.frac_hit*100:5.2f}  {r.frac_nonhit*100:6.2f}   "
              f"({r.frac_nonhit_strict*100:5.2f})  {paper}  {enr_s}  {r.fdr:.1e}")
    mb.to_csv(f"{args.out}_motif_background.csv", index=False)

    print("\n  Read the 'enrich' column, not the 'hit%' column: a motif at 0.2%")
    print("  of domains against a 0% background is absolutely enriched, while")
    print("  one at 37.9% against a 12.3% background is only ~3x.")

    print("\ncollapsing overlapping hit tiles into domains ...")
    dom = build_domains(df)
    print(f"  {len(dom):,} domains from {dom.n_tiles.sum():,} hit tiles "
          f"across {dom.protein.nunique():,} proteins")
    print(f"  single-tile domains: {(dom.n_tiles == 1).sum():,} "
          f"({(dom.n_tiles == 1).mean()*100:.0f}%)")

    dom = dom[dom.n_tiles >= args.min_tiles].reset_index(drop=True)
    print(f"  kept {len(dom):,} domains with >= {args.min_tiles} tile(s)")

    # ---------------- one vector per domain --------------------------------
    if args.emb.lower() == "none":
        print("\nno ESM given: using 5-mer composition vectors (pipeline test mode)")
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=5)
        X = vec.fit_transform(dom.sequence)
        X = TruncatedSVD(n_components=64, random_state=args.seed).fit_transform(X)
    else:
        emb = np.load(args.emb)
        if len(emb) != len(df):
            sys.exit(f"{args.emb} has {len(emb):,} rows but data.csv has {len(df):,}. "
                     "They must be row-aligned.")
        # Pooling level B: several tiles -> one domain. The original code hard-
        # coded mean(0). The paper's own convention is the STRONGEST tile in
        # each domain, so that is selectable and the two are compared by AMI.
        if args.domain_pool == "mean":
            X = np.vstack([emb[idx].mean(0) for idx in dom.row_idx])
        elif args.domain_pool == "max":
            X = np.vstack([emb[idx].max(0) for idx in dom.row_idx])
        else:                                    # "strongest": the paper's rule
            rows = []
            for idx, pk in zip(dom.row_idx, dom.peak_row):
                rows.append(emb[int(pk)])
            X = np.vstack(rows)
        print(f"\nESM vectors: {X.shape}  (domain-pool={args.domain_pool})")

    X = (X - X.mean(0)) / (X.std(0) + 1e-8)

    # ---------------- cluster ---------------------------------------------
    print(f"clustering ({args.method}) ...")
    if args.method == "kmeans":
        from sklearn.cluster import KMeans
        lab = KMeans(n_clusters=args.k, n_init=10,
                     random_state=args.seed).fit_predict(X)
    else:
        try:
            from sklearn.cluster import HDBSCAN
        except ImportError:
            sys.exit("needs scikit-learn >= 1.3:  pip install -U scikit-learn")
        # "leaf" selects fine-grained sub-clusters instead of the largest stable
        # blob. On this data the default "eom" tends to return one giant cluster.
        lab = HDBSCAN(min_cluster_size=args.min_cluster_size,
                      cluster_selection_method="leaf" if args.method == "leaf" else "eom",
                      copy=True).fit_predict(X)
    dom["cluster"] = lab
    n_cl = len([c for c in set(lab) if c >= 0])
    print(f"  {n_cl} clusters, {(lab < 0).sum():,} domains unassigned "
          f"({(lab < 0).mean()*100:.0f}%)")
    if n_cl == 0:
        sys.exit("no clusters found -- lower --min-cluster-size and retry")

    # ---------------- features --------------------------------------------
    for name, pat in MOTIFS.items():
        dom[name] = dom.sequence.str.contains(pat, regex=True)
    print("\nmotif prevalence across all domains (background rates):")
    for name in MOTIFS:
        rate = dom[name].mean()
        flag = ""
        if dom[name].sum() < 20:
            flag = "  <- too rare to test, no statistical power"
        elif rate > 0.25:
            flag = "  <- matches too often to be specific, see warning below"
        print(f"  {name:<14} {rate*100:5.1f}%  ({int(dom[name].sum()):,} domains){flag}")
    print("  NOTE: these are plain regexes, not position weight matrices. The")
    print("  SUMO consensus is only 4 degenerate residues, so it fires by chance")
    print("  in any lysine- and glutamate-rich sequence. A 'SUMO enrichment' may")
    print("  be a composition effect. Confirm with GPS-SUMO or JASSA, and check")
    print("  whether hi_frac_basic/hi_frac_acidic are enriched in the same")
    print("  cluster -- if they are, suspect composition rather than motif.")
    comp = pd.DataFrame([composition_features(s) for s in dom.sequence])
    for c in comp.columns:                       # binarise at the global median
        dom[f"hi_{c}"] = comp[c] > comp[c].median()
    feat_cols = list(MOTIFS) + [f"hi_{c}" for c in comp.columns]

    # ---------------- per-cluster summary ----------------------------------
    print("measuring internal sequence identity and promoter robustness ...")
    # Background: similarity between random domain pairs anywhere in the data.
    # A cluster only counts as a homology family if it sits clearly ABOVE this.
    # A fixed cut-off would be wrong -- the scale depends on the k and on how
    # related the whole set happens to be.
    allseq = dom.sequence.tolist()
    bg = float(np.median([
        kmer_identity(allseq[i], allseq[j])
        for i, j in rng.integers(0, len(allseq), size=(600, 2)) if i != j
    ]))
    bg_cut = max(3 * bg, bg + 0.05)
    print(f"  background pairwise 5-mer identity = {bg:.4f}; "
          f"homology-family threshold = {bg_cut:.4f}")
    # Real homology labels, if MMseqs2 was run. These supersede the 5-mer proxy
    # for the homology-vs-convergent call.
    mm = None
    if args.mmseqs:
        t = pd.read_csv(args.mmseqs, sep="\t", header=None, names=["rep", "member"])
        mm = dict(zip(t.member, t.rep))
        dom["mmseqs_rep"] = dom.domain_id.map(mm)
        unmapped = int(dom.mmseqs_rep.isna().sum())
        print(f"  MMseqs2 labels: {dom.mmseqs_rep.nunique():,} families over "
              f"{len(dom):,} domains ({unmapped} unmapped)")
        dom["mmseqs_rep"] = dom.mmseqs_rep.fillna(dom.domain_id)

    rows = []
    for cl in sorted(c for c in dom.cluster.unique() if c >= 0):
        sub = dom[dom.cluster == cl]
        ident = cluster_identity(sub.sequence.tolist(), rng)
        # Purity against MMseqs2: what share of the cluster sits in its single
        # most common homology family. High purity = the cluster IS a family.
        if mm is not None:
            vc = sub.mmseqs_rep.value_counts()
            purity = float(vc.iloc[0] / len(sub))
            n_fam = int(sub.mmseqs_rep.nunique())
        else:
            purity, n_fam = np.nan, -1
        rows.append({
            "cluster": cl,
            "n_domains": len(sub),
            "n_proteins": sub.protein.nunique(),
            "mmseqs_purity": purity,
            "n_mmseqs_families": n_fam,
            "median_kmer_identity": ident,
            "identity_vs_background": (ident / bg) if (bg > 0 and ident == ident) else np.nan,
            # The distinction that decides whether a cluster is interesting.
            # Above background => the cluster is a protein family, already in
            # Pfam. Near background => unrelated sequences that landed together
            # anyway, which is the only kind that can be a new mechanism.
            # With MMseqs2 available the call is made on real homology: a
            # cluster drawn from many families at low purity is convergent.
            "kind": (("homology family" if purity >= 0.5 else "convergent")
                     if mm is not None else
                     ("homology family" if (ident or 0) > bg_cut else "convergent")),
            "frac_PGK_robust": float(sub.pgk_hit.mean()),
            "frac_bifunctional": float(sub.mincmv_hit.mean()),
            "median_max_pEF": float(sub.max_pEF.median()),
            "median_n_tiles": float(sub.n_tiles.median()),
            "example_proteins": ", ".join(sub.protein.value_counts().head(4).index),
        })
    cl_df = pd.DataFrame(rows)
    base_pgk = dom.pgk_hit.mean()
    cl_df["pgk_vs_background"] = cl_df.frac_PGK_robust - base_pgk
    # Wilson CI on the robustness fraction: mandatory, see the note in wilson().
    ci = [wilson(int(r.frac_PGK_robust * r.n_domains), int(r.n_domains))
          for _, r in cl_df.iterrows()]
    cl_df["pgk_ci_lo"] = [c[0] for c in ci]
    cl_df["pgk_ci_hi"] = [c[1] for c in ci]

    # ---------------- how well does strength alone predict robustness? -----
    # The claim being replaced is "92.6% of robustness is outside the model",
    # which came from 1 - r^2 of a point-biserial correlation. On a binary
    # target with a 24% base rate that denominator does not exist. AUC is the
    # measure that does: P(a robust domain ranks above a non-robust one).
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(dom.pgk_hit.values, dom.max_pEF.values)
        print(f"\n  ranking domains by repression strength alone (max_pEF):")
        print(f"     AUC for predicting PGK robustness = {auc:.4f}  "
              f"(0.5 = coin flip)")
        print(f"     base rate of robust domains        = {base_pgk:.4f}")
        print("     Better than chance, far from usable for picking candidates:")
        print("     strength and robustness are different axes.")
    except Exception as e:                      # pragma: no cover
        print(f"  (AUC skipped: {e})")

    # ---------------- homology control (the "what's new vs Pfam?" answer) ---
    if mm is not None:
        from sklearn.metrics import adjusted_mutual_info_score as ami
        assigned = dom[dom.cluster >= 0]
        score = ami(assigned.mmseqs_rep.astype(str), assigned.cluster)
        print(f"\n  AMI(embedding clusters, MMseqs2 families) = {score:.4f}")
        print("     high  -> the clustering is a protein-family classifier, so")
        print("              it is a positive control, not a discovery")
        print("     low   -> clusters cut across families; only then can a")
        print("              cluster be a convergent mechanism")

    # ---------------- enrichment ------------------------------------------
    print("testing motif enrichment (Fisher exact + BH FDR) ...")
    enr = enrichment_table(dom, feat_cols)
    sig = enr[enr.fdr < 0.05] if len(enr) else enr

    # ---------------- negative control -------------------------------------
    shuffled = dom.copy()
    shuffled["cluster"] = rng.permutation(dom.cluster.values)
    null = enrichment_table(shuffled, feat_cols)
    n_null = int((null.fdr < 0.05).sum()) if len(null) else 0
    print(f"  real: {len(sig)} enrichments at FDR<0.05 | "
          f"shuffled control: {n_null}")
    if n_null > max(2, 0.25 * max(len(sig), 1)):
        print("  WARNING: the shuffled control is also producing hits. "
              "Do not trust the enrichments.")

    # ---------------- report ----------------------------------------------
    print("\n" + "=" * 72)
    print("CLUSTERS  (convergent = low internal identity = not just a Pfam family)")
    print("=" * 72)
    show = cl_df.sort_values("n_domains", ascending=False).head(15)
    for _, r in show.iterrows():
        tag = "CONVERGENT" if r.kind == "convergent" else "homology  "
        pur = "   -  " if r.mmseqs_purity != r.mmseqs_purity else f"pur={r.mmseqs_purity:.2f}"
        print(f"  c{int(r.cluster):<3} n={int(r.n_domains):<4} {tag} {pur} "
              f"fam={int(r.n_mmseqs_families):<4} "
              f"PGK-robust={r.frac_PGK_robust:.2f} "
              f"[{r.pgk_ci_lo:.2f}-{r.pgk_ci_hi:.2f}] (bg {base_pgk:.2f})  "
              f"pEF={r.median_max_pEF:.2f}")
        print(f"        {r.example_proteins}")
    print("\n  The bracket is the Wilson 95% CI. A cluster whose interval")
    print(f"  contains the background ({base_pgk:.2f}) is not distinguishable")
    print("  from it, however extreme the point estimate looks.")

    if len(sig):
        print("\n" + "=" * 72)
        print("ENRICHED FEATURES  (FDR < 0.05)")
        print("=" * 72)
        print(f"  {'':5}{'feature':<22}{'n':>5}{'in%':>6}{'out%':>6}"
              f"{'OR':>8}{'95% CI':>18}{'FDR':>9}")
        for _, r in sig.head(20).iterrows():
            ci = f"[{r.or_ci_lo:.1f}, {r.or_ci_hi:.1f}]"
            print(f"  c{int(r.cluster):<3} {r.feature:<22}{int(r.cluster_size):>5}"
                  f"{r.frac_in*100:>6.0f}{r.frac_out*100:>6.0f}"
                  f"{r.or_haldane:>8.2f}{ci:>18}{r.fdr:>9.1e}")
        print("\n  OR is Haldane-corrected. A CI spanning two orders of")
        print("  magnitude means the point estimate carries no weight, even")
        print("  when the FDR looks decisive -- read the CI, not the OR.")
    else:
        print("\nNo feature enrichment survives FDR correction.")
        print("That is a real result, not a failure: it means these clusters are")
        print("not organised by the known co-repressor motifs.")

    print("\n" + "=" * 72)
    print("VALIDITY CHECKS -- do these before trusting any novel cluster")
    print("=" * 72)
    print("  1. Does a cluster recover a KNOWN family (KRAB zinc fingers, HES/HEY")
    print("     with WRPW, CtBP-binding)? If known biology does not come out,")
    print("     do not believe the unknown clusters.")
    print(f"  2. Shuffled-label control produced {n_null} enrichments "
          f"(should be ~0).")
    print("  3. A cluster is a CANDIDATE NEW MECHANISM only if all four hold:")
    print("       n_domains >= 20, kind == convergent, no enriched known motif,")
    print("       median_max_pEF above background.")
    print("  4. Check mean_FLAG per cluster. A cluster of poorly expressed")
    print("     domains is a measurement artefact, not a mechanism.")

    dom.drop(columns=["row_idx"]).to_csv(f"{args.out}_domains.csv", index=False)
    cl_df.to_csv(f"{args.out}_clusters.csv", index=False)
    if len(enr):
        enr.to_csv(f"{args.out}_enrichment.csv", index=False)
    print(f"\nwrote {args.out}_domains.csv, {args.out}_clusters.csv, "
          f"{args.out}_enrichment.csv")


if __name__ == "__main__":
    main()
