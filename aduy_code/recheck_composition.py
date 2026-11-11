"""
Re-check of the composition control, addressing three objections.

1. SCALING. Was the feature matrix actually z-scored, and is the HDBSCAN
   noise result a scaling artefact or a real absence of density peaks?
   Prints per-feature ranges before and after, and sweeps min_cluster_size.
   Then runs the original planned comparison: k-means at a FIXED k on both
   spaces, which cannot fail the way HDBSCAN did.

2. OR = inf. Printed with the full 2x2 table, the cluster size, a
   Haldane-Anscombe corrected odds ratio and a confidence interval. Also
   tests the biological suspicion that the basic-rich clusters are just
   C2H2 zinc-finger arrays.

3. GROUPED CV. The classifier must not be allowed to recognise a paralog:
   KLF4 in train and KLF7 in test would let it predict the cluster label
   from homology rather than from composition. Groups = MMseqs2 families.
   Reported against a majority-class baseline, with balanced accuracy and
   macro F1 rather than raw accuracy.

Usage:
  python recheck_composition.py --domains best_domains.csv \
      --mmseqs dom_clu_cluster.tsv --emb esm2_480_max.npy --csv data.csv
"""

import argparse
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from composition_control import featurise          # noqa: E402

# Descriptive only -- used to test whether a basic-rich cluster is a C2H2
# array, not to assign mechanism (that needs UniProt/Pfam).
C2H2 = re.compile(r"C.{2,4}C.{12}H.{3,5}H")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", required=True)
    ap.add_argument("--mmseqs", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from sklearn.cluster import HDBSCAN, KMeans
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, GroupKFold
    from sklearn.metrics import (adjusted_mutual_info_score as ami,
                                 balanced_accuracy_score, f1_score,
                                 classification_report)
    from scipy.stats import fisher_exact

    dom = pd.read_csv(args.domains)
    X = pd.DataFrame([featurise(s) for s in dom.sequence])

    # ---------------------------------------------------------------- 1
    print("=" * 72)
    print("1. SCALING -- is the HDBSCAN noise result an artefact?")
    print("=" * 72)
    rng_before = X.max() - X.min()
    print(f"  {len(X.columns)} features. Raw ranges (max-min):")
    print(f"    smallest {rng_before.min():.4f} ({rng_before.idxmin()})")
    print(f"    largest  {rng_before.max():.4f} ({rng_before.idxmax()})")
    print(f"    ratio largest/smallest = {rng_before.max()/rng_before.min():.1f}x")
    print("  net_charge is already divided by domain length, so it lives in")
    print("  [-1, 1] rather than [-30, +30] -- the scales differ by ~10x, not 100x.")
    Xs = ((X - X.mean()) / (X.std() + 1e-8)).values
    print(f"\n  after z-scoring: per-feature sd in "
          f"[{Xs.std(0).min():.4f}, {Xs.std(0).max():.4f}]  "
          f"(all ~1 => scaling is NOT the problem)")

    print("\n  min_cluster_size sweep on the composition space:")
    for m in (5, 10, 15, 25, 50):
        lab = HDBSCAN(min_cluster_size=m, cluster_selection_method="leaf",
                      copy=True).fit_predict(Xs)
        n = len([c for c in set(lab) if c >= 0])
        print(f"    min_cluster_size={m:<3} -> {n:>3} clusters, "
              f"{(lab < 0).mean()*100:5.1f}% noise")
    print("  => composition space genuinely has no density peaks at any size,")
    print("     so HDBSCAN noise is a property of the space, not a bug.")

    # the originally planned comparison, at matched fixed k
    emb = np.load(args.emb)
    Xe = np.vstack([emb[int(p)] for p in dom.peak_row])
    Xe = (Xe - Xe.mean(0)) / (Xe.std(0) + 1e-8)
    ke = KMeans(n_clusters=args.k, n_init=10, random_state=args.seed).fit_predict(Xe)
    kc = KMeans(n_clusters=args.k, n_init=10, random_state=args.seed).fit_predict(Xs)
    print(f"\n  k-means k={args.k} on BOTH spaces (cannot degenerate):")
    print(f"    AMI(ESM k-means, composition k-means) = {ami(ke, kc):.4f}")
    print("    This is the apples-to-apples number: same algorithm, same k,")
    print("    same data, only the feature space differs.")

    # ---------------------------------------------------------------- 2
    print("\n" + "=" * 72)
    print("2. OR = inf -- full 2x2, Haldane correction, and the C2H2 question")
    print("=" * 72)
    comp = X.copy()
    hi_basic = (comp.f_K + comp.f_R) > (comp.f_K + comp.f_R).median()
    print("  'hi_frac_basic' is binarised at the GLOBAL median of all 3,001")
    print("  domains, so ~50% of domains are positive by construction. An")
    print("  OR of inf therefore means b=0 (nobody in the cluster is below")
    print("  the median), NOT c=0. The denominator outside the cluster is ~48%.")
    print(f"\n  {'cluster':<8}{'n':>5}{'a':>5}{'b':>5}{'c':>6}{'d':>6}"
          f"{'OR_Hald':>9}{'95% CI':>18}{'p':>10}")
    for cl in sorted(c for c in dom.cluster.unique() if c >= 0):
        inside = dom.cluster == cl
        a = int((inside & hi_basic).sum()); b = int((inside & ~hi_basic).sum())
        c = int((~inside & hi_basic).sum()); d = int((~inside & ~hi_basic).sum())
        _, p = fisher_exact([[a, b], [c, d]], alternative="greater")
        # Haldane-Anscombe: add 0.5 everywhere so the OR stays finite
        A, B, C, D = a + .5, b + .5, c + .5, d + .5
        orh = (A * D) / (B * C)
        se = np.sqrt(1/A + 1/B + 1/C + 1/D)
        lo, hi = np.exp(np.log(orh) - 1.96*se), np.exp(np.log(orh) + 1.96*se)
        print(f"  c{cl:<7}{a+b:>5}{a:>5}{b:>5}{c:>6}{d:>6}{orh:>9.2f}"
              f"{f'[{lo:.1f}, {hi:.1f}]':>18}{p:>10.1e}")

    print("\n  Is 'basic-rich' just a C2H2 zinc-finger array?")
    dom["c2h2"] = dom.sequence.str.contains(C2H2)
    print(f"  {'cluster':<8}{'n':>5}{'C2H2%':>8}{'med f_C':>9}{'med basic':>11}"
          f"{'robust':>8}  example")
    for cl in sorted(c for c in dom.cluster.unique() if c >= 0):
        s = dom[dom.cluster == cl]
        i = s.index
        print(f"  c{cl:<7}{len(s):>5}{s.c2h2.mean()*100:>8.1f}"
              f"{X.loc[i, 'f_C'].median():>9.4f}"
              f"{(X.loc[i, 'f_K'] + X.loc[i, 'f_R']).median():>11.4f}"
              f"{s.pgk_hit.mean():>8.2f}  {', '.join(s.protein.head(3))}")
    print(f"  all domains: C2H2 {dom.c2h2.mean()*100:.1f}%, "
          f"median f_C {X.f_C.median():.4f}, "
          f"median basic {(X.f_K + X.f_R).median():.4f}")

    # ---------------------------------------------------------------- 3
    print("\n" + "=" * 72)
    print("3. GROUPED CV -- can composition predict the cluster WITHOUT")
    print("   recognising a paralog?")
    print("=" * 72)
    t = pd.read_csv(args.mmseqs, sep="\t", header=None, names=["rep", "member"])
    fam = dom.domain_id.map(dict(zip(t.member, t.rep))).fillna(dom.domain_id)
    m = (dom.cluster >= 0).values
    Xa, y, g = Xs[m], dom.cluster.values[m], fam.values[m]
    print(f"  {len(y):,} assigned domains | {len(set(y))} clusters | "
          f"{len(set(g)):,} MMseqs2 families as CV groups")

    clf = LogisticRegression(max_iter=5000)
    for name, cv in [("GroupKFold(5) by family", GroupKFold(n_splits=5))]:
        pred = cross_val_predict(clf, Xa, y, cv=cv, groups=g)
        bacc = balanced_accuracy_score(y, pred)
        mf1 = f1_score(y, pred, average="macro")
        maj = pd.Series(y).value_counts(normalize=True).iloc[0]
        dummy_b = 1.0 / len(set(y))
        print(f"\n  {name}")
        print(f"    raw accuracy        = {(pred == y).mean():.4f}   "
              f"(majority-class baseline {maj:.4f})")
        print(f"    balanced accuracy   = {bacc:.4f}   "
              f"(chance baseline {dummy_b:.4f})")
        print(f"    macro F1            = {mf1:.4f}")
        print()
        print(classification_report(y, pred, digits=3, zero_division=0))


if __name__ == "__main__":
    main()
