"""
Does the pooling choice change the clustering?

Two pooling decisions sit between the ESM tensor and a domain vector, and
neither has an obvious right answer:

  level A  80 residues -> 1 tile      mean | max
  level B  several tiles -> 1 domain  mean | max | strongest-tile

The original pipeline fixed both to mean without saying so. This runs the
same clustering over the combinations and reports AMI between the resulting
labellings.

  AMI high across the board -> the choice does not matter, and one line in
     the methods section covers it.
  AMI low -> the choice IS a result and has to be stated, because the cluster
     membership a reader sees depends on it.

Usage:
  python pooling_compare.py --csv data.csv \
      --emb-mean esm2_480_mean.npy --emb-max esm2_480_max.npy --out pool_cmp
"""

import argparse
import itertools
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from cluster_domains import build_domains          # noqa: E402


def domain_matrix(emb, dom, level_b):
    if level_b == "mean":
        return np.vstack([emb[idx].mean(0) for idx in dom.row_idx])
    if level_b == "max":
        return np.vstack([emb[idx].max(0) for idx in dom.row_idx])
    return np.vstack([emb[int(p)] for p in dom.peak_row])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--emb-mean", required=True)
    ap.add_argument("--emb-max", required=True)
    ap.add_argument("--min-cluster-size", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="pool_cmp")
    args = ap.parse_args()

    from sklearn.cluster import HDBSCAN
    from sklearn.metrics import adjusted_mutual_info_score as ami

    df = pd.read_csv(args.csv)
    dom = build_domains(df)
    print(f"{len(dom):,} domains")

    embs = {"mean": np.load(args.emb_mean), "max": np.load(args.emb_max)}
    labels, summary = {}, []

    for a, b in itertools.product(["mean", "max"], ["mean", "max", "strongest"]):
        X = domain_matrix(embs[a], dom, b)
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)
        lab = HDBSCAN(min_cluster_size=args.min_cluster_size,
                      cluster_selection_method="leaf", copy=True).fit_predict(X)
        key = f"A={a},B={b}"
        labels[key] = lab
        n_cl = len([c for c in set(lab) if c >= 0])
        summary.append({"combo": key, "n_clusters": n_cl,
                        "pct_unassigned": round((lab < 0).mean() * 100, 1),
                        "largest": int(pd.Series(lab[lab >= 0]).value_counts().iloc[0])
                        if n_cl else 0})
        print(f"  {key:22s} {n_cl:>3} clusters, "
              f"{(lab < 0).mean()*100:4.1f}% unassigned")

    print("\n" + "=" * 70)
    print("AMI BETWEEN POOLING CHOICES  (1.0 = identical partitions)")
    print("=" * 70)
    keys = list(labels)
    M = pd.DataFrame(index=keys, columns=keys, dtype=float)
    for i in keys:
        for j in keys:
            # compare only domains that both runs assigned; agreeing on which
            # points are noise is not agreement about cluster structure
            m = (labels[i] >= 0) & (labels[j] >= 0)
            M.loc[i, j] = ami(labels[i][m], labels[j][m]) if m.sum() > 10 else np.nan
    pd.set_option("display.width", 200)
    print(M.round(3).to_string())

    print("\nthe comparison that was asked for -- level A, mean vs max:")
    for b in ["mean", "max", "strongest"]:
        i, j = f"A=mean,B={b}", f"A=max,B={b}"
        print(f"  with B={b:<10} AMI = {M.loc[i, j]:.4f}")
    print("\nlevel B, mean vs the paper's strongest-tile rule:")
    for a in ["mean", "max"]:
        i, j = f"A={a},B=mean", f"A={a},B=strongest"
        print(f"  with A={a:<5} AMI = {M.loc[i, j]:.4f}")

    pd.DataFrame(summary).to_csv(f"{args.out}_summary.csv", index=False)
    M.to_csv(f"{args.out}_ami.csv")
    out = dom[["domain_id", "protein", "max_pEF", "pgk_hit"]].copy()
    for k, v in labels.items():
        out[k] = v
    out.to_csv(f"{args.out}_labels.csv", index=False)
    print(f"\nwrote {args.out}_ami.csv, {args.out}_summary.csv, {args.out}_labels.csv")


if __name__ == "__main__":
    main()
