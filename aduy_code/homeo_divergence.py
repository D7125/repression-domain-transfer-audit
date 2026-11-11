"""
Which residue positions separate a promoter-robust homeodomain from a fragile one?

This is the within-family experiment the patch doc's section 7 argues for, on
the cleanest possible comparison: 44 repression domains that Pfam annotates as
Homeodomain 100%, split 20 robust / 24 not. Same family, same length, same
fold -- so the family is its own control and length, expression and
composition are matched by construction.

Statistics for a small n and many columns:
  * per column, mutual information between the residue and the robust label
  * a PERMUTATION test on the MAX MI across columns. Shuffling the labels and
    re-taking the maximum gives the null for "the best of 57 columns", which
    is the quantity actually being looked at. A per-column p-value with BH
    would be weaker here, because the columns are not independent (one
    substitution often tracks a whole sub-family).
  * residues are grouped into physicochemical classes as well as tested raw:
    with 20 sequences a specific residue rarely repeats, but a charge change
    does.

Usage:
  python homeo_divergence.py --sto homeo44.sto --meta homeo44_meta.csv
"""

import argparse
from collections import Counter

import numpy as np
import pandas as pd

# Physicochemical classes: a 20-letter alphabet is too sparse at n=44.
CLASS = {}
for aa, cl in [("KRH", "basic"), ("DE", "acidic"), ("NQST", "polar"),
               ("AVLIMFWYC", "hydrophobic"), ("GP", "small"), ("-", "gap")]:
    for a in aa:
        CLASS[a] = cl


def read_stockholm(path):
    seqs = {}
    for line in open(path):
        line = line.rstrip("\n")
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        parts = line.split()
        if len(parts) == 2:
            seqs.setdefault(parts[0], "")
            seqs[parts[0]] += parts[1]
    return seqs


def mutual_info(col, labels):
    """MI in bits between a residue column and a binary label."""
    n = len(labels)
    mi = 0.0
    for res in set(col):
        for lab in (0, 1):
            joint = sum(1 for c, l in zip(col, labels) if c == res and l == lab)
            if not joint:
                continue
            p_joint = joint / n
            p_res = col.count(res) / n
            p_lab = sum(labels) / n if lab else 1 - sum(labels) / n
            mi += p_joint * np.log2(p_joint / (p_res * p_lab))
    return mi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sto", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--perms", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    seqs = read_stockholm(args.sto)
    meta = pd.read_csv(args.meta)
    # names in the alignment carry |cN|ben as suffixes; key on the domain id
    byid = {k.split("|")[0]: v for k, v in seqs.items()}
    meta = meta[meta.domain_id.isin(byid)].reset_index(drop=True)
    aln = [byid[d] for d in meta.domain_id]
    L = len(aln[0])
    labels = meta.pgk_hit.astype(int).tolist()
    print(f"{len(aln)} aligned sequences, {L} alignment columns")
    print(f"  robust {sum(labels)} | not robust {len(labels) - sum(labels)}")
    print(f"  cluster split: {dict(meta.cluster.value_counts())}")

    for mode in ("raw residue", "physicochemical class"):
        cols = []
        for i in range(L):
            col = [a[i] for a in aln]
            if mode != "raw residue":
                col = [CLASS.get(c, "other") for c in col]
            cols.append(col)
        mis = np.array([mutual_info(c, labels) for c in cols])

        # permutation null for the MAXIMUM across columns
        null_max = np.empty(args.perms)
        lab = np.array(labels)
        for p in range(args.perms):
            sh = rng.permutation(lab).tolist()
            null_max[p] = max(mutual_info(c, sh) for c in cols)

        obs_max = mis.max()
        pval = (1 + (null_max >= obs_max).sum()) / (args.perms + 1)
        print(f"\n--- {mode} ---")
        print(f"  best column MI = {obs_max:.4f} bits at column {int(mis.argmax()) + 1}")
        print(f"  permutation null for the max: median {np.median(null_max):.4f}, "
              f"95th pct {np.percentile(null_max, 95):.4f}")
        print(f"  p (max-statistic, {args.perms:,} perms) = {pval:.4f}")
        if pval < 0.05:
            print("  -> at least one position separates robust from fragile")
        else:
            print("  -> no position separates them beyond what label shuffling gives.")
            print("     With 20 vs 24 sequences this is the expected outcome unless")
            print("     the effect is large; it does NOT show the domains are identical.")

        # show the top columns regardless, as description
        order = np.argsort(mis)[::-1][:5]
        print(f"  top 5 columns by MI (descriptive only):")
        for i in order:
            rob = Counter(c for c, l in zip(cols[i], labels) if l == 1)
            fra = Counter(c for c, l in zip(cols[i], labels) if l == 0)
            print(f"    col {i+1:>3}  MI={mis[i]:.3f}  "
                  f"robust={dict(rob.most_common(3))}  "
                  f"fragile={dict(fra.most_common(3))}")

    # What distinguishes the two CLUSTERS (not the robust label)? This says
    # what sub-families HDBSCAN actually separated.
    print("\n=== what separates cluster 4 from cluster 5 ===")
    cl = (meta.cluster == 5).astype(int).tolist()
    cols = [[a[i] for a in aln] for i in range(L)]
    mis_cl = np.array([mutual_info(c, cl) for c in cols])
    print(f"  best column MI = {mis_cl.max():.4f} bits "
          f"(vs {np.array([mutual_info(c, labels) for c in cols]).max():.4f} for the robust label)")
    print("  If the clusters separate much more cleanly than robustness does,")
    print("  HDBSCAN split sub-families and robustness only rides along.")
    print(f"  representative proteins: c4 = "
          f"{', '.join(meta[meta.cluster==4].protein.head(5))}")
    print(f"                           c5 = "
          f"{', '.join(meta[meta.cluster==5].protein.head(5))}")

    # ------------------------------------------------------------------
    # The confound test. The two clusters are distinct sub-families, and
    # robustness is unevenly distributed between them, so a column that
    # merely marks the sub-family will look like it marks robustness.
    # Fix: permute the robust label ONLY WITHIN each cluster. That keeps
    # the sub-family structure intact and asks whether residues still
    # track robustness once sub-family is held constant.
    # ------------------------------------------------------------------
    print("\n=== CONFOUND TEST: permute robustness WITHIN each cluster ===")
    cl_arr = meta.cluster.values
    lab_arr = np.array(labels)
    for c in (4, 5):
        m = cl_arr == c
        print(f"  cluster {c}: {lab_arr[m].sum()} robust / {(~lab_arr[m].astype(bool)).sum()} not")

    for mode in ("raw residue", "physicochemical class"):
        cols = []
        for i in range(L):
            col = [a[i] for a in aln]
            if mode != "raw residue":
                col = [CLASS.get(c, "other") for c in col]
            cols.append(col)
        obs_max = max(mutual_info(c, labels) for c in cols)

        null_max = np.empty(args.perms)
        for p in range(args.perms):
            sh = lab_arr.copy()
            for c in (4, 5):                      # shuffle inside each cluster
                m = cl_arr == c
                sh[m] = rng.permutation(sh[m])
            sh = sh.tolist()
            null_max[p] = max(mutual_info(c, sh) for c in cols)

        pval = (1 + (null_max >= obs_max).sum()) / (args.perms + 1)
        print(f"\n  {mode}: observed max MI = {obs_max:.4f}")
        print(f"    within-cluster null: median {np.median(null_max):.4f}, "
              f"95th pct {np.percentile(null_max, 95):.4f}")
        print(f"    p = {pval:.4f}")
        if pval < 0.05:
            print("    -> survives. Residues track robustness even with")
            print("       sub-family held constant: a real within-family signal.")
        else:
            print("    -> does NOT survive. The earlier p-value came from the")
            print("       sub-family split, not from robustness. Report the")
            print("       sub-family difference instead, and do not claim")
            print("       residue-level determinants of promoter robustness.")


if __name__ == "__main__":
    main()
