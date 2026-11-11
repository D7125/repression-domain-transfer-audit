"""
Task 4: the composition + disorder control.

Criterion 5 for calling a cluster a new mechanism: it must NOT reappear when
you cluster on amino-acid composition alone. If the ESM clusters and the
composition clusters agree, ESM contributed nothing beyond composition, and
"we found a proline-rich cluster" is not a mechanism.

Procedure:
  1. feature vector = 20 aa frequencies + net charge + a disorder proxy
     + predicted-helix proxy
  2. cluster with the SAME algorithm and parameters as the ESM run
  3. AMI between the two labellings

Disorder note: this uses a TOP-IDP-style disorder-propensity proxy, NOT
IUPred3. A real run should install IUPred3; the proxy is here so the control
exists at all rather than being skipped.

Usage:
  python composition_control.py --domains esm_domains.csv --out comp_ctrl
"""

import argparse

import numpy as np
import pandas as pd

AA = "ACDEFGHIKLMNPQRSTVWY"
# Disorder-promoting vs order-promoting residues (Dunker/Uversky ordering).
DISORDER_PROMOTING = "PESKQRGA"
ORDER_PROMOTING = "WCFIYVLN"


def featurise(seq):
    n = max(len(seq), 1)
    f = {f"f_{a}": seq.count(a) / n for a in AA}
    f["net_charge"] = (seq.count("K") + seq.count("R")
                       - seq.count("D") - seq.count("E")) / n
    f["disorder_proxy"] = sum(seq.count(a) for a in DISORDER_PROMOTING) / n
    f["order_proxy"] = sum(seq.count(a) for a in ORDER_PROMOTING) / n
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", required=True,
                    help="<out>_domains.csv from the ESM clustering run")
    ap.add_argument("--min-cluster-size", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--groups", default=None,
                    help="MMseqs2 domain cluster TSV. Required for an honest "
                         "criterion-5 verdict: without it the CV lets the "
                         "classifier recognise paralogs instead of composition.")
    ap.add_argument("--out", default="comp_ctrl")
    args = ap.parse_args()

    from sklearn.cluster import HDBSCAN
    from sklearn.metrics import adjusted_mutual_info_score as ami

    dom = pd.read_csv(args.domains)
    X = pd.DataFrame([featurise(s) for s in dom.sequence])
    print(f"{len(dom):,} domains | {X.shape[1]} composition features")
    Xs = ((X - X.mean()) / (X.std() + 1e-8)).values

    lab = HDBSCAN(min_cluster_size=args.min_cluster_size,
                  cluster_selection_method="leaf", copy=True).fit_predict(Xs)
    dom["comp_cluster"] = lab
    n_cl = len([c for c in set(lab) if c >= 0])
    print(f"composition-only clustering: {n_cl} clusters, "
          f"{(lab < 0).sum():,} unassigned ({(lab < 0).mean()*100:.0f}%)")

    # Compare only where BOTH labellings assigned a cluster: comparing against
    # noise points would reward the two runs for agreeing on what to discard.
    both = dom[(dom.cluster >= 0) & (dom.comp_cluster >= 0)]
    print(f"\ndomains assigned by both: {len(both):,}")
    if len(both) > 10:
        score = ami(both.cluster, both.comp_cluster)
        print(f"AMI(ESM clusters, composition clusters) = {score:.4f}")
        print("  near 1 -> ESM added nothing over composition; criterion 5 FAILS")
        print("  near 0 -> the ESM structure is not reducible to composition")
    else:
        print("too little overlap to compare")

    # ------------------------------------------------------------------
    # HDBSCAN returning zero clusters does NOT show composition is
    # uninformative -- it only shows there are no density peaks in this space.
    # Two stronger tests follow.
    # ------------------------------------------------------------------
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.dummy import DummyClassifier

    assigned = dom[dom.cluster >= 0]
    k = assigned.cluster.nunique()
    if k >= 2:
        # (a) same resolution, forced: k-means on composition with the same k
        km = KMeans(n_clusters=k, n_init=10, random_state=args.seed).fit_predict(Xs)
        dom["comp_kmeans"] = km
        a = dom[dom.cluster >= 0]
        print(f"\n(a) k-means on composition, k={k} (matched resolution)")
        print(f"    AMI(ESM clusters, composition k-means) = "
              f"{ami(a.cluster, a.comp_kmeans):.4f}")

        # (b) supervised: can composition PREDICT the ESM cluster label?
        #
        # The CV MUST be grouped by homology family. An ungrouped split puts
        # ALX1 in train and ALX3 in test, and the classifier then scores well
        # by recognising a paralog rather than by composition carrying the
        # information. Measured both ways here, because the gap between them
        # is large enough to reverse the conclusion: ungrouped balanced
        # accuracy ~0.87 vs grouped ~0.25 on this data.
        #
        # Raw accuracy is also useless when one cluster holds 93% of the
        # domains, so balanced accuracy and macro F1 are what get reported.
        from sklearn.model_selection import GroupKFold, cross_val_predict
        from sklearn.metrics import balanced_accuracy_score, f1_score

        y = assigned.cluster.values
        Xa = Xs[(dom.cluster >= 0).values]
        clf = LogisticRegression(max_iter=5000)
        maj = pd.Series(y).value_counts(normalize=True).iloc[0]
        chance = 1.0 / len(set(y))
        print(f"\n(b) predicting the ESM cluster label from composition alone")
        print(f"    majority class holds {maj:.1%} of domains; "
              f"chance balanced accuracy = {chance:.4f}")

        if args.groups:
            t = pd.read_csv(args.groups, sep="\t", header=None,
                            names=["rep", "member"])
            fam = dom.domain_id.map(dict(zip(t.member, t.rep))).fillna(dom.domain_id)
            g = fam.values[(dom.cluster >= 0).values]
            pred = cross_val_predict(clf, Xa, y, cv=GroupKFold(n_splits=5),
                                     groups=g)
            tag = f"GroupKFold by {len(set(g)):,} homology families"
        else:
            folds = StratifiedKFold(5, shuffle=True, random_state=args.seed)
            pred = cross_val_predict(clf, Xa, y, cv=folds)
            tag = "StratifiedKFold -- LEAKS PARALOGS, pass --groups"

        bacc = balanced_accuracy_score(y, pred)
        print(f"    CV: {tag}")
        print(f"    raw accuracy       = {(pred == y).mean():.4f}  "
              f"(baseline {maj:.4f})")
        print(f"    balanced accuracy  = {bacc:.4f}  (chance {chance:.4f})")
        print(f"    macro F1           = {f1_score(y, pred, average='macro', zero_division=0):.4f}")
        if not args.groups:
            print("    -> UNGROUPED. Do not draw a criterion-5 conclusion from this.")
        elif bacc > 0.5 + 0.5 * chance:
            print("    -> composition recovers the labelling even across")
            print("       families. CRITERION 5 FAILS: report these as")
            print("       composition groups, not mechanisms.")
        else:
            print("    -> composition does NOT recover the labelling once")
            print("       paralogs are held out. CRITERION 5 PASSES: the")
            print("       cluster structure is not reducible to composition.")

    # Per-ESM-cluster: is it recovered by a single composition cluster?
    print("\nper-ESM-cluster recovery by composition alone:")
    for cl in sorted(c for c in dom.cluster.unique() if c >= 0):
        sub = dom[dom.cluster == cl]
        asg = sub[sub.comp_cluster >= 0]
        if not len(asg):
            print(f"  c{cl}: n={len(sub)} -- composition clustering assigned none")
            continue
        vc = asg.comp_cluster.value_counts()
        print(f"  c{cl}: n={len(sub):<4} {len(asg):>4} assigned by composition, "
              f"largest single composition cluster holds "
              f"{vc.iloc[0]}/{len(asg)} ({vc.iloc[0]/len(asg)*100:.0f}%)")

    dom.to_csv(f"{args.out}_domains.csv", index=False)
    print(f"\nwrote {args.out}_domains.csv")


if __name__ == "__main__":
    main()
