"""
B3: the baseline the paper's own argument demands.

Section 3.1 spends its length proving that regex motifs mostly count amino
acid composition. Section 3.2 then reports MotifCNN r = 0.585 with no
composition baseline anywhere -- only mean-pooled ESM (3.2) and Avg_pEF (3.4).
If ridge regression on 20 amino acid frequencies reaches r = 0.50, then most of
what the ESM model contributes is composition too, and 3.2-3.3 have to be read
differently. This is the cheapest baseline imaginable and it is the one 3.1
asks for.

Three feature sets, identical splits/seeds to the deep arms:
  aa20    20 residue frequencies + tile length     (the baseline 3.1 implies)
  aa20+   the above plus 5 physico-chemical fractions already in the pipeline
  dipep   400 dipeptide frequencies                (composition with one step
          of local order -- an upper bound on "composition-only")
Targets: Avg_pEF (the published arm) and cd43 (the dCas9 arm).
"""
import argparse
import json
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import sys
sys.path.insert(0, "/home/dmin/ai&stem_cell/cnn_AI&stemcell")
from train_cnn import grouped_split

ROOT = "/home/dmin/ai&stem_cell"
AA = "ACDEFGHIKLMNPQRSTVWY"
IDX = {a: i for i, a in enumerate(AA)}
GROUPS = {"acidic": "DE", "basic": "KR", "aromatic": "FWY",
          "hydrophobic": "AVILMFWC", "polar": "STNQ"}


def featurise(seqs):
    n = len(seqs)
    aa = np.zeros((n, 20), dtype=np.float32)
    dip = np.zeros((n, 400), dtype=np.float32)
    extra = np.zeros((n, len(GROUPS) + 1), dtype=np.float32)
    for i, s in enumerate(seqs):
        s = s.strip().upper()
        L = max(len(s), 1)
        for c in s:
            j = IDX.get(c)
            if j is not None:
                aa[i, j] += 1
        for a, b in zip(s, s[1:]):
            ja, jb = IDX.get(a), IDX.get(b)
            if ja is not None and jb is not None:
                dip[i, ja * 20 + jb] += 1
        aa[i] /= L
        dip[i] /= max(L - 1, 1)
        for k, (_, letters) in enumerate(GROUPS.items()):
            extra[i, k] = sum(c in letters for c in s) / L
        extra[i, -1] = L
    return aa, dip, extra


def score(Xtr, ytr, Xte, yte):
    mdl = make_pipeline(StandardScaler(),
                        RidgeCV(alphas=np.logspace(-2, 4, 25)))
    mdl.fit(Xtr, ytr)
    p = mdl.predict(Xte)
    return float(pearsonr(p, yte)[0]), float(spearmanr(p, yte).statistic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="ridge is closed form, so these are split seeds only")
    ap.add_argument("--out", default=f"{ROOT}/run/rev2/composition_baseline.csv")
    ap.add_argument("--match-cd43-subset", action="store_true",
                    help="score on exactly the tiles the CD43 transfer table "
                         "scores -- test tiles with CD43 in BOTH bioreps, for "
                         "both targets -- so the ridge row and the CNN rows "
                         "share a test set and a ceiling. Training masks follow "
                         "cd43_model.py: the cd43 arm trains only where the "
                         "target exists, the pEF arm trains on everything.")
    args = ap.parse_args()

    df = pd.read_csv(f"{ROOT}/run/rev2/master.csv", low_memory=False)
    print(f"featurising {len(df):,} tiles ...", flush=True)
    aa, dip, extra = featurise(df.sequence.astype(str).values)
    sets = {"aa20": np.hstack([aa, extra[:, -1:]]),
            "aa20+": np.hstack([aa, extra]),
            "dipep": np.hstack([dip, extra[:, -1:]])}

    cl = pd.read_csv(f"{ROOT}/run/protein_clusters.tsv", sep="\t",
                     header=None, names=["rep", "member"])
    cmap = dict(zip(cl["member"], cl["rep"]))
    groups = np.array([cmap.get(g, g) for g in df.protein.astype(str)])

    has_cd43 = (df.cd43.notna().values & df.cd43_br1.notna().values
                & df.cd43_br2.notna().values)
    if args.match_cd43_subset:
        print(f"matching the CD43 transfer table: scoring on the "
              f"{has_cd43.sum():,} tiles with CD43 in both bioreps", flush=True)

    rows = []
    for target in ("Avg_pEF", "cd43"):
        y = df[target].values
        ok = pd.notna(y)
        tr_ok = ok & has_cd43 if (args.match_cd43_subset
                                  and target == "cd43") else ok
        te_ok = ok & has_cd43 if args.match_cd43_subset else ok
        for seed in args.seeds:
            tr, va, te = grouped_split(groups, seed)
            tr = tr[tr_ok[tr]]; te = te[te_ok[te]]
            for name, X in sets.items():
                r, rho = score(X[tr], y[tr], X[te], y[te])
                rows.append(dict(target=target, seed=seed, features=name,
                                 n_train=len(tr), n_test=len(te),
                                 test_r=r, test_rho=rho))
                print(f"  {target:8s} seed {seed}  {name:6s} "
                      f"r = {r:.4f}  rho = {rho:.4f}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"\n=== mean over {len(args.seeds)} seeds ===")
    piv = out.pivot_table(index="features", columns="target",
                          values="test_r", aggfunc=["mean", "std"])
    print(piv.round(4).to_string())
    print(f"\nwrote {args.out}")
    print("DONE_COMPOSITION_BASELINE")


if __name__ == "__main__":
    main()
