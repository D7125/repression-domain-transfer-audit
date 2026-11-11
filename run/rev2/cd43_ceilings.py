"""Recompute the pEF ceiling on the CD43 test subsets (cd43_model.py left it
NaN because a few tiles lack one pEF replicate).

Takes the input/output stems as arguments so it can be pointed at the
deterministic rerun instead of only the original three-seed file:

  python cd43_ceilings.py cd43_fixed_det10 cd43_fixed_det10_final
"""
import sys
import numpy as np, pandas as pd
from scipy.stats import pearsonr
sys.path.insert(0, "/home/dmin/ai&stem_cell/cnn_AI&stemcell")
from train_cnn import grouped_split

ROOT = "/home/dmin/ai&stem_cell"
sb = lambda r: 2 * r / (1 + r)
df = pd.read_csv(f"{ROOT}/run/rev2/master.csv", low_memory=False)
has = df.cd43.notna().values & df.cd43_br1.notna().values & df.cd43_br2.notna().values
cl = pd.read_csv(f"{ROOT}/run/protein_clusters.tsv", sep="\t", header=None,
                 names=["rep", "member"])
cmap = dict(zip(cl["member"], cl["rep"]))
groups = np.array([cmap.get(g, g) for g in df.protein.astype(str)])

IN_STEM = sys.argv[1] if len(sys.argv) > 1 else "cd43_fixed"
OUT_STEM = sys.argv[2] if len(sys.argv) > 2 else "cd43_fixed_final"
res = pd.read_json(f"{ROOT}/run/rev2/{IN_STEM}.json")
rows = []
for seed in res.seed:
    _, _, te = grouped_split(groups, seed)
    te = te[has[te]]
    sub = df.iloc[te]
    m = sub.R1_pEF.notna() & sub.R2_pEF.notna()
    r = float(pearsonr(sub.R1_pEF[m], sub.R2_pEF[m])[0])
    rows.append(dict(seed=int(seed), n_test=len(te), n_pef_pairs=int(m.sum()),
                     r_pEF_reps=r, ceil_pEF=float(np.sqrt(sb(r)))))
    print(f"  seed {seed}: n={len(te):,} pairs={int(m.sum()):,} "
          f"r(R1,R2)={r:.4f} -> ceiling {np.sqrt(sb(r)):.4f}")
c = pd.DataFrame(rows)
out = res.merge(c[["seed", "r_pEF_reps", "ceil_pEF"]], on="seed",
                suffixes=("_old", ""))
out["pct_ceil_pEFmodel_pEF"] = out.r_pEFmodel_pEF / out.ceil_pEF
out.to_json(f"{ROOT}/run/rev2/{OUT_STEM}.json", orient="records", indent=2)
cols = ["ceil_pEF", "ceil_cd43", "r_meas_pEF_cd43", "rho_meas_pEF_cd43",
        "r_pEFmodel_pEF", "pct_ceil_pEFmodel_pEF",
        "r_pEFmodel_cd43", "pct_ceil_pEFmodel_cd43",
        "r_cd43model_cd43", "rho_cd43model_cd43", "pct_ceil_cd43model_cd43"]
print("\n=== mean over seeds ===")
print(out[cols].mean().round(4).to_string())
print("\n=== sd over seeds ===")
print(out[cols].std(ddof=1).round(4).to_string())
print(f"\nwrote {ROOT}/run/rev2/{OUT_STEM}.json")
