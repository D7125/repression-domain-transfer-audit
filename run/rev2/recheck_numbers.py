"""B6: recompute every number the manuscript reports in two conflicting forms."""
import numpy as np, pandas as pd
from scipy.stats import fisher_exact

ROOT = "/home/dmin/ai&stem_cell"
dc = pd.read_csv(f"{ROOT}/run/tycko/domains_cd43.csv")
THR = 0.6296596656141653

print("="*70); print("(e) how many domains carry BOTH measurements?")
print(f"  total domains in the tiling-library set : {len(dc):,}")
for c in ("cd43_max", "cd43_first"):
    print(f"  with {c:11s}: {int(dc[c].notna().sum()):,}")
both = dc[dc.cd43_max.notna()].copy()
both["cd43_hit_max"] = both.cd43_max > THR
both["cd43_hit_first"] = both.cd43_first > THR
rob = both.pgk_hit.astype(bool)
print(f"  robust (pgk_hit) among them            : {int(rob.sum()):,}")
print(f"  non-robust among them                  : {int((~rob).sum()):,}")

print("\n(3.9) robustness axis vs CD43, outcome = max over matched tiles")
k1, n1 = int(both.cd43_hit_max[rob].sum()), int(rob.sum())
k0, n0 = int(both.cd43_hit_max[~rob].sum()), int((~rob).sum())
orv, p = fisher_exact([[k1, n1-k1], [k0, n0-k0]], alternative="greater")
print(f"  robust     {k1}/{n1} = {k1/n1*100:.1f}%")
print(f"  not robust {k0}/{n0} = {k0/n0*100:.1f}%")
print(f"  crude OR {orv:.3f}  Fisher one-sided p = {p:.3g}")
print(f"  base rate (max outcome)   {int(both.cd43_hit_max.sum())}/{len(both)}"
      f" = {both.cd43_hit_max.mean()*100:.1f}%")
print(f"  base rate (fixed tile)    {int(both.cd43_hit_first.sum())}/{len(both)}"
      f" = {both.cd43_hit_first.mean()*100:.1f}%")

print("\n(3.10) bifunctional domains vs CD43")
bi = both.mincmv_hit.astype(bool)
k1, n1 = int(both.cd43_hit_max[bi].sum()), int(bi.sum())
k0, n0 = int(both.cd43_hit_max[~bi].sum()), int((~bi).sum())
orv, p = fisher_exact([[k1, n1-k1], [k0, n0-k0]], alternative="less")
print(f"  bifunctional {k1}/{n1} = {k1/n1*100:.2f}%   others {k0}/{n0} = {k0/n0*100:.1f}%")
print(f"  OR {orv:.4f}  p = {p:.3g}")
print("  and on the PGK robustness axis, all 3,001 domains:")
allbi = dc.mincmv_hit.astype(bool)
k1, n1 = int(dc.pgk_hit[allbi].sum()), int(allbi.sum())
k0, n0 = int(dc.pgk_hit[~allbi].sum()), int((~allbi).sum())
print(f"  bifunctional robust {k1}/{n1} = {k1/n1*100:.2f}%  others {k0}/{n0} = {k0/n0*100:.2f}%")

print("\n" + "="*70); print("(a) domains robust in >=5 of the 9 repression screens (Table S1)")
sc = pd.read_csv(f"{ROOT}/run/tycko/s1_scores.csv")
thr = pd.read_csv(f"{ROOT}/run/tycko/s1_thresholds.csv", index_col=0).iloc[:, 0]
screens = list(thr.index)
print(f"  repression screens used ({len(screens)}): {screens}")
hits = pd.DataFrame({s: sc[s] > thr[s] for s in screens})
n_hit = hits.sum(axis=1)
scored = sc[["label", "Category", "Gene entry name", "Domain"]].copy()
scored["n_screens_hit"] = n_hit
scored["n_screens_scored"] = sc[screens].notna().sum(axis=1)
rob5 = scored[scored.n_screens_hit >= 5]
print(f"  total S1 elements            : {len(sc):,}")
is_krab = rob5.Domain.astype(str).str.contains("KRAB", case=False, na=False)
print(f"  robust in >=5/9 screens      : {len(rob5):,}")
print(f"     of which KRAB-annotated   : {int(is_krab.sum()):,}"
      f"  ({is_krab.mean()*100:.1f}%)")
print(f"  all KRAB elements in S1      : "
      f"{int(sc.Domain.astype(str).str.contains('KRAB', case=False, na=False).sum()):,}")
krab_all = sc.Domain.astype(str).str.contains("KRAB", case=False, na=False)
print(f"  KRAB robust >=5/9            : {int((krab_all & (n_hit>=5)).sum()):,}"
      f" / {int(krab_all.sum()):,}")
print(f"  excluding Random/DMD controls: "
      f"{int((scored.n_screens_hit>=5)[~sc.Category.astype(str).str.contains('control', case=False, na=False)].sum()):,}")
