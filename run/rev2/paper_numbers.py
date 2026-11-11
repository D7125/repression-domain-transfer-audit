"""Every figure that goes into the revised text, computed once, in one place."""
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from scipy.stats import fisher_exact, chi2

ROOT = "/home/dmin/ai&stem_cell"
THR = 0.6296596656141653
d = pd.read_csv(f"{ROOT}/run/tycko/domains_cd43.csv")
d = d[d.cd43_max.notna()].copy()
d["hit_max"] = (d.cd43_max > THR).astype(int)
d["hit_fix"] = (d.cd43_first > THR).astype(int)
d["robust"] = d.pgk_hit.astype(int)
d["bifun"] = d.mincmv_hit.astype(int)
d["logn"] = np.log(d.n_tiles)
print(f"domains with BOTH measurements: {len(d):,}  ({d.protein.nunique():,} proteins)")

print("\n--- 3.9 robustness axis ---")
for out in ("hit_max", "hit_fix"):
    k1 = d[out][d.robust == 1]; k0 = d[out][d.robust == 0]
    orv, p = fisher_exact([[int(k1.sum()), len(k1)-int(k1.sum())],
                           [int(k0.sum()), len(k0)-int(k0.sum())]], alternative="greater")
    print(f"  {out}: robust {int(k1.sum())}/{len(k1)} = {k1.mean()*100:.1f}%  "
          f"other {int(k0.sum())}/{len(k0)} = {k0.mean()*100:.1f}%  "
          f"crude OR {orv:.3f}  p {p:.3g}")
    b = smf.logit(f"{out} ~ max_pEF + logn", d).fit(disp=0)
    f = smf.logit(f"{out} ~ robust + max_pEF + logn", d).fit(disp=0)
    lr = 2*(f.llf-b.llf)
    ci = np.exp(f.conf_int().loc["robust"])
    g = smf.gee(f"{out} ~ robust + max_pEF + logn", "protein", d,
                family=sm.families.Binomial()).fit()
    gci = np.exp(g.conf_int().loc["robust"])
    print(f"      adjusted OR {np.exp(f.params['robust']):.3f} "
          f"[{ci[0]:.3f}-{ci[1]:.3f}]  LRT chi2(1) {lr:.3f} p {chi2.sf(lr,1):.4f}")
    print(f"      GEE by protein OR {np.exp(g.params['robust']):.3f} "
          f"[{gci[0]:.3f}-{gci[1]:.3f}]  p {g.pvalues['robust']:.4f}")

print("\n--- 3.10 bifunctional domains ---")
for out in ("hit_max", "hit_fix"):
    k1 = d[out][d.bifun == 1]; k0 = d[out][d.bifun == 0]
    orv, p = fisher_exact([[int(k1.sum()), len(k1)-int(k1.sum())],
                           [int(k0.sum()), len(k0)-int(k0.sum())]], alternative="less")
    print(f"  {out}: bifunctional {int(k1.sum())}/{len(k1)} = {k1.mean()*100:.2f}%  "
          f"other {int(k0.sum())}/{len(k0)} = {k0.mean()*100:.1f}%  "
          f"OR {orv:.4f}  p {p:.3g}")

print("\n--- AUC table of 3.9, both outcomes ---")
from sklearn.metrics import roc_auc_score
for out in ("hit_max", "hit_fix"):
    y = d[out]
    print(f"  {out}: logn {roc_auc_score(y, d.logn):.4f} | "
          f"max_pEF {roc_auc_score(y, d.max_pEF):.4f} | "
          f"robust {roc_auc_score(y, d.robust):.4f}")
    m2 = smf.logit(f"{out} ~ max_pEF + logn", d).fit(disp=0)
    m3 = smf.logit(f"{out} ~ max_pEF + logn + robust", d).fit(disp=0)
    print(f"      strength+length {roc_auc_score(y, m2.predict()):.4f} -> "
          f"+robust {roc_auc_score(y, m3.predict()):.4f} "
          f"({roc_auc_score(y, m3.predict())-roc_auc_score(y, m2.predict()):+.4f})")

print("\n--- 3.6 named-group coverage ---")
ann = pd.read_csv(f"{ROOT}/run/pfam_domains_annotated.csv")
print(f"  domains with any Pfam annotation : {ann.domain_id.nunique():,}")
print(f"  domains in one of the 14 groups  : "
      f"{ann[ann.pfam_group.notna()].domain_id.nunique():,}")
