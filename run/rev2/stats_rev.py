"""
B2 + B5 + the rest of B6: apply the paper's own standards to its own numbers.

  B2  attenuation-correct every headline rho by sqrt(rel1*rel2), since 2.2
      says coefficients must be read against their ceiling, not against 1.0
  B5  replace the seven uncorrected ORs of the mean_FLAG ladder with ONE trend
      test, BH-correct the CD43 group comparisons, and give the strength
      quintiles a trend test instead of one p
  B6  the label-set ratios that the manuscript compresses into one sentence
"""
import numpy as np, pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import fisher_exact, chi2
from statsmodels.stats.multitest import multipletests

ROOT = "/home/dmin/ai&stem_cell"
THR = 0.6296596656141653
sb = lambda r: 2*r/(1+r)

print("="*72); print("B2  attenuation-corrected rank correlations")
print("="*72)
# Pfam library, rTetR vs dCas9 on the same pEF reporter (sec 4.3)
rows = [
    ("Pfam library, rTetR vs dCas9", 0.193, 0.619, 0.800),
    ("Pfam library, excluding KRAB", 0.061, 0.619, 0.800),
]
# tile level: pEF reliability from R1/R2, CD43 from its two bioreps
m = pd.read_csv(f"{ROOT}/run/rev2/master.csv", low_memory=False)
sub = m[m.cd43.notna() & m.cd43_br1.notna() & m.cd43_br2.notna()]
from scipy.stats import spearmanr, pearsonr
rp = sub[sub.R1_pEF.notna() & sub.R2_pEF.notna()]
rho_pef = float(spearmanr(rp.R1_pEF, rp.R2_pEF).statistic)
rho_cd43 = float(spearmanr(sub.cd43_br1, sub.cd43_br2).statistic)
rho_tile = float(spearmanr(sub.Avg_pEF, sub.cd43).statistic)
print(f"  tile level, n = {len(sub):,}")
print(f"    rho(R1_pEF , R2_pEF )  = {rho_pef:.4f} -> rel {sb(rho_pef):.4f}")
print(f"    rho(CD43 BR1, BR2)     = {rho_cd43:.4f} -> rel {sb(rho_cd43):.4f}")
rows.append(("Tile level, pEF vs CD43", rho_tile, rho_pef, rho_cd43))
print()
print(f"  {'comparison':34s} {'rho raw':>8s} {'rel1':>7s} {'rel2':>7s}"
      f" {'sqrt(r1r2)':>11s} {'rho corr':>9s}")
out = []
for name, rho, rsplit1, rsplit2 in rows:
    rel1, rel2 = sb(rsplit1), sb(rsplit2)
    att = np.sqrt(rel1*rel2)
    print(f"  {name:34s} {rho:8.3f} {rel1:7.3f} {rel2:7.3f} {att:11.3f}"
          f" {rho/att:9.3f}")
    out.append(dict(comparison=name, rho_raw=rho, rel1=rel1, rel2=rel2,
                    attenuation=att, rho_corrected=rho/att))
pd.DataFrame(out).to_csv(f"{ROOT}/run/rev2/attenuation.csv", index=False)

print()
print("="*72); print("B5  trend tests replacing ladders of uncorrected p-values")
print("="*72)
dc = pd.read_csv(f"{ROOT}/run/tycko/domains_cd43.csv")
d = dc[dc.cd43_max.notna()].copy()
d["hit_max"] = (d.cd43_max > THR).astype(int)
d["hit_fix"] = (d.cd43_first > THR).astype(int)
d["robust"] = d.pgk_hit.astype(int)
d["logn"] = np.log(d.n_tiles)

print("\n[3.9] robustness x expression: one interaction term, not seven ORs")
dd = d[d.mean_FLAG.notna()].copy()
base = smf.logit("hit_max ~ robust + max_pEF + logn + mean_FLAG", dd).fit(disp=0)
full = smf.logit("hit_max ~ robust * mean_FLAG + max_pEF + logn", dd).fit(disp=0)
lr = 2*(full.llf - base.llf)
print(f"  n = {len(dd):,}")
print(f"  interaction robust x mean_FLAG  beta = {full.params['robust:mean_FLAG']:+.4f}"
      f"  (SE {full.bse['robust:mean_FLAG']:.4f})")
print(f"  LRT chi2(1) = {lr:.3f}, p = {chi2.sf(lr,1):.4f}")
gee = smf.gee("hit_max ~ robust * mean_FLAG + max_pEF + logn", "protein", dd,
              family=sm.families.Binomial()).fit()
print(f"  GEE clustered by protein: beta = {gee.params['robust:mean_FLAG']:+.4f},"
      f" p = {gee.pvalues['robust:mean_FLAG']:.4f}")
print("  the ladder it replaces (ADJUSTED OR of robust, controls as in 3.9):")
for cut in (-np.inf, -3, -2, -1, 0, 1, 2):
    ss = dd[dd.mean_FLAG > cut] if np.isfinite(cut) else dd
    if int(ss.robust.sum()) < 20:
        continue
    fit = smf.logit("hit_max ~ robust + max_pEF + logn", ss).fit(disp=0)
    print(f"    mean_FLAG > {cut:>5}: n={len(ss):5,}  adj OR "
          f"{np.exp(fit.params['robust']):.3f}  p {fit.pvalues['robust']:.3g}")

print("\n[3.9] strength quintiles: trend test instead of one p")
d["q"] = pd.qcut(d.max_pEF, 5, labels=False)
for outcome in ("hit_max", "hit_fix"):
    print(f"  outcome = {outcome}")
    for q, s in d.groupby("q"):
        k1 = s[outcome][s.robust == 1]; k0 = s[outcome][s.robust == 0]
        orv, p = fisher_exact([[int(k1.sum()), len(k1)-int(k1.sum())],
                               [int(k0.sum()), len(k0)-int(k0.sum())]])
        print(f"    Q{q+1}: robust {int(k1.sum())}/{len(k1)} = "
              f"{k1.mean()*100:.1f}%  other {int(k0.sum())}/{len(k0)} = "
              f"{k0.mean()*100:.1f}%  OR {orv:.3f}  p {p:.3f}")
b2 = smf.logit("hit_max ~ robust + q + logn", d).fit(disp=0)
f2 = smf.logit("hit_max ~ robust * q + logn", d).fit(disp=0)
print(f"  (quintiles computed on all {len(d):,} domains with both measurements)")
lr2 = 2*(f2.llf - b2.llf)
print(f"  interaction robust x quintile index: beta = "
      f"{f2.params['robust:q']:+.4f}, LRT chi2(1) = {lr2:.3f}, p = {chi2.sf(lr2,1):.4f}")

print("\n[3.6] CD43 group comparisons, BH-corrected over the 14 named groups")
ann = pd.read_csv(f"{ROOT}/run/pfam_domains_annotated.csv")
g = ann[["domain_id", "pfam_group"]].dropna().drop_duplicates("domain_id")
dg = d.merge(g, on="domain_id", how="left")
base_rate = dg.hit_fix.mean()
res = []
for grp, s in dg[dg.pfam_group.notna()].groupby("pfam_group"):
    rest = dg[dg.pfam_group != grp]
    orv, p = fisher_exact([[int(s.hit_fix.sum()), len(s)-int(s.hit_fix.sum())],
                           [int(rest.hit_fix.sum()), len(rest)-int(rest.hit_fix.sum())]])
    res.append(dict(group=grp, k=int(s.hit_fix.sum()), n=len(s),
                    frac=s.hit_fix.mean(), n_prot=s.protein.nunique(),
                    OR=orv, p=p))
r = pd.DataFrame(res).sort_values("frac", ascending=False)
r["q_BH"] = multipletests(r.p, method="fdr_bh")[1]
print(f"  base rate, fixed-tile outcome: {dg.hit_fix.mean()*100:.1f}% "
      f"({int(dg.hit_fix.sum()):,}/{len(dg):,})")
print(r.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
r.to_csv(f"{ROOT}/run/rev2/cd43_groups_bh.csv", index=False)

print()
print("="*72); print("B6  the three label sets the manuscript compresses into one clause")
print("="*72)
yes = (m.Hit_pEF == "Hit"); no_broad = (m.Hit_pEF != "Hit")
well = (m.Hit_FLAG == "Hit"); meas_no = (m.Hit_pEF == "Not Hit")
print(f"  pooled negative, expression-filtered : YES {int((yes&well).sum()):,} /"
      f" NO {int((no_broad&well).sum()):,}"
      f"  = 1 : {int((no_broad&well).sum())/int((yes&well).sum()):.1f}")
print(f"  measured negatives, NO filter        : YES {int(yes.sum()):,} /"
      f" NO {int(meas_no.sum()):,}  n = {int((yes|meas_no).sum()):,}"
      f"  positive base {yes.sum()/(yes.sum()+meas_no.sum()):.4f}")
print(f"  measured negatives, filtered on both : YES {int((yes&well).sum()):,} /"
      f" NO {int((meas_no&well).sum()):,}  n = {int(((yes|meas_no)&well).sum()):,}"
      f"  positive base {(yes&well).sum()/((yes&well).sum()+(meas_no&well).sum()):.4f}"
      f"  = {(yes&well).sum()/(meas_no&well).sum():.1f} : 1")
