"""
B2: read every headline rank correlation against its own noise ceiling.

Section 2.2 states the rule -- "every correlation coefficient must be read as
a fraction of its ceiling, not of 1.0" -- and then 4.3 reports rho(rTetR,
dCas9) raw. Correcting for attenuation does not rescue rank preservation, but
it blocks the obvious rebuttal ("your rho is low only because the CD43 screen
is noisy, and you said so yourself"), and it turns out to change one claim:
the tile-level and Pfam-level figures agree once corrected.

Everything here is recomputed from the source tables, including the biorep
correlations that the correction needs.
"""
import json
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ROOT = "/home/dmin/ai&stem_cell"
sb = lambda r: 2 * r / (1 + r)          # Spearman-Brown, 2-replicate mean


def rho(a, b):
    m = pd.notna(a) & pd.notna(b)
    return float(spearmanr(a[m], b[m]).statistic), int(m.sum())


print("=" * 74)
print("Pfam library (Table S1): same pEF reporter, same K562, rTetR vs dCas9")
print("=" * 74)
sc = pd.read_csv(f"{ROOT}/run/tycko/s1_scores.csv")
dat = pd.read_csv(f"{ROOT}/run/tycko/s1_data.csv")
br = dat.pivot_table(index="label", columns="unique sample", values="rho")
sc = sc.merge(br, left_on="label", right_index=True, how="left")
real = sc[~sc.Category.astype(str).str.contains("control", case=False, na=False)]

r_tetr, n1 = rho(real["NucRepr_D5_R1"], real["NucRepr_D5_R2"])
r_cas9, n2 = rho(real["216Pfam_TetOsg1_JT39_BR1"], real["216Pfam_TetOsg1_JT39_BR2"])
r_cross, n3 = rho(real["K562 rTetR pEF D5"], real["K562 dCas9 pEF"])
print(f"  biorep rho, rTetR pEF  : {r_tetr:.4f}  (n = {n1:,})  -> rel {sb(r_tetr):.4f}")
print(f"  biorep rho, dCas9 pEF  : {r_cas9:.4f}  (n = {n2:,})  -> rel {sb(r_cas9):.4f}")
print(f"  rho(rTetR, dCas9)      : {r_cross:.4f}  (n = {n3:,})")

krab = real.Domain.astype(str).str.contains("KRAB", case=False, na=False)
r_nokrab, n4 = rho(real["K562 rTetR pEF D5"][~krab], real["K562 dCas9 pEF"][~krab])
r_krab, n5 = rho(real["K562 rTetR pEF D5"][krab], real["K562 dCas9 pEF"][krab])
print(f"  rho, KRAB only         : {r_krab:.4f}  (n = {n5:,})")
print(f"  rho, excluding KRAB    : {r_nokrab:.4f}  (n = {n4:,})")

print()
print("=" * 74)
print("Tiling library (Table S2): the unit an effector designer actually picks")
print("=" * 74)
m = pd.read_csv(f"{ROOT}/run/rev2/master.csv", low_memory=False)
t = m[m.cd43.notna() & m.cd43_br1.notna() & m.cd43_br2.notna()]
r_pef_reps, np1 = rho(t.R1_pEF, t.R2_pEF)
r_cd43_reps, np2 = rho(t.cd43_br1, t.cd43_br2)
r_tile, np3 = rho(t.Avg_pEF, t.cd43)
print(f"  biorep rho, pEF (rTetR): {r_pef_reps:.4f}  (n = {np1:,})  -> rel {sb(r_pef_reps):.4f}")
print(f"  biorep rho, CD43 (dCas9): {r_cd43_reps:.4f}  (n = {np2:,})  -> rel {sb(r_cd43_reps):.4f}")
print(f"  rho(Avg_pEF, cd43)     : {r_tile:.4f}  (n = {np3:,})")

print()
print("=" * 74)
print("corrected = raw / sqrt(rel_1 * rel_2)")
print("=" * 74)
rows = [
    ("Pfam library, all domains", r_cross, sb(r_tetr), sb(r_cas9)),
    ("Pfam library, KRAB only", r_krab, sb(r_tetr), sb(r_cas9)),
    ("Pfam library, excluding KRAB", r_nokrab, sb(r_tetr), sb(r_cas9)),
    ("Tiling library, tile level", r_tile, sb(r_pef_reps), sb(r_cd43_reps)),
]
print(f"  {'comparison':32s} {'raw':>7s} {'rel1':>7s} {'rel2':>7s} {'atten':>7s} {'corr':>7s}")
out = []
for name, r, rel1, rel2 in rows:
    att = float(np.sqrt(rel1 * rel2))
    print(f"  {name:32s} {r:7.3f} {rel1:7.3f} {rel2:7.3f} {att:7.3f} {r/att:7.3f}")
    out.append(dict(comparison=name, rho_raw=r, rel_rTetR=rel1, rel_dCas9=rel2,
                    attenuation=att, rho_corrected=r / att))
pd.DataFrame(out).to_csv(f"{ROOT}/run/rev2/attenuation.csv", index=False)
json.dump(dict(r_tetr_biorep=r_tetr, r_cas9_biorep=r_cas9, r_cross=r_cross,
               r_krab=r_krab, r_nokrab=r_nokrab, n_pfam=n3,
               r_pef_biorep_tile=r_pef_reps, r_cd43_biorep_tile=r_cd43_reps,
               r_tile=r_tile, n_tile=np3),
          open(f"{ROOT}/run/rev2/attenuation.json", "w"), indent=2)
print(f"\nwrote {ROOT}/run/rev2/attenuation.csv")
