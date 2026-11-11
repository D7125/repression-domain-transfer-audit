import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
ROOT="/home/dmin/ai&stem_cell"

# --- per-biorep CD43, and a per-biorep hit threshold from Random_control ---
s2d = pd.read_csv(f"{ROOT}/run/tycko/s2_data.csv")
s2e = pd.read_csv(f"{ROOT}/run/tycko/s2_elem.csv")
print("s2_elem cols:", list(s2e.columns)[:12])
cd43 = s2d[s2d["unique sample"].str.contains("CD43")].copy()
cd43["br"] = cd43["unique sample"].str[-1]
print("\nCD43 rows per biorep:", dict(cd43.br.value_counts()))

key = s2e[["label","Sublib","Gene","Tile number","AA sequence"]].drop_duplicates("label")
cd43 = cd43.merge(key, on="label", how="left")
rc = cd43[cd43.Sublib=="Random_control"]
thr = {}
for b,g in rc.groupby("br"):
    v=g.rho.dropna(); thr[b]=v.median()+2*v.std(ddof=1)
    print(f"  BR{b}: Random_control n={len(v)} median={v.median():+.4f} sd={v.std(ddof=1):.4f} thr={thr[b]:+.4f}")

w = cd43.pivot_table(index="label", columns="br", values="rho").rename(columns={"1":"cd43_br1","2":"cd43_br2"})
w = w.join(key.set_index("label"))
lib = w[w.Sublib.isin(["TFTiles","EpiFactors_Main","EpiFactors_Extended","GO_CRs"])].dropna(subset=["cd43_br1","cd43_br2"])
from scipy.stats import spearmanr
print(f"\nbiorep agreement on {len(lib):,} tiling elements: rho={spearmanr(lib.cd43_br1,lib.cd43_br2).statistic:.3f}")
lib = lib.assign(hit_br1=lib.cd43_br1>=thr["1"], hit_br2=lib.cd43_br2>=thr["2"])
print(f"  hit in BR1 only {(lib.hit_br1&~lib.hit_br2).sum():,} | BR2 only {(~lib.hit_br1&lib.hit_br2).sum():,} "
      f"| BOTH {(lib.hit_br1&lib.hit_br2).sum():,}")
lib["seq"]=lib["AA sequence"].str.strip().str.upper()
lib.groupby("seq")[["cd43_br1","cd43_br2","hit_br1","hit_br2"]].max().to_csv(
  "/tmp/claude-1000/-home-dmin/aa65e27c-e221-478d-b43a-6fbd74ac2284/scratchpad/cd43_bireps.csv")

# --- activation threshold on Avg_minCMV, recovered from Hit_minCMV ---
d = pd.read_csv(f"{ROOT}/run/data.csv")
sub = d[d.Hit_minCMV.notna() & d.Avg_minCMV.notna()]
y = (sub.Hit_minCMV=="Hit").values
print(f"\nHit_minCMV labelled & scored: {len(sub)} ({y.sum()} Hit)")
print(f"  AUROC(Avg_minCMV -> Hit_minCMV) = {roc_auc_score(y, sub.Avg_minCMV):.4f}")
cuts=np.quantile(sub.Avg_minCMV,np.linspace(0.001,0.999,400))
acc=[( (sub.Avg_minCMV>=c)==y ).mean() for c in cuts]
best=cuts[int(np.argmax(acc))]
print(f"  best single cut {best:+.4f} reproduces {max(acc)*100:.1f}% of labels")
print(f"  Avg_minCMV among Hit: p5={np.percentile(sub.Avg_minCMV[y],5):+.3f} median={np.median(sub.Avg_minCMV[y]):+.3f}")
print(f"  tiles in full data with Avg_minCMV >= {best:+.3f}: {(d.Avg_minCMV>=best).sum():,}")
