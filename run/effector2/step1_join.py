"""Join the rTetR tiling screen (data.csv) to the dCas9 tiling screen (S2),
and reproduce the CD43 hit threshold from Random_control."""
import pandas as pd, numpy as np

ROOT = "/home/dmin/ai&stem_cell"
d  = pd.read_csv(f"{ROOT}/run/data.csv")
s2 = pd.read_csv(f"{ROOT}/run/tycko/s2_scores.csv")

print(f"data.csv tiles      : {len(d):,}")
print(f"S2 elements         : {len(s2):,}")

# --- threshold: 2 SD above median of Random_control in the SAME screen ---
rc = s2[s2.Sublib == "Random_control"]
for scr in ("cd43", "cd2"):
    v = rc[scr].dropna()
    thr = v.median() + 2*v.std(ddof=1)
    print(f"{scr}: Random_control n={len(v)} median={v.median():.4f} "
          f"sd={v.std(ddof=1):.4f} -> thr={thr:+.4f}")
THR_CD43 = rc["cd43"].dropna().median() + 2*rc["cd43"].dropna().std(ddof=1)

# --- join on exact 80-aa sequence ---
d["seq"] = d.sequence.str.strip().str.upper()
s2["seq"] = s2["AA sequence"].astype(str).str.strip().str.upper()

lib = s2[s2.Sublib.isin(["TFTiles","EpiFactors_Main","EpiFactors_Extended","GO_CRs"])]
print(f"\nS2 tiling sublibs   : {len(lib):,}")
print(f"unique seqs in data : {d.seq.nunique():,}")
print(f"unique seqs in lib  : {lib.seq.nunique():,}")

agg = lib.groupby("seq").agg(cd43=("cd43","mean"), cd2=("cd2","mean"),
                             n_elem=("cd43","size")).reset_index()
m = d.merge(agg, on="seq", how="left")
print(f"\njoined              : {m.cd43.notna().sum():,} of {len(m):,} tiles "
      f"have a CD43 score ({100*m.cd43.notna().mean():.1f}%)")

# sanity: do the two screens agree where both exist, as memory says (rho~0.19)?
both = m[m.cd43.notna() & m.Avg_pEF.notna()]
from scipy.stats import spearmanr
print(f"rho(Avg_pEF, cd43) on {len(both):,} tiles = "
      f"{spearmanr(both.Avg_pEF, both.cd43).statistic:.3f}")

m.to_parquet("/tmp/claude-1000/-home-dmin/aa65e27c-e221-478d-b43a-6fbd74ac2284/scratchpad/tiles_joined.parquet")
print(f"\nTHR_CD43 = {THR_CD43:+.4f}")
