import pandas as pd, numpy as np, re
SP="/tmp/claude-1000/-home-dmin/aa65e27c-e221-478d-b43a-6fbd74ac2284/scratchpad"
h=pd.read_csv(f"{SP}/all_dbd2_hits.csv")

# families that actually hit, curated: does the family bind DNA itself?
NOT_DBD=[r"zf-C3HC4.*",r"zf-C3HC.*",r"PHD.*",r"zf-TAZ",r"zf-FCS",r"zf-MYND",r"zf-MYM.*",
         r"53-BP1_Tudor",r"ADNP_N",r"PRDM2_PR",r"PRDM4_PR",r".*_PR$",r"zf-RING.*",r"zf-UBR",
         r"KRAB",r"SCAN",r"zf-C2HC5",r"Tudor.*",r"ASPP2-like_RA",r"zf-HC5HC2H.*",r"zf-UDP"]
def isdbd(n):
    return not any(re.fullmatch(p,n,re.I) for p in NOT_DBD)
h["dbd"]=h.pfam.map(isdbd)
dbd=h[h.dbd]; krab_prot=set(h.loc[h.pfam.str.contains("KRAB",case=False),"protein"])
print(f"DBD domains: {len(dbd):,} in {dbd.protein.nunique():,} proteins | "
      f"KRAB proteins in library: {len(krab_prot)}")
print("excluded-as-not-DBD families that hit:",
      sorted(set(h.loc[~h.dbd,'pfam'])))

print("\n=== FINAL FILTER ===")
out={}
for tier in ("tierA","tierB"):
    t=pd.read_csv(f"{SP}/{tier}.csv")
    n0=len(t)
    overlap_krab = t.protein.isin(krab_prot).sum()
    t=t[~t.protein.isin(krab_prot)]
    idx={p:g[["env_from","env_to"]].values for p,g in dbd.groupby("protein")}
    def ov(r):
        a=idx.get(r.protein)
        if a is None: return False
        return bool(((a[:,0]<=r.tile_end)&(r.tile_start<=a[:,1])).any())
    t["hits_dbd"]=t.apply(ov,axis=1)
    clean=t[~t.hits_dbd].copy()
    fam={p:",".join(sorted(set(g.pfam))) for p,g in dbd.groupby("protein")}
    clean["prot_dbd_family"]=clean.protein.map(fam).fillna("-")
    print(f"\n{tier}: {n0:,} tiles | KRAB proteins removed here: {overlap_krab} | "
          f"overlap a DBD: {t.hits_dbd.sum():,} -> FINAL {len(clean):,} tiles, "
          f"{clean.protein.nunique()} proteins")
    clean=clean.sort_values("cd43",ascending=False)
    clean.to_csv(f"{SP}/{tier}_V2.csv",index=False)
    out[tier]=clean

A=out["tierA"]
cols=["protein","tile_num","tile_start","tile_end","cd43","cd43_br1","cd43_br2",
      "Avg_pEF","Avg_pEF_retest","Avg_FLAG","prot_dbd_family"]
print("\n### TIER A FINAL — top 30 (reproducible in both dCas9 bioreps, confirmed in rTetR"
      " retest, well expressed, non-activator, non-KRAB, tile avoids the protein's own DBD)")
print(A[cols].head(30).to_string(index=False,float_format=lambda x:f"{x:+.2f}"))

print("\n### one best tile per protein, top 25")
best=A.loc[A.groupby("protein").cd43.idxmax()].sort_values("cd43",ascending=False)
print(best[["protein","tile_num","tile_start","tile_end","cd43","Avg_pEF","Avg_pEF_retest"]]
      .head(25).to_string(index=False,float_format=lambda x:f"{x:+.2f}"))
best.to_csv(f"{SP}/tierA_best_per_protein.csv",index=False)
with open(f"{SP}/tierA_V2.fasta","w") as fh:
    for _,r in A.iterrows(): fh.write(f">{r.protein}_t{int(r.tile_num)}_cd43{r.cd43:.2f}\n{r.seq}\n")
print(f"\nwrote {SP}/tierA_V2.csv , tierB_V2.csv , tierA_V2.fasta")
