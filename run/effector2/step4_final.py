import pandas as pd, numpy as np
ROOT="/home/dmin/ai&stem_cell"; SP="/tmp/claude-1000/-home-dmin/aa65e27c-e221-478d-b43a-6fbd74ac2284/scratchpad"
ACT_THR = 5.7809

d  = pd.read_csv(f"{ROOT}/run/data.csv")
d["seq"] = d.sequence.str.strip().str.upper()
br = pd.read_csv(f"{SP}/cd43_bireps.csv")
m  = d.merge(br, on="seq", how="left")

pf = pd.read_csv(f"{ROOT}/run/pfam_domains_annotated.csv")
krab_prot = set(pf.loc[pf.pfam_group.eq("KRAB"),"protein"])

def show(n,label,prev=None):
    print(f"{n:>8,}  {label}" + (f"   (-{prev-n:,})" if prev is not None else "")); return n

print("=== FUNNEL: reproducible, non-KRAB dCas9 repressor effector ===")
n=show(len(m),"all tiles")
f=m[m.cd43_br1.notna()&m.cd43_br2.notna()];      n=show(len(f),"scored in both CD43 bioreps",n)
f=f[f.hit_br1&f.hit_br2];                        n=show(len(f),"repression hit in BOTH bioreps",n)
f=f[f.Hit_FLAG.eq("Hit")];                       n=show(len(f),"well expressed",n)
f=f[~(f.Avg_minCMV>=ACT_THR)];                   n=show(len(f),f"not an activator (Avg_minCMV < {ACT_THR:+.2f})",n)
f=f[~f.protein.isin(krab_prot)];                 n=show(len(f),"protein carries no KRAB domain",n)
f=f.assign(cd43=f[["cd43_br1","cd43_br2"]].mean(axis=1))

print(f"\n--> {len(f):,} tiles from {f.protein.nunique():,} proteins survive the measurement filters")

# tier A: also confirmed as a repressor in the independent rTetR retest
A = f[f.Avg_pEF_retest>0]
print(f"--> of those, {len(A):,} tiles ({A.protein.nunique()} proteins) ALSO repress in the rTetR retest (Avg_pEF_retest>0)")
print(f"    ({f.Avg_pEF_retest.notna().sum():,} of the {len(f):,} were retested at all)")

cols=["protein","tile_num","cd43","cd43_br1","cd43_br2","Avg_pEF","Avg_pEF_retest","Avg_minCMV","Avg_FLAG"]
print("\n### TIER A — reproducible in dCas9 AND confirmed in rTetR retest, ranked by CD43")
print(A.sort_values("cd43",ascending=False)[cols].to_string(index=False,float_format=lambda x:f"{x:+.2f}"))

f.sort_values("cd43",ascending=False).to_csv(f"{SP}/tierB.csv",index=False)
A.sort_values("cd43",ascending=False).to_csv(f"{SP}/tierA.csv",index=False)
for name,df in (("tierA",A),("tierB",f)):
    with open(f"{SP}/{name}.fasta","w") as fh:
        for _,r in df.sort_values("cd43",ascending=False).iterrows():
            fh.write(f">{r.protein}_t{r.tile_num}\n{r.seq}\n")
print(f"\nwrote {SP}/tierA.fasta ({len(A)}) and {SP}/tierB.fasta ({len(f)})")
