"""Rebuild each protein on its TRUE coordinate axis from tile_start, so that
hmmscan envelope coordinates are directly comparable to tile_start/tile_end."""
import pandas as pd, numpy as np
SP="/tmp/claude-1000/-home-dmin/aa65e27c-e221-478d-b43a-6fbd74ac2284/scratchpad"
d=pd.read_csv("/home/dmin/ai&stem_cell/run/data.csv")
B=pd.read_csv(f"{SP}/tierB.csv"); prots=set(B.protein)
sub=d[d.protein.isin(prots)]

gapstat=[]
with open(f"{SP}/prot_coord.fasta","w") as fh:
    for p,g in sub.groupby("protein"):
        L=int(g.tile_end.max())
        arr=np.full(L,"X",dtype="<U1")
        for _,r in g.iterrows():
            s=str(r.sequence).strip().upper()
            a,b=int(r.tile_start)-1,int(r.tile_end)
            if len(s)==b-a: arr[a:b]=list(s)
        gap=(arr=="X").mean()
        gapstat.append((p,L,gap))
        fh.write(f">{p}\n"+"".join(arr)+"\n")
gs=pd.DataFrame(gapstat,columns=["protein","L","gap"])
print(f"{len(gs)} proteins rebuilt; length median {int(gs.L.median())}, max {gs.L.max()}")
print(f"gap fraction: median {gs.gap.median():.4f}, mean {gs.gap.mean():.4f}, "
      f"proteins with >5% gap: {(gs.gap>0.05).sum()}")
print("\nspot check vs real lengths (REST=1097, WIZ=1920, MYB=640, IKZF1=519, ZNF275=381):")
print(gs[gs.protein.isin(['REST','WIZ','MYB','IKZF1','ZNF275','ZNF592','MNT','SAP25'])].to_string(index=False))
gs.to_csv(f"{SP}/prot_gapstat.csv",index=False)
