"""Independent check: does each candidate appear in an EBI Complex Portal
curated complex, and which one?"""
import urllib.request, json, time, pandas as pd
CAND=[l.strip() for l in open("cand30.txt") if l.strip()]
CAND+=[l.strip() for l in open("cand_noDBD.txt") if l.strip()]
CAND=list(dict.fromkeys(CAND))
rows=[]
for g in CAND:
    try:
        u=f"https://www.ebi.ac.uk/intact/complex-ws/search/{g}?format=json&species=9606"
        d=json.load(urllib.request.urlopen(u,timeout=30))
        names=[e.get("complexName","") for e in d.get("elements",[])]
        rows.append(dict(protein=g,n=d.get("totalNumberOfResults",0),
                         complexes="; ".join(names[:6])))
    except Exception as ex:
        rows.append(dict(protein=g,n=-1,complexes=f"ERR {ex}"))
    time.sleep(0.15)
R=pd.DataFrame(rows)
R.to_csv("complexportal.csv",index=False)
print(f"{(R.n>0).sum()} of {len(R)} candidates appear in a curated complex\n")
print(R[R.n>0].to_string(index=False))
print("\nno curated complex:", " ".join(R.loc[R.n==0,'protein']))
