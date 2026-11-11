"""Complex Portal, filtered to complexes where the candidate is an ACTUAL
participant. Also flags co-membership with the KRAB route (TRIM28/SETDB1/HP1)."""
import urllib.request, json, time, pandas as pd
CAND=[l.strip() for l in open("cand30.txt") if l.strip()]
CAND+=[l.strip() for l in open("cand_noDBD.txt") if l.strip()]
CAND=list(dict.fromkeys(CAND))
KRAB={"TRIM28","SETDB1","CBX1","CBX3","CBX5","ATF7IP"}
rows=[]
for g in CAND:
    try:
        d=json.load(urllib.request.urlopen(
            f"https://www.ebi.ac.uk/intact/complex-ws/search/{g}?format=json&species=9606",timeout=30))
    except Exception as ex:
        rows.append(dict(protein=g,n_member=-1,complexes=f"ERR {ex}",krab_comember="?")); continue
    mine=[]; kr=set()
    for e in d.get("elements",[]):
        names={i.get("name") for i in e.get("interactors",[])}
        if g in names:
            mine.append(e["complexName"]); kr |= (names & KRAB)
    rows.append(dict(protein=g, n_member=len(mine),
                     complexes=" | ".join(mine[:4])[:150],
                     krab_comember=",".join(sorted(kr)) or "-"))
    time.sleep(0.15)
R=pd.DataFrame(rows).sort_values("n_member",ascending=False)
R.to_csv("complexportal_member.csv",index=False)
print(f"{(R.n_member>0).sum()} of {len(R)} are a curated participant in >=1 complex\n")
pd.set_option("display.width",200)
print(R[R.n_member>0].to_string(index=False))
print("\nNOT a participant in any curated complex:", " ".join(R.loc[R.n_member==0,'protein']))
