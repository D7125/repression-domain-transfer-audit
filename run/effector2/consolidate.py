"""Consolidate STRING (experimental/curated edges) + Complex Portal (curated
named complexes) into one verdict per candidate."""
import urllib.request, json, time, pandas as pd, re
CAND=[l.strip() for l in open("cand30.txt") if l.strip()]
CAND+=[l.strip() for l in open("cand_noDBD.txt") if l.strip()]
CAND=list(dict.fromkeys(CAND))
KRAB={"TRIM28","SETDB1","CBX1","CBX3","CBX5","ATF7IP"}
# a NAMED curated complex has a prose name; AP-MS blobs are "A:B:C:..." lists
named=lambda s: ":" not in s

rows=[]
for g in CAND:
    try:
        d=json.load(urllib.request.urlopen(
          f"https://www.ebi.ac.uk/intact/complex-ws/search/{g}?format=json&species=9606",timeout=30))
    except Exception: d={"elements":[]}
    nm=[]; kr=set(); krnamed=set()
    for e in d.get("elements",[]):
        ns={i.get("name") for i in e.get("interactors",[])}
        if g not in ns: continue
        if e.get("predictedComplex"): continue
        hits=ns & KRAB
        kr |= hits
        if named(e["complexName"]):
            nm.append(e["complexName"])
            if hits: krnamed |= hits
    rows.append(dict(protein=g, named_complexes=" | ".join(sorted(set(nm))[:3])[:120] or "-",
                     cp_krab=",".join(sorted(krnamed)) or "-"))
    time.sleep(0.15)
cp=pd.DataFrame(rows)

st=pd.read_csv("mechanism_routes.csv")
M=st.merge(cp,on="protein",how="left")

ACTIVATING=re.compile(r"activation|acetyltransferase|SET1|H3K4|Myb-MuvB",re.I)
def verdict(r):
    kap = (r.kap1_score>=0.40) or (r.cp_krab!="-")
    if kap: return "CAUTION: KRAB-route contact"
    if r.named_complexes!="-" and ACTIVATING.search(r.named_complexes): return "WRONG DIRECTION: activating complex"
    if r.route1!="-" and r.route1_score>=0.80: return "PASS: strong orthogonal route"
    if r.route1!="-": return "PASS(weak): orthogonal route, modest score"
    return "UNVERIFIED: no mechanism evidence"
M["verdict"]=M.apply(verdict,axis=1)

A=pd.read_csv("candidates_tierA.csv")
best=A.loc[A.groupby("protein").cd43.idxmax()][["protein","tile_start","tile_end","cd43","Avg_pEF_retest","prot_dbd_family"]]
M=M.merge(best,on="protein",how="left").sort_values(["verdict","cd43"],ascending=[True,False])
M["noDBD"]=M.prot_dbd_family=="-"
M.to_csv("mechanism_verdict.csv",index=False)

pd.set_option("display.width",250); pd.set_option("display.max_colwidth",60)
for v in sorted(M.verdict.unique()):
    s=M[M.verdict==v]
    print(f"\n########## {v}  (n={len(s)})")
    print(s[["protein","tile_start","tile_end","cd43","Avg_pEF_retest","noDBD","route1","route1_score","named_complexes"]]
          .to_string(index=False,float_format=lambda x:f"{x:+.2f}"))
print("\n=== counts ==="); print(M.verdict.value_counts().to_string())
