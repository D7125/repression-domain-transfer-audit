"""Which repressive route does each candidate connect to? Uses STRING's
EXPERIMENTAL (escore) and CURATED-DATABASE (dscore) channels only -- textmining
is excluded because the mechanism claims we are testing came from the literature."""
import urllib.request, urllib.parse, io, pandas as pd, collections

CAND = [l.strip() for l in open("cand30.txt") if l.strip()]
EXTRA = [l.strip() for l in open("cand_noDBD.txt") if l.strip()]
CAND = list(dict.fromkeys(CAND + EXTRA))

ROUTES = {
 "KRAB/KAP1":      ["TRIM28","SETDB1","CBX5","CBX1","CBX3","ZNF274","ATF7IP"],
 "SIN3A/HDAC":     ["SIN3A","SIN3B","HDAC1","HDAC2","SAP30","SAP30L","SUDS3","ARID4A","ARID4B","SAP130","SAP18"],
 "NuRD":           ["CHD3","CHD4","MTA1","MTA2","MTA3","GATAD2A","GATAD2B","MBD2","MBD3","RBBP4","RBBP7"],
 "CoREST/LSD1":    ["RCOR1","RCOR2","KDM1A","PHF21A","HMG20B"],
 "NCoR/SMRT":      ["NCOR1","NCOR2","HDAC3","TBL1X","TBL1XR1","GPS2"],
 "PRC1":           ["RNF2","RING1","PCGF1","PCGF2","PCGF3","PCGF5","PCGF6","RYBP","YAF2","CBX7","CBX8"],
 "PRC2":           ["EZH2","SUZ12","EED","RBBP4"],
 "G9a/GLP":        ["EHMT1","EHMT2","WIZ"],
 "CtBP":           ["CTBP1","CTBP2"],
 "PRC1.6/E2F6":    ["E2F6","MGA","MAX","L3MBTL2","TFDP1","CBX3"],
}
MARK = sorted({g for v in ROUTES.values() for g in v})
ids = list(dict.fromkeys(CAND + MARK))
print(f"{len(CAND)} candidates + {len(MARK)} marker genes = {len(ids)} identifiers")

url = "https://string-db.org/api/tsv/network"
data = urllib.parse.urlencode({"identifiers":"\r".join(ids),"species":"9606",
                               "caller_identity":"local_analysis"}).encode()
txt = urllib.request.urlopen(urllib.request.Request(url,data=data),timeout=120).read().decode()
e = pd.read_csv(io.StringIO(txt), sep="\t")
print(f"STRING returned {len(e):,} edges; columns: {list(e.columns)}")
e.to_csv("string_edges_raw.csv", index=False)

# keep only physical/curated evidence
e["hard"] = e[["escore","dscore"]].max(axis=1)
E = e[e.hard >= 0.40]
print(f"edges with experimental or curated score >= 0.40: {len(E):,}")

g2r = {g:r for r,v in ROUTES.items() for g in v}
rows=[]
for c in CAND:
    sub = E[(E.preferredName_A==c)|(E.preferredName_B==c)]
    best = collections.defaultdict(float); partners=collections.defaultdict(list)
    for _,r in sub.iterrows():
        other = r.preferredName_B if r.preferredName_A==c else r.preferredName_A
        if other in g2r:
            rt=g2r[other]
            best[rt]=max(best[rt],r.hard); partners[rt].append(f"{other}:{r.hard:.2f}")
    kap = best.get("KRAB/KAP1",0.0)
    ranked = sorted([(v,k) for k,v in best.items() if k!="KRAB/KAP1"], reverse=True)
    rows.append(dict(protein=c, kap1_score=round(kap,3),
        kap1_partners=";".join(sorted(partners["KRAB/KAP1"])) or "-",
        route1=ranked[0][1] if ranked else "-", route1_score=round(ranked[0][0],3) if ranked else 0.0,
        route2=ranked[1][1] if len(ranked)>1 else "-",
        evidence=";".join(sorted(partners[ranked[0][1]])) if ranked else "-"))
R=pd.DataFrame(rows).sort_values(["kap1_score","route1_score"],ascending=[True,False])
R.to_csv("mechanism_routes.csv",index=False)
print("\n=== ROUTE ASSIGNMENT (experimental/curated evidence only) ===")
print(R.to_string(index=False))
print(f"\ncandidates with NO KAP1/SETDB1 evidence: {(R.kap1_score==0).sum()} of {len(R)}")
print(f"candidates with an orthogonal route identified: {(R.route1!='-').sum()}")
