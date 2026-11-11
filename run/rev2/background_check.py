"""
B7: the motif background pools the same two states that 3.4 proves must not be
pooled -- so measure whether it matters here instead of arguing that it does not.

2.8/3.1 compute the motif background on "96,682 non-hit tiles", and
103,248 - 6,566 = 96,682, i.e. non-hit = "not a confirmed Hit", which merges
the 2,156 measured `Not Hit` tiles with 94,526 tiles that never entered the
confirmation screen. Section 3.4 spends itself showing that exact merge
invents a result. The first reviewer will use 3.4 against 3.1.

The defence is that 3.1 measures amino acid COMPOSITION, not a hit rate, and
composition does not depend on whether a tile was queued for confirmation.
That is testable: recompute every motif frequency on the measured negatives
alone and see whether the ratios move.
"""
import re
import numpy as np, pandas as pd

ROOT = "/home/dmin/ai&stem_cell"
MOTIFS = {
    "TLE WRPW": r"WRPW",
    "HP1 P.V.L": r"P.V.L",
    "CtBP broad P.[DENS][LM].": r"P.[DENS][LM].",
    "CtBP refined PL[DN]L.": r"PL[DN]L.",
    "SUMO [VILMF]K.E": r"[VILMF]K.E",
    "SUMO rev E.K[VILMF]": r"E.K[VILMF]",
}
rng = np.random.default_rng(0)


def frac(seqs, rx):
    return float(np.mean([bool(rx.search(s)) for s in seqs]))


def shuffled(seqs):
    out = []
    for s in seqs:
        a = np.frombuffer(s.encode(), dtype="S1").copy()
        rng.shuffle(a)
        out.append(b"".join(a.tolist()).decode())
    return out


m = pd.read_csv(f"{ROOT}/run/rev2/master.csv", low_memory=False)
m = m[m.sequence.str.len() == 80]
pooled = m[m.Hit_pEF != "Hit"]
measured = m[m.Hit_pEF == "Not Hit"]
untested = m[m.Hit_pEF.isna()]
print(f"pooled 'non-hit'            : {len(pooled):,}")
print(f"  of which measured Not Hit : {len(measured):,}")
print(f"  of which never tested     : {len(untested):,}")

sets = {"pooled non-hit": pooled.sequence.tolist(),
        "measured Not Hit only": measured.sequence.tolist(),
        "never tested only": untested.sequence.tolist()}
shuf = {k: shuffled(v) for k, v in sets.items()}

rows = []
print(f"\n{'motif':28s}" + "".join(f"{k:>24s}" for k in sets))
for name, pat in MOTIFS.items():
    rx = re.compile(pat)
    line = f"{name:28s}"
    rec = {"motif": name}
    for k in sets:
        o, s = frac(sets[k], rx), frac(shuf[k], rx)
        line += f"   {o*100:7.2f}% /{s*100:6.2f}% {o/s if s else np.nan:5.2f}x"
        rec[f"{k}_real"] = o; rec[f"{k}_shuf"] = s
        rec[f"{k}_ratio"] = o / s if s else np.nan
    print(line)
    rows.append(rec)
out = pd.DataFrame(rows)
out.to_csv(f"{ROOT}/run/rev2/background_check.csv", index=False)

d = (out["pooled non-hit_ratio"] - out["measured Not Hit only_ratio"]).abs()
print(f"\nlargest |ratio difference| between the pooled background and the "
      f"measured negatives: {d.max():.3f}")
print("columns are: real% / shuffled% ratio")
print(f"wrote {ROOT}/run/rev2/background_check.csv")
