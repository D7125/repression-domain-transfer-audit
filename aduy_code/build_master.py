"""
Build the one tile-level table every revision analysis reads.

Why: data.csv carries the rTetR/pEF axis only. Three columns that the revision
needs live elsewhere --
  R1_pEF_batchRetest, R2_pEF_batchRetest   (xlsx cols 50/51) -> MEASURED retest
                                            ceiling, instead of the derived one
  cd43 per biological replicate            (S2 long form)    -> CD43 ceiling and
                                            the both-bioreps hit rule
and without them the CD43 arm cannot be scored against its own noise ceiling.

Row order is data.csv's, which is the row order of the ESM memmap, so the
output can be indexed with the same integer indices the training scripts use.

Usage:
  python build_master.py            # writes run/rev2/master.parquet
"""
import json
import numpy as np
import pandas as pd
from openpyxl import load_workbook

ROOT = "/home/dmin/ai&stem_cell"
XLSX = f"{ROOT}/data/41586_2023_5906_MOESM5_ESM.xlsx"
OUT = f"{ROOT}/run/rev2/master.csv"

# 0-based positions in "CRTF Tiles"; verified against the header in this file's
# commit message. predata_v2.py keeps rows with Avg_pEF and Protein Sequence.
IDX = {0: "protein", 1: "tile_num", 27: "_avg_pef", 50: "R1_pEF_retest",
       51: "R2_pEF_retest", 56: "_seq"}
EXPECT = {27: "Avg_pEF", 50: "R1_pEF_batchRetest", 51: "R2_pEF_batchRetest",
          56: "Protein Sequence"}


def read_retest():
    wb = load_workbook(XLSX, read_only=True)
    it = wb["CRTF Tiles"].iter_rows(values_only=True)
    header = next(it)
    for i, name in EXPECT.items():
        if str(header[i]).strip() != name:
            raise SystemExit(f"col {i} is {header[i]!r}, expected {name!r}")
    rows = []
    for n, r in enumerate(it, 1):
        if r[27] is None or r[56] is None:
            continue
        rows.append([r[k] for k in IDX])
        if n % 20000 == 0:
            print(f"  read {n:,} sheet rows", flush=True)
    df = pd.DataFrame(rows, columns=list(IDX.values()))
    for c in ("tile_num", "_avg_pef", "R1_pEF_retest", "R2_pEF_retest"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def cd43_bioreps():
    """cd43 score per biological replicate, keyed by exact 80-aa sequence."""
    sc = pd.read_csv(f"{ROOT}/run/tycko/s2_scores.csv")
    dat = pd.read_csv(f"{ROOT}/run/tycko/s2_data.csv")
    br = (dat[dat["unique sample"].isin(["216CRTF_CD43_BR1", "216CRTF_CD43_BR2"])]
          .pivot_table(index="label", columns="unique sample", values="rho")
          .rename(columns={"216CRTF_CD43_BR1": "cd43_br1",
                           "216CRTF_CD43_BR2": "cd43_br2"}))
    sc = sc.merge(br, left_on="label", right_index=True, how="left")
    lib = sc[sc.Sublib.isin(["TFTiles", "EpiFactors_Main",
                             "EpiFactors_Extended", "GO_CRs"])].copy()
    lib["seq"] = lib["AA sequence"].astype(str).str.strip().str.upper()
    # a handful of sequences appear in more than one sublibrary element
    agg = lib.groupby("seq").agg(cd43=("cd43", "mean"),
                                 cd43_br1=("cd43_br1", "mean"),
                                 cd43_br2=("cd43_br2", "mean"),
                                 n_elem=("cd43", "size")).reset_index()
    # the published convention: 2 SD above the Random_control median, per screen
    rc = sc[sc.Sublib == "Random_control"]
    thr = {}
    for k, col in (("avg", "cd43"), ("br1", "cd43_br1"), ("br2", "cd43_br2")):
        v = rc[col].dropna()
        thr[k] = float(v.median() + 2 * v.std(ddof=1))
    return agg, thr


def main():
    d = pd.read_csv(f"{ROOT}/run/data.csv", low_memory=False)
    print(f"data.csv {len(d):,} tiles", flush=True)

    rt = read_retest()
    print(f"xlsx scored rows {len(rt):,}", flush=True)
    if len(rt) != len(d):
        raise SystemExit("row counts differ -- the data.csv filter changed")
    same = (rt.protein.values == d.protein.values).mean()
    print(f"protein column agrees on {same*100:.3f}% of rows (must be 100)")
    if same < 1.0:
        raise SystemExit("row order differs; join on (protein, tile_num) instead")
    d["R1_pEF_retest"] = rt.R1_pEF_retest.values
    d["R2_pEF_retest"] = rt.R2_pEF_retest.values

    agg, thr = cd43_bioreps()
    d["seq"] = d.sequence.str.strip().str.upper()
    d = d.merge(agg, on="seq", how="left")
    print(f"\nCD43 thresholds (2 SD over Random_control median): {thr}")
    print(f"tiles with a CD43 average score : {int(d.cd43.notna().sum()):,}")
    both = d.cd43_br1.notna() & d.cd43_br2.notna()
    print(f"tiles scored in BOTH bioreps    : {int(both.sum()):,}")
    print(f"tiles with retest replicates    : "
          f"{int((d.R1_pEF_retest.notna() & d.R2_pEF_retest.notna()).sum()):,}")

    d["cd43_hit_both"] = (both & (d.cd43_br1 > thr["br1"])
                          & (d.cd43_br2 > thr["br2"]))
    d.to_csv(OUT, index=False)
    json.dump(thr, open(f"{ROOT}/run/rev2/cd43_thresholds.json", "w"), indent=2)
    print(f"\nwrote {OUT}  ({len(d):,} rows, {len(d.columns)} cols)")


if __name__ == "__main__":
    main()
