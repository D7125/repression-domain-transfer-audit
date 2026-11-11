"""
Step 0: turn the Nature supplementary workbook into the data.csv the rest of
the pipeline expects.

Input : 41586_2023_5906_MOESM5_ESM.xlsx  (Supplementary Table 5, ~65 MB)
Output: data.csv, one row per SCORED tile, with columns renamed to:
            protein   (was "HGNC Symbol")   <- grouping key for every split
            sequence  (was "Protein Sequence")
            Avg_pEF   (unchanged)           <- the regression target
        plus tile coordinates, both pEF replicates, FLAG expression, and the
        three hit labels (pEF, PGK, minCMV) that the clustering analysis needs.

It also re-runs the dataset checks, so you can confirm your copy of the
workbook matches the model report before training or clustering anything.

Usage:
    pip install pandas openpyxl scipy
    python prepare_data.py --xlsx 41586_2023_5906_MOESM5_ESM.xlsx --out data.csv

Takes a few minutes: openpyxl streams 113,528 rows x 62 columns.
"""

import argparse

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from scipy.stats import pearsonr, spearmanr

# 0-based column positions in the "CRTF Tiles" sheet -> output names.
COLS = {
    0:  "protein",       # HGNC Symbol
    1:  "tile_num",
    2:  "tile_start",
    3:  "tile_end",
    11: "Avg_minCMV",    # activation score, for the bifunctional analysis
    19: "Avg_FLAG",      # expression; low expression causes false negatives
    21: "R1_pEF",
    24: "R2_pEF",
    27: "Avg_pEF",       # the target
    28: "SE_pEF",
    # The PGK screen DOES carry numeric scores, not just the binary label:
    # cols 29/32/35/36. The patch doc asked for a full 62-column scan to settle
    # this -- it is settled, so the robustness axis gets its own noise ceiling.
    29: "R1_PGK",
    32: "R2_PGK",
    35: "Avg_PGK",
    36: "SE_PGK",
    52: "Avg_pEF_retest",  # confirmation screen, 10x coverage (see note below)
    56: "sequence",      # Protein Sequence
    58: "Hit_PGK",       # second promoter -- drives the robustness analysis
    59: "Hit_pEF",       # "Hit" / "Not Hit" / blank
    60: "Hit_minCMV",    # activation screen; a hit here means bifunctional
    61: "Hit_FLAG",
}

EXPECTED = {0: "HGNC Symbol", 27: "Avg_pEF", 35: "Avg_PGK",
            52: "Avg_pEF_batchRetest", 56: "Protein Sequence",
            58: "Hit PGK", 59: "Hit pEF", 60: "Hit minCMV"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--sheet", default="CRTF Tiles")
    ap.add_argument("--out", default="data.csv")
    ap.add_argument("--keep-unscored", action="store_true")
    args = ap.parse_args()

    wb = load_workbook(args.xlsx, read_only=True)
    if args.sheet not in wb.sheetnames:
        raise SystemExit(f"sheet '{args.sheet}' not found. sheets = {wb.sheetnames}")

    it = wb[args.sheet].iter_rows(values_only=True)
    header = next(it)
    for i, name in EXPECTED.items():
        if str(header[i]).strip() != name:
            raise SystemExit(
                f"column {i} is '{header[i]}', expected '{name}'.\n"
                f"Workbook layout differs -- fix COLS before continuing.\n"
                f"Full header: {list(header)}")

    rows = []
    for n, r in enumerate(it, 1):
        rows.append([r[k] for k in COLS])
        if n % 20000 == 0:
            print(f"  read {n:,} rows", flush=True)

    df = pd.DataFrame(rows, columns=list(COLS.values()))
    for c in ["Avg_minCMV", "Avg_FLAG", "R1_pEF", "R2_pEF", "Avg_pEF", "SE_pEF",
              "R1_PGK", "R2_PGK", "Avg_PGK", "SE_PGK", "Avg_pEF_retest",
              "tile_num", "tile_start", "tile_end"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    print("\n" + "=" * 64)
    print("DATASET CHECKS")
    print("=" * 64)
    print(f"  total tiles                  {len(df):>10,}   expect 113,528")
    print(f"  tiles with an Avg_pEF score  {int(df.Avg_pEF.notna().sum()):>10,}   expect 103,248")
    print(f"  distinct proteins            {df.protein.nunique():>10,}   expect  2,045")
    print("     (the paper says 2,047 -- the tiles sheet holds 2,045)")

    scored = df[df.Avg_pEF.notna()]
    hits_all = int((df.Hit_pEF == "Hit").sum())
    hits_scored = int((scored.Hit_pEF == "Hit").sum())
    print(f"\n  'Hit' rows, whole sheet      {hits_all:>10,}   paper reports 6,755")
    print(f"  'Hit' rows WITH a score      {hits_scored:>10,}   report uses  6,566")
    print(f"     -> {hits_all - hits_scored} confirmed hits carry no Avg_pEF value.")
    print("        That is the whole 6,755 vs 6,566 discrepancy. Not an error.")
    print(f"\n  pEF hits also hit at PGK     {int(((df.Hit_pEF=='Hit') & (df.Hit_PGK=='Hit')).sum()):>10,}")
    print(f"  pEF hits also hit at minCMV  {int(((df.Hit_pEF=='Hit') & (df.Hit_minCMV=='Hit')).sum()):>10,}   (bifunctional)")
    print(f"\n  Avg_pEF  min {scored.Avg_pEF.min():.2f}  max {scored.Avg_pEF.max():.2f}  "
          f"mean {scored.Avg_pEF.mean():.3f}")

    def ceiling(a, b, label):
        """Spearman-Brown ceiling for the mean of two replicates."""
        m = df[df[a].notna() & df[b].notna()]
        if not len(m):
            print(f"\n  {label}: no tile has both replicates -- no ceiling")
            return
        r = pearsonr(m[a], m[b])[0]
        rho = spearmanr(m[a], m[b])[0]
        rel = 2 * r / (1 + r)
        print(f"\n  {label}: replicate agreement on {len(m):,} tiles")
        print(f"     Pearson  r = {r:.4f}   (r^2 = {r**2:.4f})")
        print(f"     Spearman   = {rho:.4f}")
        print(f"     Spearman-Brown reliability of the 2-rep mean = {rel:.4f}")
        print(f"     => best Pearson any model can reach = {np.sqrt(rel):.4f}")

    ceiling("R1_pEF", "R2_pEF", "pEF (the regression target)")
    print("     The Pearson/Spearman gap means a minority of extreme tiles")
    print("     carries the correlation; treat the ceiling as a soft estimate.")
    # The robustness axis now gets the same treatment -- this is the check the
    # patch doc listed as unresolved. Avg_PGK exists, so it is resolvable.
    ceiling("R1_PGK", "R2_PGK", "PGK (the robustness axis)")

    print(f"\n  tiles with an Avg_PGK score  {int(df.Avg_PGK.notna().sum()):>10,}")
    print(f"  tiles with a retest score    {int(df.Avg_pEF_retest.notna().sum()):>10,}")
    L = df.sequence.dropna().str.len()
    print(f"\n  tile length: min {L.min()} max {L.max()} "
          f"| {int((L == 80).sum()):,} are exactly 80 aa "
          f"({int((L != 80).sum()):,} are not)")
    print("=" * 64)

    out = df if args.keep_unscored else df[df.Avg_pEF.notna() & df.sequence.notna()]
    out = out.reset_index(drop=True)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(out):,} rows)")


if __name__ == "__main__":
    main()
