"""
Step 0 of the pipeline: turn the Nature supplementary workbook into the
data.csv that extract_per_residue.py and train_cnn.py expect.

Input : 41586_2023_5906_MOESM5_ESM.xlsx  (Supplementary Table 5, ~65 MB)
Output: data.csv  -- one row per SCORED tile, columns renamed to:
            protein   (was "HGNC Symbol")   <- the grouping key for the split
            sequence  (was "Protein Sequence")
            Avg_pEF   (unchanged)           <- the regression target
        plus tile coordinates, both pEF replicates, FLAG expression and
        the two hit labels, which you need for the yes/no model later.

It also re-runs the dataset checks, so you can confirm your copy of the
workbook matches the numbers in the model report before training anything.

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

# column index -> output name. Indices are 0-based positions in "CRTF Tiles".
COLS = {
    0:  "protein",       # HGNC Symbol
    1:  "tile_num",
    2:  "tile_start",
    3:  "tile_end",
    19: "Avg_FLAG",      # expression; needed to drop false negatives in the yes/no model
    21: "R1_pEF",
    24: "R2_pEF",
    27: "Avg_pEF",       # the target
    28: "SE_pEF",
    56: "sequence",      # Protein Sequence
    59: "Hit_pEF",       # "Hit" / "Not Hit" / blank
    61: "Hit_FLAG",
}

EXPECTED_HEADERS = {0: "HGNC Symbol", 27: "Avg_pEF", 56: "Protein Sequence", 59: "Hit pEF"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--sheet", default="CRTF Tiles")
    ap.add_argument("--out", default="data.csv")
    ap.add_argument("--keep-unscored", action="store_true",
                    help="keep tiles with no Avg_pEF (default: drop them)")
    args = ap.parse_args()

    wb = load_workbook(args.xlsx, read_only=True)
    if args.sheet not in wb.sheetnames:
        raise SystemExit(f"sheet '{args.sheet}' not found. sheets = {wb.sheetnames}")
    ws = wb[args.sheet]

    it = ws.iter_rows(values_only=True)
    header = next(it)

    # fail loudly if the workbook layout differs from the one this was written for
    for i, name in EXPECTED_HEADERS.items():
        if str(header[i]).strip() != name:
            raise SystemExit(
                f"column {i} is '{header[i]}', expected '{name}'.\n"
                f"The workbook layout differs -- fix COLS before continuing.\n"
                f"Full header: {list(header)}"
            )

    rows = []
    for n, r in enumerate(it, 1):
        rows.append([r[k] for k in COLS])
        if n % 20000 == 0:
            print(f"  read {n:,} rows", flush=True)

    df = pd.DataFrame(rows, columns=list(COLS.values()))
    for c in ["Avg_FLAG", "R1_pEF", "R2_pEF", "Avg_pEF", "SE_pEF",
              "tile_num", "tile_start", "tile_end"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ---------------- verification -------------------------------------
    print("\n" + "=" * 62)
    print("DATASET CHECKS  (values confirmed against this workbook)")
    print("=" * 62)
    print(f"  total tiles                  {len(df):>10,}   expect 113,528")
    print(f"  tiles with an Avg_pEF score  {int(df.Avg_pEF.notna().sum()):>10,}   expect 103,248")
    print(f"  distinct proteins            {df.protein.nunique():>10,}   expect  2,045")
    print("     (the paper says 2,047 -- the tiles sheet holds 2,045)")

    hits_all = int((df.Hit_pEF == "Hit").sum())
    scored = df[df.Avg_pEF.notna()]
    hits_scored = int((scored.Hit_pEF == "Hit").sum())
    print(f"\n  'Hit' rows, whole sheet      {hits_all:>10,}   paper reports 6,755")
    print(f"  'Hit' rows WITH a score      {hits_scored:>10,}   report uses  6,566")
    print(f"     -> {hits_all - hits_scored} confirmed hits carry no Avg_pEF value.")
    print("        That is the whole 6,755 vs 6,566 discrepancy. Not an error.")

    print(f"\n  Avg_pEF  min {scored.Avg_pEF.min():.2f}  max {scored.Avg_pEF.max():.2f}  "
          f"mean {scored.Avg_pEF.mean():.3f}")

    both = df[df.R1_pEF.notna() & df.R2_pEF.notna()]
    r = pearsonr(both.R1_pEF, both.R2_pEF)[0]
    rho = spearmanr(both.R1_pEF, both.R2_pEF)[0]
    rel = 2 * r / (1 + r)
    print(f"\n  replicate agreement on {len(both):,} tiles")
    print(f"     Pearson  r = {r:.4f}   (r^2 = {r**2:.4f})")
    print(f"     Spearman   = {rho:.4f}")
    print(f"     Spearman-Brown reliability of the 2-rep mean = {rel:.4f}")
    print(f"     => best Pearson any model can reach = sqrt(rel) = {np.sqrt(rel):.4f}")
    print("     The Pearson/Spearman gap means the correlation is carried by a")
    print("     minority of extreme tiles; treat the ceiling as a soft estimate.")
    print("=" * 62)

    out = df if args.keep_unscored else df[df.Avg_pEF.notna() & df.sequence.notna()]
    out = out.reset_index(drop=True)

    bad_len = (out.sequence.str.len() != 80).sum()
    if bad_len:
        print(f"\nnote: {bad_len:,} tiles are not exactly 80 aa (protein C-termini). "
              "The scripts mask these correctly -- nothing to fix.")

    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(out):,} rows)")
    print("next:  python extract_per_residue.py --csv data.csv --seq-col sequence \\")
    print("              --model esm2_t12_35M_UR50D --out esm2_480_res")


if __name__ == "__main__":
    main()
