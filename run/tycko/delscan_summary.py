"""Recompute the delScan descriptive numbers quoted in section 3.3.

  python3 delscan_summary.py [path/to/delscan.csv]

The deletion-window sublibrary is the only systematic window-deletion screen in
either source paper, and the paper uses it to answer one question: are those
scans anchored on activation domains rather than repression domains? That is a
descriptive claim about 51 scans, so it is reproduced here from the released
table rather than left as a number in the prose.

Every value is asserted against what the paper states. A mismatch exits
non-zero and names the value, so this file cannot drift away from the text.

`delscan.csv` is the delScan rows of Tycko et al. 2025 Table S2 with six
columns derived here (prot, startlab, dellab, coord, plen, delpos). It is
released with the code because no script builds it: it was assembled directly
from the supplementary table.
"""
import sys

import pandas as pd

# Hit-calling convention, stated once in section 2: 2 sd above the median of
# that screen's own random controls. For the CD43 repression screen this is
# +0.6297 over 1,995 controls.
CD43_THRESHOLD = 0.6297

# What section 3.3 says. Keys are the labels printed below.
PAPER = {
    "elements": 2024,
    "with CD43 score": 2010,
    "scans": 51,
    "proteins": 47,
    "max variants per scan": 41,
    "scans above the repression threshold": 1,
    "scans with a negative median": 40,
}


def main(path="delscan.csv"):
    d = pd.read_csv(path)
    scans = d.groupby(["prot", "startlab"])
    med = scans.cd43.median()
    size = scans.size()

    got = {
        "elements": len(d),
        "with CD43 score": int(d.cd43.notna().sum()),
        "scans": scans.ngroups,
        "proteins": int(d.prot.nunique()),
        "max variants per scan": int(size.max()),
        "scans above the repression threshold": int((med > CD43_THRESHOLD).sum()),
        "scans with a negative median": int((med < 0).sum()),
    }

    bad = []
    for k, want in PAPER.items():
        flag = "ok" if got[k] == want else "MISMATCH"
        if flag == "MISMATCH":
            bad.append(f"{k}: paper says {want}, table gives {got[k]}")
        print(f"  {k:<38} {got[k]:>6}   (paper: {want})  {flag}")

    # Reported separately because the design figure (41) is not the count in
    # every scan: truncated scans are shorter.
    print(f"\n  variants per scan: median {int(size.median())}, "
          f"range {int(size.min())}-{int(size.max())} over {scans.ngroups} scans")
    print(f"  CD43 medians: min {med.min():.3f}, max {med.max():.3f}, "
          f"threshold {CD43_THRESHOLD}")

    if bad:
        raise SystemExit("\n".join(["", "values no longer match the paper:"] + bad))
    print("\nDONE_DELSCAN_SUMMARY")


if __name__ == "__main__":
    main(*sys.argv[1:])
