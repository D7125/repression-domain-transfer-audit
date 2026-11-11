#!/bin/bash
# Run the remaining GPU arms back to back; each writes its own completion line.
set -u
PY=~/miniconda3/envs/machine_deep_learning/bin/python
R="/home/dmin/ai&stem_cell/run/rev2"
cd "/home/dmin/ai&stem_cell/cnn_AI&stemcell" || exit 1

# wait for pass A to finish writing
while pgrep -f "cd43_model.py --mode traj" >/dev/null; do sleep 20; done

NSTAR=$($PY -I -c "
import pandas as pd
d=pd.read_json('$R/cd43_traj.json')
piv=d.pivot_table(index='epoch',values='val_mse')
print(int(piv.val_mse.idxmin()))
")
echo "### N* chosen on the CD43 validation curve = $NSTAR"

echo "### [1/3] CD43 pass B"
$PY -u cd43_model.py --mode fixed --n "$NSTAR" --seeds 0 1 2 \
    --out "$R/cd43_fixed.json" > "$R/cd43_fixed.log" 2>&1
echo "### [1/3] done rc=$?"

echo "### [2/3] retest seeds + TOST"
$PY -u -I "$R/retest_seeds.py" --seeds 0 1 2 3 4 5 6 7 8 9 \
    > "$R/retest_seeds.log" 2>&1
echo "### [2/3] done rc=$?"

echo "### [3/3] saliency sanity checks"
$PY -u -I "$R/saliency_sanity.py" > "$R/saliency_sanity.log" 2>&1
echo "### [3/3] done rc=$?"
echo "### CHAIN COMPLETE"
