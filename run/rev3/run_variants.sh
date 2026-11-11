#!/bin/bash
# The GPU programme for referee items M10, M4 and Mock-M5. Sequential on
# purpose: one 8 GB card, and two concurrent runs would just thrash.
set -u
P=~/miniconda3/envs/machine_deep_learning/bin/python
E=:4096:8
R="$HOME/ai&stem_cell/run/rev3"
D480="$HOME/ai&stem_cell/run/esm2_480_res"
D640="$R/esm2_640"

echo "######## M10: ESM-2 150M (D=640) vs the 35M (D=480) baseline"
CUBLAS_WORKSPACE_CONFIG=$E $P "$R/cd43_variants.py" --tag esm640 --emb "$D640" \
    --seeds 0 1 2 3 4 --deterministic --both-arms
echo "######## M10 control: same 5 seeds at D=480, same script"
CUBLAS_WORKSPACE_CONFIG=$E $P "$R/cd43_variants.py" --tag esm480_5 --emb "$D480" \
    --seeds 0 1 2 3 4 --deterministic --both-arms

echo "######## M4: train on one CD43 replicate, score the other"
CUBLAS_WORKSPACE_CONFIG=$E $P "$R/cd43_variants.py" --tag br1 --emb "$D480" \
    --label br1 --score-against other_rep --seeds 0 1 2 3 4 5 6 7 8 9 --deterministic
CUBLAS_WORKSPACE_CONFIG=$E $P "$R/cd43_variants.py" --tag br2 --emb "$D480" \
    --label br2 --score-against other_rep --seeds 0 1 2 3 4 5 6 7 8 9 --deterministic

echo "######## M4b: labels restricted to tiles called hit in BOTH replicates"
CUBLAS_WORKSPACE_CONFIG=$E $P "$R/cd43_variants.py" --tag hitboth --emb "$D480" \
    --restrict-hitboth --seeds 0 1 2 3 4 5 6 7 8 9 --deterministic

echo "######## Mock-M5: MMseqs2 identity 20 / 30 / 40 percent"
for id in 20 30 40; do
  CUBLAS_WORKSPACE_CONFIG=$E $P "$R/cd43_variants.py" --tag id$id --emb "$D480" \
      --clusters "$R/clu/protein_clusters_id$id.tsv" \
      --seeds 0 1 2 3 4 --deterministic --both-arms
done
echo "ALL_VARIANTS_DONE"
