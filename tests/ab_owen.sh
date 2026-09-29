#!/usr/bin/env bash
# A/B the Owen scramble mode on the all-ops SC path.
set -u
cd /nfs/turbo/coe-nbleier/yjrcs/HPCA/vit_sc || exit 1
export TORCH_HOME=$PWD/.torch_cache
D=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data
N=${1:-256}

for m in bitrev off; do
  echo "--- SC_OWEN_MODE=$m (all 5 ops, chunk_d=128, sl=256) ---"
  SC_OWEN_MODE=$m python -u cls/eval.py --mode sc --sc_prec 8 --stoc_len 256 \
    --sc_ops mlp_fc1,mlp_fc2,qk,av,proj --max_images "$N" \
    --batch_size 16 --workers 4 --data_root "$D" 2>&1 \
    | grep -viE "futurewarning|import pynvml" | grep -E "^\[result\]"
done
