#!/usr/bin/env bash
# How much throughput is left on the table at batch 16?
#
# The 50k figure runs are ~2.1 h/cell at batch 16 and ~7 img/s, which does not
# all fit in one allocation. Batch size is the obvious lever: at batch 16 a
# ViT-L/14 forward is 4112 rows, far too small to saturate a Blackwell.
#
# UNIFORM SC is batch-INVARIANT in accuracy — per_row quantization derives each
# row's (max,min) from that row alone, so pooling more rows per call changes
# nothing but occupancy. MP is NOT: the classifier min-max normalizes the
# dispatch metric over the rows in one call, so calibration and eval must share
# a batch size. Hence uniform can simply be run faster; MP needs a paired
# re-calibration, which sweep_wave2 covers.
set -uo pipefail
cd /nfs/turbo/coe-nbleier/yjrcs/HPCA/vit_sc || exit 1
export TORCH_HOME=$PWD/.torch_cache
D=${IMAGENET_ROOT:-/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data}
N=${N:-1024}
OUT=results/probe
mkdir -p $OUT

for BS in 16 32 64 128; do
  echo "--- batch $BS ---"
  python -u cls/eval.py --mode sc --sc_prec 8 --halve 1 --chunk_d 128 \
    --stoc_len 64 --sc_ops mlp_fc1,mlp_fc2,qk,av,proj \
    --data_root "$D" --max_images $N --batch_size $BS --workers 8 --seed 0 \
    --out_json $OUT/probe_bs$BS.json 2>&1 \
    | grep -viE "futurewarning|pynvml|warnings.warn" \
    | grep -E "^\[result\]|Error|Traceback|out of memory"
done
echo "=== probe_batch COMPLETE $(date +%T)"
