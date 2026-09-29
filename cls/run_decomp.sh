#!/usr/bin/env bash
# The per_head uniform cells, so the decomposition chain has a starting point.
#
# Phase A runs uniform SC at qk per_row only, because that is the configuration
# everything ships at. But the session's headline is that per_row is itself a
# fix worth +3.2 pt, and that claim needs its own before-picture ON THIS
# SUBSET — the +3.2 was measured at n=2048, a different sample.
#
# With these, the whole result decomposes into one chain at a fixed budget:
#   uniform per_head  ->  uniform per_row  ->  + protected channels  ->  + MP
# i.e. two scale fixes and then the allocator, which is the honest ordering of
# what actually bought the accuracy.
set -uo pipefail
cd /nfs/turbo/coe-nbleier/yjrcs/HPCA/vit_sc || exit 1
export TORCH_HOME=$PWD/.torch_cache
D=${IMAGENET_ROOT:-/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data}
N=${N:-1000}
OUT=results/fig1k

echo "[decomp] waiting for earlier queues $(date +%T)"
while pgrep -f "cls/run_all_1k.sh|cls/run_mask_control.sh|cls/run_protect_control.sh" > /dev/null; do
  sleep 30
done
echo "[decomp] GPU free $(date +%T)"

for SL in 32 64; do
  tag=uni_head_$SL
  [ -f "$OUT/$tag.json" ] && continue
  echo "=== [$tag] $(date +%T)"
  SC_MP_TRACE=$OUT/${tag}_trace.json python -u cls/eval.py \
    --mode sc --sc_prec 8 --halve 1 --chunk_d 128 --stoc_len $SL \
    --sc_ops mlp_fc1,mlp_fc2,qk,av,proj --qk_granularity per_head \
    --data_root "$D" --max_images "$N" --batch_size 16 --workers 8 --seed 0 \
    --out_json "$OUT/$tag.json" 2>&1 \
    | grep -viE "futurewarning|pynvml|warnings.warn" \
    | grep -E "^\[result\]|Error|Traceback"
done
echo "=== run_decomp COMPLETE $(date +%T)"
