#!/usr/bin/env bash
# Control: does the sensitivity RANKING matter, or would any 10% mask do?
#
# The t32 and t64 masks share only 2 of 14 groups, and the probe's mean dL was
# NEGATIVE for every operator — both say the ranking is largely noise. If a
# random 10% mask (matched per-op, so only WHICH blocks differ) performs the
# same as the ranked one, then "10% of groups in INT7" is the whole effect and
# the sensitivity probe bought nothing. That is a result either way, and it is
# two evals.
#
# Runs AFTER the main queue so it never contends for the Exclusive_Process GPU.
set -uo pipefail
cd /nfs/turbo/coe-nbleier/yjrcs/HPCA/vit_sc || exit 1
export TORCH_HOME=$PWD/.torch_cache
D=${IMAGENET_ROOT:-/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data}
N=${N:-1000}
OUT=results/fig1k

echo "[ctrl] waiting for the main queue $(date +%T)"
while pgrep -f "cls/run_all_1k.sh" > /dev/null; do sleep 30; done
echo "[ctrl] GPU free $(date +%T)"

for T in 32 64; do
  M=results/hybrid/maskrand10_t$T.json
  [ -f "$M" ] || continue
  tag=hyb_uni_rand_$T
  [ -f "$OUT/$tag.json" ] && continue
  echo "=== [$tag] $(date +%T)"
  SC_MP_TRACE=$OUT/${tag}_trace.json python -u cls/eval.py \
    --mode sc --sc_prec 8 --halve 1 --chunk_d 128 --stoc_len $T \
    --sc_ops mlp_fc1,mlp_fc2,qk,av,proj --qk_granularity per_row \
    --hybrid_config_json "$M" \
    --data_root "$D" --max_images "$N" --batch_size 16 --workers 8 --seed 0 \
    --out_json "$OUT/$tag.json" 2>&1 \
    | grep -viE "futurewarning|pynvml|warnings.warn" \
    | grep -E "^\[hybrid\]|^\[result\]|Error|Traceback"
done
echo "=== run_mask_control COMPLETE $(date +%T)"
