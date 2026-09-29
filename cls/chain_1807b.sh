#!/usr/bin/env bash
# gl1807 stage 2: build the hybrid masks and validate the hybrid eval path.
#
# The mask BUILD only depends on the budget, not on which MP table wins, so it
# can run before wave 2 settles. Masks are cut at qk per_row because that is
# what every deployed cell now uses, and SC fragility is measured on the model
# as it will actually run.
set -uo pipefail
cd /nfs/turbo/coe-nbleier/yjrcs/HPCA/vit_sc || exit 1
export TORCH_HOME=$PWD/.torch_cache
D=${IMAGENET_ROOT:-/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data}

echo "[chain1807b] waiting for stage 1 $(date +%T)"
while pgrep -f "cls/chain_1807.sh" > /dev/null; do sleep 30; done
echo "[chain1807b] GPU free $(date +%T)"

for T in 32 64 96; do
  python -u cls/build_hybrid_mask.py --stoc_len $T --tag t$T \
    --frac 0.05,0.10,0.20 --int_bits 7 --int_sym 0 \
    --qk_granularity per_row \
    --n_images 512 --batch_size 16 --data_root "$D" 2>&1 \
    | grep -viE "futurewarning|pynvml|warnings.warn" \
    | grep -E "^\[sens\]|^ +[0-9]+ +|Error|Traceback"
done

# Validate the hybrid eval path end-to-end on a small subset before any 50k run
# depends on it.
SC_MP_TRACE=results/hybrid/smoke_hyb_trace.json python -u cls/eval.py \
  --mode sc --sc_prec 8 --halve 1 --chunk_d 128 --stoc_len 64 \
  --sc_ops mlp_fc1,mlp_fc2,qk,av,proj --qk_granularity per_row \
  --hybrid_config_json results/hybrid/mask10_t64.json \
  --data_root "$D" --max_images 512 --batch_size 16 --workers 4 --seed 0 \
  --out_json results/hybrid/smoke_hyb.json 2>&1 \
  | grep -viE "futurewarning|pynvml|warnings.warn" \
  | grep -E "^\[hybrid\]|^\[result\]|Error|Traceback"

echo "=== chain_1807b COMPLETE $(date +%T)"
