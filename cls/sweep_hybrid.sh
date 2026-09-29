#!/usr/bin/env bash
# 10% hybrid INT mask: measure SC fragility, cut the mask, then run BOTH arms.
#
# The point of running uniform+mask and MP+mask is that the mask is not free —
# it moves work off the SC budget into INT, so a masked cell's SC average
# understates its total compute. Comparing MP+mask against uniform+mask holds
# the mask fixed, which is the only comparison the mask cannot flatter.
# Comparing either against an UNMASKED cell is an energy-axis claim.
set -uo pipefail
cd /nfs/turbo/coe-nbleier/yjrcs/HPCA/vit_sc || exit 1
export TORCH_HOME=$PWD/.torch_cache
D=${IMAGENET_ROOT:-/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data}

NEVAL=${NEVAL:-2048}
BUDGETS=${BUDGETS:-"32 64"}
HYB=results/hybrid
OUT=results/hybrid_eval
mkdir -p $HYB $OUT

for T in $BUDGETS; do
  # 1. per-(op, block) SC fragility at THIS budget. Fragility is
  #    budget-dependent, so each budget gets its own sensitivity run.
  if [ ! -f "$HYB/sens_t$T.json" ]; then
    python -u cls/build_hybrid_mask.py --stoc_len $T --tag t$T \
      --frac 0.05,0.10,0.20 --int_bits 7 --int_sym 0 \
      --n_images 512 --batch_size 16 --data_root "$D" 2>&1 \
      | grep -viE "futurewarning|pynvml|warnings.warn" \
      | grep -E "^\[sens\]|^ +[0-9]+ |Error|Traceback"
  else
    echo "[hyb] sensitivity for t$T cached"
  fi

  M=$HYB/mask10_t$T.json
  [ -f "$M" ] || { echo "[hyb] FAILED: no mask for t$T"; continue; }

  # 2. uniform SC + the mask
  TAG=uni_hyb10_t$T
  if [ ! -f "$OUT/$TAG.json" ]; then
    SC_MP_TRACE=$OUT/${TAG}_trace.json python -u cls/eval.py \
      --mode sc --sc_prec 8 --halve 1 --chunk_d 128 --stoc_len $T \
      --sc_ops mlp_fc1,mlp_fc2,qk,av,proj \
      --hybrid_config_json "$M" \
      --data_root "$D" --max_images "$NEVAL" --batch_size 16 --workers 4 --seed 0 \
      --out_json "$OUT/$TAG.json" 2>&1 \
      | grep -viE "futurewarning|pynvml|warnings.warn" \
      | grep -E "^\[hybrid\]|^\[result\]|Error|Traceback"
  fi

  # 3. MP + the same mask. Reuses the wave-1/2 winner's table: the mask changes
  #    which groups run SC at all, not how the surviving ones allocate, so a
  #    table calibrated without the mask is still the right allocation for the
  #    groups that stay. (Re-calibrating UNDER the mask is the stronger version
  #    and is the obvious follow-up if this arm is close.)
  SRC=${MP_TABLE_T:-results/mp_calib/s${T}_base_wrapper.json}
  TAG=mp_hyb10_t$T
  if [ -f "$SRC" ] && [ ! -f "$OUT/$TAG.json" ]; then
    SC_MP_TRACE=$OUT/${TAG}_trace.json python -u cls/eval.py \
      --mode sc --sc_prec 8 --halve 1 --chunk_d 128 --stoc_len $T \
      --sc_ops mlp_fc1,mlp_fc2,qk,av,proj \
      --adaptive_mp 1 --adaptive_mp_table "$SRC" \
      --adaptive_mp_ops qkv_proj,out_proj,mlp_fc1,mlp_fc2 \
      --hybrid_config_json "$M" \
      --data_root "$D" --max_images "$NEVAL" --batch_size 16 --workers 4 --seed 0 \
      --out_json "$OUT/$TAG.json" 2>&1 \
      | grep -viE "futurewarning|pynvml|warnings.warn" \
      | grep -E "^\[hybrid\]|^\[sc\] adaptive|^\[result\]|Error|Traceback"
  elif [ ! -f "$SRC" ]; then
    echo "[hyb] no MP table at $SRC — skipping MP+mask for t$T"
  fi
done

echo "=== sweep_hybrid COMPLETE $(date +%T)"
