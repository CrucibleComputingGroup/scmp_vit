#!/usr/bin/env bash
# Corrected recipe cells, after the escape-gate interaction was measured.
#
# Phase B was designed before the result that the escape gate HURTS once
# protected channels are on: r32_pc 0.828 -> r32_pc_esc 0.808, paired -2.00 pt
# (CI [-3.60,-0.40], p=0.017), both iso-compute. The gate is a workaround for a
# broken row ranking; protection repairs the ranking, so the override then
# damages a good one. That makes the cells Phase B carries to eff 128/192
# (r64_pc_esc, r96_pc_esc) the WRONG recipe, and r32_full confounded.
#
# Those cells are still worth having — if the gate also hurts at 128/192 the
# interaction replicates across budgets — so this does not replace them, it
# adds the protection-only recipe they should be compared against.
#
# r32_pc_auto additionally tests a prediction the mechanism makes: with the
# metric already repaired by protection, --metric-select auto should add
# LITTLE, because there is no longer a badly-ranked operator for it to rescue.
set -uo pipefail
cd /nfs/turbo/coe-nbleier/yjrcs/HPCA/vit_sc || exit 1
export TORCH_HOME=$PWD/.torch_cache

echo "[followup] waiting for run_rest $(date +%T)"
while pgrep -f "cls/run_rest.sh" > /dev/null; do sleep 30; done
echo "[followup] GPU free $(date +%T)"

N=${N:-1000}
NCAL=${NCAL:-128}
S32=64,48,32,24,16
S64=128,96,64,48,32
S96=128,112,96,80,64,48
ALLOPS=qkv_proj,out_proj,mlp_fc1,mlp_fc2,qk,av
OUT=results/fig1k
export QKG=per_row MPOPS=$ALLOPS PRICING=macs

mp () {
  local tag=$1 t=$2 ladder=$3
  if [ ! -f "results/mp_calib/${tag}_wrapper.json" ]; then
    bash cls/mp_cell2.sh "$tag" "$t" "$ladder" "$NCAL" 16 2>&1 \
      | grep -E "^\[calib\] (expected|escape|protected)|Error|Traceback|CUDA" || true
  fi
  [ -f "results/mp_calib/${tag}_wrapper.json" ] || { echo "[fu] $tag NO TABLE"; return; }
  [ -f "$OUT/$tag.json" ] && { echo "[fu] $tag cached"; return; }
  echo "=== [$tag] $(date +%T)"
  SC_MP_TRACE=$OUT/${tag}_trace.json python -u cls/eval.py \
    --mode sc --sc_prec 8 --halve 1 --chunk_d 128 --stoc_len "$t" \
    --sc_ops mlp_fc1,mlp_fc2,qk,av,proj --qk_granularity per_row \
    --adaptive_mp 1 --adaptive_mp_table "results/mp_calib/${tag}_wrapper.json" \
    --adaptive_mp_ops "$ALLOPS" \
    --data_root "${IMAGENET_ROOT:-/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data}" \
    --max_images "$N" --batch_size 16 --workers 8 --seed 0 \
    --out_json "$OUT/$tag.json" 2>&1 \
    | grep -viE "futurewarning|pynvml|warnings.warn" \
    | grep -E "^\[result\]|Error|Traceback"
}

# protection + metric-select, NO gate. Compare against r32_pc (0.828) to see
# whether metric selection still has anything to fix once protection has
# repaired the ranking.
CLW=macs PCF=0.02 METRIC=auto mp r32_pc_auto 32 $S32
# the protection-only recipe at the other two figure budgets
         PCF=0.02             mp r64_pc      64 $S64
         PCF=0.02             mp r96_pc      96 $S96

# Calibration-seed replication of the ACTUAL winner. Phase D seeds
# r32_pc_esc, which the gate interaction showed is 2 pt worse — replicating it
# would put three draws behind the wrong recipe. At n_calib=128 a single draw
# carries ~1 pt of calibration noise (MP_REPORT), which is a quarter of the
# +4.50 effect, so this is what decides whether +4.50 is a result or a draw.
for SEED in 7 23 41; do
  CLW=macs PCF=0.02 CALIB_SEED=$SEED mp "r32_pc_s$SEED" 32 $S32
done

echo "=== run_followup COMPLETE $(date +%T)"

# Hybrid MP arm rebuilt on the CORRECT table. Phase C in run_all_1k.sh prefers
# r{T}_pc_esc, chosen before the gate interaction was known; that table is 2 pt
# worse than r{T}_pc, so hyb_mp_{T} understates the MP+mask arm. These are the
# cells the uniform+mask arm should actually be differenced against.
D=${IMAGENET_ROOT:-/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data}
for T in 32 64; do
  M=results/hybrid/mask10_t$T.json
  W=results/mp_calib/r${T}_pc_wrapper.json
  tag=hyb_mppc_$T
  [ -f "$M" ] && [ -f "$W" ] || { echo "[fu] $tag: 缺 mask 或表"; continue; }
  [ -f "$OUT/$tag.json" ] && { echo "[fu] $tag cached"; continue; }
  echo "=== [$tag] $(date +%T)"
  SC_MP_TRACE=$OUT/${tag}_trace.json python -u cls/eval.py \
    --mode sc --sc_prec 8 --halve 1 --chunk_d 128 --stoc_len $T \
    --sc_ops mlp_fc1,mlp_fc2,qk,av,proj --qk_granularity per_row \
    --adaptive_mp 1 --adaptive_mp_table "$W" --adaptive_mp_ops "$ALLOPS" \
    --hybrid_config_json "$M" \
    --data_root "$D" --max_images "$N" --batch_size 16 --workers 8 --seed 0 \
    --out_json "$OUT/$tag.json" 2>&1 \
    | grep -viE "futurewarning|pynvml|warnings.warn" \
    | grep -E "^\[hybrid\]|^\[result\]|Error|Traceback"
done
echo "=== followup addendum COMPLETE $(date +%T)"
