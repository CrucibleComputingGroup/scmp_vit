#!/usr/bin/env bash
# Lane I — why the hybrid-MP curve dips from eff-64 to eff-128, and the fix.
#
# `hyb_mppc_32iso` 0.8490 (eff-64) vs `hyb_mppc_64iso` 0.8480 (eff-128) reads as
# "double the compute, slightly worse". Three separate causes, none of them a
# real regression:
#
# 1. IT IS A TIE. Paired: -0.10 pt, CI [-1.60, +1.40], p=1.00, 59 discordant.
#    Both sit within 0.6 pt of FP (0.8540), so there is no headroom left to rise
#    into -- the same saturation that flattens everything above eff-128.
#
# 2. THE eff-64 POINT IS A HIGH CALIBRATION DRAW. Over 4 seeds that cell means
#    0.8453 (sd 0.59); the plotted draw is 0.8490, i.e. +0.37 above its own mean.
#    Against the seed-mean, eff-64 -> eff-128 is +0.27 -- monotone again. The
#    eff-128 point has only ONE draw, so it never got the same treatment.
#
# 3. THE eff-128 HYBRID IS STILL 3.1% UNDER BUDGET. The 1/(1-f) raise to 71
#    under-shot: realized true sl came out 67.79 against `m64_pc`'s 63.05, which
#    is 4942.3T vs 5098.4T cycles. The UNIFORM t64 arm landed at +0.0%, so this
#    is specific to the MP arm -- the mask removed groups whose MP allocation was
#    ABOVE average, so dropping them pulled the mean down further than the raise
#    pushed it up. The 1/(1-f) rule assumes the masked groups are allocated at
#    the average, which is true for a uniform arm and false for an MP one.
#    Corrected target: 71 * 1.0316 = 73.
#
# So this lane recalibrates the eff-128 hybrid at 73 AND replicates it over the
# same 4 calibration seeds as eff-64, making the two budgets comparable on both
# axes. If the dip is what the three causes say it is, the corrected seed-mean at
# eff-128 should land at or above the eff-64 seed-mean.
#
#   bash cls/lane_i_eff128fix.sh
set -uo pipefail
cd /nfs/turbo/coe-nbleier/yjrcs/HPCA/vit_sc || exit 1
export TORCH_HOME=$PWD/.torch_cache
D=${IMAGENET_ROOT:-/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data}

N=${N:-1000}
NCAL=${NCAL:-128}
OUT=results/fig1k
S64=128,96,64,48,32
ALLOPS=qkv_proj,out_proj,mlp_fc1,mlp_fc2,qk,av
M64=results/hybrid/mask10_t64.json
export QKG=per_row MPOPS=$ALLOPS PRICING=macs

cell () {   # tag, calib_seed
  local tag=$1 seed=$2
  [ -f "$OUT/$tag.json" ] && { echo "[I] $tag cached"; return; }
  if [ ! -f "results/mp_calib/${tag}_wrapper.json" ]; then
    CLW=macs PCF=0.02 CALIB_SEED=$seed \
      bash cls/mp_cell2.sh "$tag" 73 $S64 "$NCAL" 16 2>&1 \
      | grep -E "^\[calib\] expected|^\[cell\]|Error|Traceback|SystemExit|CUDA" || true
  fi
  [ -f "results/mp_calib/${tag}_wrapper.json" ] || { echo "[I] $tag NO TABLE"; return; }
  echo "=== [$tag] MP@73 + mask10_t64, seed $seed, n=$N $(date +%T)"
  SC_MP_TRACE=$OUT/${tag}_trace.json python -u cls/eval.py \
    --mode sc --sc_prec 8 --halve 1 --chunk_d 128 --stoc_len 73 \
    --sc_ops mlp_fc1,mlp_fc2,qk,av,proj --qk_granularity per_row \
    --adaptive_mp 1 --adaptive_mp_table "results/mp_calib/${tag}_wrapper.json" \
    --adaptive_mp_ops "$ALLOPS" --hybrid_config_json "$M64" \
    --data_root "$D" --max_images "$N" --batch_size 16 --workers 8 --seed 0 \
    --out_json "$OUT/$tag.json" 2>&1 \
    | grep -viE "futurewarning|pynvml|warnings.warn" \
    | grep -E "^\[result\]|Error|Traceback|SystemExit"
}

echo "########## Lane I: eff-128 hybrid at the corrected target 73, 4 seeds  $(date +%T)"
cell hyb_mppc_64iso2     1234
cell hyb_mppc_64iso2_s7  7
cell hyb_mppc_64iso2_s23 23
cell hyb_mppc_64iso2_s41 41
echo "########## Lane I COMPLETE $(date +%T)"
