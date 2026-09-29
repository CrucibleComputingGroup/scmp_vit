#!/usr/bin/env bash
# Lane J — is protection's sign set by the RATIO protect_sl / average budget?
#
# THE OBSERVATION. --protect-channel-stoc-len defaults to the ladder ceiling, and
# that resolves correctly everywhere (verified on 8 tables: t96 protects at 128,
# not at 64). But the ceiling is the HARD GRID LIMIT 2**(sc_prec-1) = 128 under
# --halve, so the ratio of protected stream length to average budget collapses
# with the budget:
#     eff-64   r32_pc   31.4 -> 64    2.04x   protection = +4.50
#     eff-128  m64_pc   62.8 -> 128   2.04x   protection = -0.10
#     eff-192  m96_pc   95.4 -> 128   1.34x   protection = -0.50
# Protection's whole mechanism is giving outlier channels a stream much longer
# than the bulk. At 1.34x it barely does that, while still paying full price:
# compensate_budget shrinks every other group to fund it. Cost flat, benefit
# halved -> sign flips.
#
# But eff-128 already sits at -0.10 with the ratio still at 2.04x, so budget and
# ratio are confounded across those three cells and the observation alone cannot
# separate them.
#
# THE TEST. Hold the budget at eff-128 and dial the ratio down by hand:
#   PC_SL=86  -> 86/62.8 = 1.37x, matching eff-192's ratio
#   PC_SL=128 -> 2.04x  (= m64_pc, already measured, the control)
# If protection at eff-128 degrades when only the ratio changes, the ratio is the
# mechanism and eff-192 is unfixable without sc_prec=9 or halve off. If PC_SL=86
# behaves like PC_SL=128, the ratio is NOT the cause and the eff-192 sign flip
# needs a different explanation.
#
# Also runs PC_SL=96 (1.53x) so a null result is not a two-point fluke, and the
# same ratio sweep at eff-64 where protection is strongly positive -- if the
# ratio story holds, eff-64 should ALSO degrade when throttled to 1.37x
# (PC_SL=43), which is the stronger version of the prediction because it has
# +4.50 of headroom to lose.
#
#   bash cls/lane_j_pcratio.sh
set -uo pipefail
cd /nfs/turbo/coe-nbleier/yjrcs/HPCA/vit_sc || exit 1
export TORCH_HOME=$PWD/.torch_cache
D=${IMAGENET_ROOT:-/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data}

N=${N:-1000}
NCAL=${NCAL:-128}
OUT=results/fig1k
S32=64,48,32,24,16
S64=128,96,64,48,32
ALLOPS=qkv_proj,out_proj,mlp_fc1,mlp_fc2,qk,av
export QKG=per_row MPOPS=$ALLOPS PRICING=macs

cell () {   # tag, budget, ladder, pc_sl
  local tag=$1 t=$2 ladder=$3 psl=$4
  [ -f "$OUT/$tag.json" ] && { echo "[J] $tag cached"; return; }
  if [ ! -f "results/mp_calib/${tag}_wrapper.json" ]; then
    CLW=macs PCF=0.02 PC_SL=$psl \
      bash cls/mp_cell2.sh "$tag" "$t" "$ladder" "$NCAL" 16 2>&1 \
      | grep -E "^\[calib\] protected|^\[calib\] expected|^\[cell\]|Error|Traceback|SystemExit|CUDA" || true
  fi
  [ -f "results/mp_calib/${tag}_wrapper.json" ] || { echo "[J] $tag NO TABLE"; return; }
  echo "=== [$tag] budget=$t protect_sl=$psl n=$N $(date +%T)"
  SC_MP_TRACE=$OUT/${tag}_trace.json python -u cls/eval.py \
    --mode sc --sc_prec 8 --halve 1 --chunk_d 128 --stoc_len "$t" \
    --sc_ops mlp_fc1,mlp_fc2,qk,av,proj --qk_granularity per_row \
    --adaptive_mp 1 --adaptive_mp_table "results/mp_calib/${tag}_wrapper.json" \
    --adaptive_mp_ops "$ALLOPS" \
    --data_root "$D" --max_images "$N" --batch_size 16 --workers 8 --seed 0 \
    --out_json "$OUT/$tag.json" 2>&1 \
    | grep -viE "futurewarning|pynvml|warnings.warn" \
    | grep -E "^\[result\]|Error|Traceback|SystemExit"
}

echo "########## Lane J: protection ratio sweep  $(hostname) $(date +%T)"
# eff-128, budget fixed at 64, ratio throttled
cell m64_pc_r137 64 $S64 86    # 1.37x  == eff-192's ratio
cell m64_pc_r153 64 $S64 96    # 1.53x
# eff-64, budget fixed at 32, ratio throttled -- the strong version of the test
cell r32_pc_r137 32 $S32 43    # 1.37x
cell r32_pc_r153 32 $S32 48    # 1.53x
echo "########## Lane J COMPLETE $(date +%T)"
