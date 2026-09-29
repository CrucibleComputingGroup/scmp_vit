#!/usr/bin/env bash
# Lane B (gl1807) — does CLW=macs fix MP at the MILD budgets?
#
# WHY. Every winning cell at t32 ran CLW=macs; t64 and t96 never did (see
# run_all_1k.sh:128-130 and run_followup.sh:61-62, which leave CLW at its
# `uniform` default). That default was chosen on 2026-07-25 from a 4-seed
# replication -- but that replication predates per_row qk, attention in the MP
# table, and protected channels, so it does not transfer to this recipe.
#
# The damage is visible in the t96 trace: with W_g=uniform the allocator counts
# benefit per ROW and cost per MAC, so it pins the cheap `out_proj` (8% of MACs,
# 1.05M MACs/row) at the 127 ladder ceiling and pays for it out of `mlp_fc1`
# (30% of MACs, 4.19M MACs/row), which drops 96 -> 64.8 cycles. W_g=macs puts
# benefit and cost in one currency and the per-row argmin collapses to
# sigma + lambda'*cycles for every operator.
#
# CELLS. Each budget gets base (MP alone) and pc (MP + 2% protected channels)
# so the protection increment stays separable, exactly as at t32.
#
#   bash cls/lane_b_clwmacs.sh
set -uo pipefail
cd /nfs/turbo/coe-nbleier/yjrcs/HPCA/vit_sc || exit 1
export TORCH_HOME=$PWD/.torch_cache
D=${IMAGENET_ROOT:-/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data}

N=${N:-1000}
NCAL=${NCAL:-128}
OUT=results/fig1k
mkdir -p "$OUT" results/logs

S64=128,96,64,48,32
S96=128,112,96,80,64,48
ALLOPS=qkv_proj,out_proj,mlp_fc1,mlp_fc2,qk,av

# Same environment every t32 winner used. PRICING=macs is mandatory with qk/av
# in the table (rep_g charges an attention row 16x a linear row and starves
# attention to the ladder floor).
export QKG=per_row MPOPS=$ALLOPS PRICING=macs

cell () {   # tag, budget(halved), ladder
  local tag=$1 t=$2 ladder=$3
  if [ -f "$OUT/$tag.json" ]; then echo "[B] $tag cached"; return; fi
  if [ ! -f "results/mp_calib/${tag}_wrapper.json" ]; then
    bash cls/mp_cell2.sh "$tag" "$t" "$ladder" "$NCAL" 16 2>&1 \
      | grep -E "^\[calib\]|^\[cell\]|^===|rho=|Error|Traceback|SystemExit|CUDA" || true
  fi
  if [ ! -f "results/mp_calib/${tag}_wrapper.json" ]; then
    echo "[B] $tag: NO TABLE — calibration failed, see above"; return
  fi
  echo "=== [$tag] eval n=$N $(date +%T)"
  SC_MP_TRACE=$OUT/${tag}_trace.json python -u cls/eval.py \
    --mode sc --sc_prec 8 --halve 1 --chunk_d 128 --stoc_len "$t" \
    --sc_ops mlp_fc1,mlp_fc2,qk,av,proj --qk_granularity per_row \
    --adaptive_mp 1 --adaptive_mp_table "results/mp_calib/${tag}_wrapper.json" \
    --adaptive_mp_ops "$ALLOPS" \
    --data_root "$D" --max_images "$N" --batch_size 16 --workers 8 --seed 0 \
    --out_json "$OUT/$tag.json" 2>&1 \
    | grep -viE "futurewarning|pynvml|warnings.warn" \
    | grep -E "^\[sc\] adaptive|^\[result\]|^\[eval\] per-image|Error|Traceback|SystemExit"
}

echo "########## Lane B: CLW=macs at the mild budgets  $(hostname) $(date +%T)"

# eff 128
CLW=macs           cell m64_base 64 $S64
CLW=macs PCF=0.02  cell m64_pc   64 $S64

# eff 192
CLW=macs           cell m96_base 96 $S96
CLW=macs PCF=0.02  cell m96_pc   96 $S96

echo "########## Lane B COMPLETE $(date +%T)"
