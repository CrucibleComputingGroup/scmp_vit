#!/usr/bin/env bash
# Lane A (gl1805) — the ISO-COMPUTE hybrid.
#
# THE BUG THIS FIXES. `--hybrid_config_json` routes 10% of (op, block) groups to
# INT7 and leaves the remaining 90% at the SAME stream length. The masked
# groups' SC cycles are therefore simply NOT SPENT -- they are not redistributed
# to the surviving SC groups. Measured off the traces at n=1000:
#
#   r32_pc -> hyb_mppc_32   SC MACs -8.3%   SC cycles spent  91.6%   top-1 -0.10
#   r64_pc -> hyb_mppc_64   SC MACs -10.0%  SC cycles spent  87.9%   top-1 -0.70
#
# So every published "hybrid vs unmasked" comparison charged the hybrid arm for
# a budget it never spent. At the eff-64 local slope (~0.26 pt/cycle) the 2.7
# missing cycles are worth ~0.7 pt, i.e. hyb_mppc_32's -0.10 pt is actually a
# WIN on the energy axis, not a loss.
#
# THE FIX. Raise the surviving groups' budget by 1/(1-f), f = the MAC-weighted
# fraction the mask removes, so total SC cycles match the unmasked cell:
#   t32: f=0.083 -> 32/0.917 = 34.9 -> 35
#   t64: f=0.100 -> 64/0.900 = 71.1 -> 71
# Both targets are straddled by their ladder (35 between 32 and 48; 71 between
# 64 and 96), which the swap search needs in order to trade at all.
#
# HONESTY CAVEAT, to be stated with any result from this lane: holding total SC
# cycles constant hands the hybrid arm its INT7 work for FREE. This is the
# OPTIMISTIC bound for the hybrid. Pricing an INT7 MAC at c SC-cycle-equivalents
# would set the target to (T - f*c)/(1-f); at c=7 that is 34 rather than 35 at
# t32, i.e. one cycle (~0.26 pt) of the measured gain. Report the bound as a
# bound.
#
#   bash cls/lane_a_isohybrid.sh
set -uo pipefail
cd /nfs/turbo/coe-nbleier/yjrcs/HPCA/vit_sc || exit 1
export TORCH_HOME=$PWD/.torch_cache
D=${IMAGENET_ROOT:-/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data}

N=${N:-1000}
NCAL=${NCAL:-128}
OUT=results/fig1k
mkdir -p "$OUT" results/logs

S32=64,48,32,24,16
S64=128,96,64,48,32
ALLOPS=qkv_proj,out_proj,mlp_fc1,mlp_fc2,qk,av

export QKG=per_row MPOPS=$ALLOPS PRICING=macs

# uniform SC + mask, at a raised stream length
uni_iso () {   # tag, stoc_len, mask
  local tag=$1 sl=$2 mask=$3
  if [ -f "$OUT/$tag.json" ]; then echo "[A] $tag cached"; return; fi
  echo "=== [$tag] uniform sl=$sl + $mask  n=$N $(date +%T)"
  SC_MP_TRACE=$OUT/${tag}_trace.json python -u cls/eval.py \
    --mode sc --sc_prec 8 --halve 1 --chunk_d 128 --stoc_len "$sl" \
    --sc_ops mlp_fc1,mlp_fc2,qk,av,proj --qk_granularity per_row \
    --hybrid_config_json "$mask" \
    --data_root "$D" --max_images "$N" --batch_size 16 --workers 8 --seed 0 \
    --out_json "$OUT/$tag.json" 2>&1 \
    | grep -viE "futurewarning|pynvml|warnings.warn" \
    | grep -E "^\[hybrid\]|^\[result\]|^\[eval\] per-image|Error|Traceback|SystemExit"
}

# MP (CLW=macs + 2% protected channels) calibrated at the RAISED target, + mask
mp_iso () {   # tag, budget(halved), ladder, mask
  local tag=$1 t=$2 ladder=$3 mask=$4
  if [ -f "$OUT/$tag.json" ]; then echo "[A] $tag cached"; return; fi
  if [ ! -f "results/mp_calib/${tag}_wrapper.json" ]; then
    CLW=macs PCF=0.02 bash cls/mp_cell2.sh "$tag" "$t" "$ladder" "$NCAL" 16 2>&1 \
      | grep -E "^\[calib\]|^\[cell\]|^===|rho=|Error|Traceback|SystemExit|CUDA" || true
  fi
  if [ ! -f "results/mp_calib/${tag}_wrapper.json" ]; then
    echo "[A] $tag: NO TABLE — calibration failed"; return
  fi
  echo "=== [$tag] MP@$t + $mask  n=$N $(date +%T)"
  SC_MP_TRACE=$OUT/${tag}_trace.json python -u cls/eval.py \
    --mode sc --sc_prec 8 --halve 1 --chunk_d 128 --stoc_len "$t" \
    --sc_ops mlp_fc1,mlp_fc2,qk,av,proj --qk_granularity per_row \
    --adaptive_mp 1 --adaptive_mp_table "results/mp_calib/${tag}_wrapper.json" \
    --adaptive_mp_ops "$ALLOPS" \
    --hybrid_config_json "$mask" \
    --data_root "$D" --max_images "$N" --batch_size 16 --workers 8 --seed 0 \
    --out_json "$OUT/$tag.json" 2>&1 \
    | grep -viE "futurewarning|pynvml|warnings.warn" \
    | grep -E "^\[hybrid\]|^\[sc\] adaptive|^\[result\]|^\[eval\] per-image|Error|Traceback|SystemExit"
}

echo "########## Lane A: iso-compute hybrid  $(hostname) $(date +%T)"

M32=results/hybrid/mask10_t32.json
M64=results/hybrid/mask10_t64.json

uni_iso hyb_uni_32iso  35 "$M32"
mp_iso  hyb_mppc_32iso 35 $S32 "$M32"
uni_iso hyb_uni_64iso  71 "$M64"
mp_iso  hyb_mppc_64iso 71 $S64 "$M64"

echo "########## Lane A COMPLETE $(date +%T)"
