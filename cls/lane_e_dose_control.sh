#!/usr/bin/env bash
# Lane E (gl1807) — is the iso-compute hybrid gain about the RANKING or the DOSE?
#
# hyb_mppc_32iso beat the unmasked MP cell by +2.10 pt at matched true cycles.
# Two things could be doing that work and they have opposite implications:
#   (a) the 77-min knock-down probe picked the right 14 groups  -> ranking matters
#   (b) moving ANY 8% of MACs to INT7 and spending the freed cycles on the rest
#       -> the mask is a budget-reallocation trick and the probe is unnecessary
# At NON-iso budgets the ranked and random masks tied (-0.30, p=0.82), but that
# test was run where both arms were 8% under-budget, so it had little to resolve.
# Re-run it at iso.
#
# CONTROL. maskrand10_t32 is matched to mask10_t32 in group count AND per-op
# composition, so its MAC-weighted fraction is IDENTICAL (0.0792 both) and the
# same raised target 35 makes both arms iso. Only WHICH blocks within each op
# were picked differs.
#
# DOSE. Iso targets computed as T/(1-f) from each mask's true MAC-weighted
# fraction (f measured against uni_32's unpadded per-(op, block) MAC counts):
#   mask05_t32  f=0.0425 -> 33      mask10_t32  f=0.0792 -> 35
#   mask20_t32  f=0.1700 -> 39
# Masks at different --frac come off ONE sensitivity run and are strictly
# NESTED, so this is a genuine dose axis rather than three unrelated draws.
#
#   bash cls/lane_e_dose_control.sh
set -uo pipefail
cd /nfs/turbo/coe-nbleier/yjrcs/HPCA/vit_sc || exit 1
export TORCH_HOME=$PWD/.torch_cache
D=${IMAGENET_ROOT:-/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data}

N=${N:-1000}
NCAL=${NCAL:-128}
OUT=results/fig1k
S32=64,48,32,24,16
ALLOPS=qkv_proj,out_proj,mlp_fc1,mlp_fc2,qk,av
export QKG=per_row MPOPS=$ALLOPS PRICING=macs

uni_m () {   # tag, stoc_len, mask
  local tag=$1 sl=$2 mask=$3
  [ -f "$OUT/$tag.json" ] && { echo "[E] $tag cached"; return; }
  echo "=== [$tag] uniform sl=$sl + $(basename $mask) $(date +%T)"
  SC_MP_TRACE=$OUT/${tag}_trace.json python -u cls/eval.py \
    --mode sc --sc_prec 8 --halve 1 --chunk_d 128 --stoc_len "$sl" \
    --sc_ops mlp_fc1,mlp_fc2,qk,av,proj --qk_granularity per_row \
    --hybrid_config_json "$mask" \
    --data_root "$D" --max_images "$N" --batch_size 16 --workers 8 --seed 0 \
    --out_json "$OUT/$tag.json" 2>&1 \
    | grep -viE "futurewarning|pynvml|warnings.warn" \
    | grep -E "^\[result\]|Error|Traceback|SystemExit"
}

mp_m () {   # tag, stoc_len, mask, table_tag   (calibrates table_tag if absent)
  local tag=$1 sl=$2 mask=$3 tbl=$4
  [ -f "$OUT/$tag.json" ] && { echo "[E] $tag cached"; return; }
  if [ ! -f "results/mp_calib/${tbl}_wrapper.json" ]; then
    CLW=macs PCF=0.02 bash cls/mp_cell2.sh "$tbl" "$sl" $S32 "$NCAL" 16 2>&1 \
      | grep -E "^\[calib\] expected|^\[cell\]|Error|Traceback|SystemExit|CUDA" || true
  fi
  [ -f "results/mp_calib/${tbl}_wrapper.json" ] || { echo "[E] $tag NO TABLE"; return; }
  echo "=== [$tag] MP@$sl (table $tbl) + $(basename $mask) $(date +%T)"
  SC_MP_TRACE=$OUT/${tag}_trace.json python -u cls/eval.py \
    --mode sc --sc_prec 8 --halve 1 --chunk_d 128 --stoc_len "$sl" \
    --sc_ops mlp_fc1,mlp_fc2,qk,av,proj --qk_granularity per_row \
    --adaptive_mp 1 --adaptive_mp_table "results/mp_calib/${tbl}_wrapper.json" \
    --adaptive_mp_ops "$ALLOPS" --hybrid_config_json "$mask" \
    --data_root "$D" --max_images "$N" --batch_size 16 --workers 8 --seed 0 \
    --out_json "$OUT/$tag.json" 2>&1 \
    | grep -viE "futurewarning|pynvml|warnings.warn" \
    | grep -E "^\[result\]|Error|Traceback|SystemExit"
}

echo "########## Lane E: ranking control + dose at iso  $(hostname) $(date +%T)"

# ---- 1. RANKING CONTROL at iso (same f, same target, different blocks) ----
# The MP arm reuses hyb_mppc_32iso's table: it was calibrated at 35 with no
# knowledge of any mask, so it is the correct shared table for both arms.
uni_m hyb_uni_rand_32iso  35 results/hybrid/maskrand10_t32.json
mp_m  hyb_mppc_rand_32iso 35 results/hybrid/maskrand10_t32.json hyb_mppc_32iso

# ---- 2. DOSE at iso: 5% / 10% / 20% of groups on INT7 ----
uni_m hyb_uni_32iso_d05  33 results/hybrid/mask05_t32.json
uni_m hyb_uni_32iso_d20  39 results/hybrid/mask20_t32.json
mp_m  hyb_mppc_32iso_d05 33 results/hybrid/mask05_t32.json hyb_mppc_33iso
mp_m  hyb_mppc_32iso_d20 39 results/hybrid/mask20_t32.json hyb_mppc_39iso

echo "########## Lane E COMPLETE $(date +%T)"
