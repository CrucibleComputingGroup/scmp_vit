#!/usr/bin/env bash
# Calibration-seed replication for the MP arm at the MILD budgets.
#
# WHY. The figure's MP arm is a 4-seed mean at eff-64 (r32_pc) but a SINGLE draw
# at eff-128 (m64_pc) and eff-192 (m96_pc). Calibration spread at n_calib=128 is
# 0.4-0.6 pt, which is the entire size of every effect at the mild end -- so the
# two questions the figure currently invites
#     "MP@192 is below MP@128"      (measured +0.00, p=1.00)
#     "MP@192 is below uniform@192" (measured -0.20, p=0.86)
# are both un-answerable from one draw, even though both are statistically ties.
# This is the same defect that made the hybrid arm look like it dipped from
# eff-64 to eff-128; it was fixed there and not here.
#
# ALSO TESTED: whether PROTECTION HURTS at the mild budgets. m96_base (MP alone)
# 0.8530 sits ABOVE uni_96 0.8500, while m96_pc (MP + 2% protected channels)
# 0.8480 sits below it -- i.e. it is the protection, not MP, that puts the arm
# under uniform at eff-192. Single draw, -0.50, p=0.40, so unresolved. The
# mechanism predicts it: protection buys its way out of outlier-driven scale
# coarseness, which only binds when the stream is short enough for the coarse
# scale to dominate. At eff-192 there is no such problem, so pinning 1.9% of
# MACs at the ladder ceiling and shrinking everyone else's budget to pay for it
# is pure cost. m96_base seeds test that.
#
#   CELLS="m96_pc:96:S96 m96_base:96:S96" bash cls/lane_seeds_mp.sh
# Each entry is <base_tag>:<budget>:<ladder-var>. PCF is inferred from the tag
# (_pc => 0.02, _base => 0).
set -uo pipefail
cd /nfs/turbo/coe-nbleier/yjrcs/HPCA/vit_sc || exit 1
export TORCH_HOME=$PWD/.torch_cache
D=${IMAGENET_ROOT:-/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data}

N=${N:-1000}
NCAL=${NCAL:-128}
OUT=results/fig1k
S64=128,96,64,48,32
S96=128,112,96,80,64,48
ALLOPS=qkv_proj,out_proj,mlp_fc1,mlp_fc2,qk,av
SEEDS=${SEEDS:-"7 23 41"}
export QKG=per_row MPOPS=$ALLOPS PRICING=macs

echo "########## MP seed replication: ${CELLS:?set CELLS}  $(hostname) $(date +%T)"

for ENTRY in $CELLS; do
  BASE=${ENTRY%%:*}; REST=${ENTRY#*:}; T=${REST%%:*}; LADVAR=${REST##*:}
  LADDER=$(eval echo "\$$LADVAR")
  case "$BASE" in *_pc) PCFV=0.02 ;; *) PCFV=0 ;; esac
  for SEED in $SEEDS; do
    TAG="${BASE}_s${SEED}"
    [ -f "$OUT/$TAG.json" ] && { echo "[seed] $TAG cached"; continue; }
    if [ ! -f "results/mp_calib/${TAG}_wrapper.json" ]; then
      CLW=macs PCF=$PCFV CALIB_SEED=$SEED \
        bash cls/mp_cell2.sh "$TAG" "$T" "$LADDER" "$NCAL" 16 2>&1 \
        | grep -E "^\[calib\] expected|^\[cell\]|Error|Traceback|SystemExit|CUDA" || true
    fi
    [ -f "results/mp_calib/${TAG}_wrapper.json" ] || { echo "[seed] $TAG NO TABLE"; continue; }
    echo "=== [$TAG] budget=$T pcf=$PCFV seed=$SEED n=$N $(date +%T)"
    SC_MP_TRACE=$OUT/${TAG}_trace.json python -u cls/eval.py \
      --mode sc --sc_prec 8 --halve 1 --chunk_d 128 --stoc_len "$T" \
      --sc_ops mlp_fc1,mlp_fc2,qk,av,proj --qk_granularity per_row \
      --adaptive_mp 1 --adaptive_mp_table "results/mp_calib/${TAG}_wrapper.json" \
      --adaptive_mp_ops "$ALLOPS" \
      --data_root "$D" --max_images "$N" --batch_size 16 --workers 8 --seed 0 \
      --out_json "$OUT/$TAG.json" 2>&1 \
      | grep -viE "futurewarning|pynvml|warnings.warn" \
      | grep -E "^\[result\]|Error|Traceback|SystemExit"
  done
done
echo "########## seed replication COMPLETE $(date +%T)"
