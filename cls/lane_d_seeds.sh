#!/usr/bin/env bash
# Lane D (gl1805) — calibration-seed replication of the new headline cell.
#
# hyb_mppc_32iso (MP@35 + 2% protected channels + 10% INT mask, iso-compute)
# scored 0.8490, i.e. +6.60 vs uniform SC and +2.10 over the unmasked MP cell.
# That is ONE calibration draw at n_calib=128, where MP_REPORT measured the
# calibration-sample sd at ~1 pt -- the same size as the +2.10 increment this
# cell is claimed on. A single draw is not a result; three more seeds are.
#
#   bash cls/lane_d_seeds.sh
set -uo pipefail
cd /nfs/turbo/coe-nbleier/yjrcs/HPCA/vit_sc || exit 1
export TORCH_HOME=$PWD/.torch_cache
D=${IMAGENET_ROOT:-/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data}

N=${N:-1000}
NCAL=${NCAL:-128}
OUT=results/fig1k
S32=64,48,32,24,16
ALLOPS=qkv_proj,out_proj,mlp_fc1,mlp_fc2,qk,av
M32=results/hybrid/mask10_t32.json

export QKG=per_row MPOPS=$ALLOPS PRICING=macs

echo "########## Lane D: seed replication of hyb_mppc_32iso  $(hostname) $(date +%T)"

for SEED in 7 23 41; do
  TAG=hyb_mppc_32iso_s$SEED
  [ -f "$OUT/$TAG.json" ] && { echo "[D] $TAG cached"; continue; }
  if [ ! -f "results/mp_calib/${TAG}_wrapper.json" ]; then
    CLW=macs PCF=0.02 CALIB_SEED=$SEED \
      bash cls/mp_cell2.sh "$TAG" 35 $S32 "$NCAL" 16 2>&1 \
      | grep -E "^\[calib\] expected|^\[cell\]|Error|Traceback|SystemExit|CUDA" || true
  fi
  [ -f "results/mp_calib/${TAG}_wrapper.json" ] || { echo "[D] $TAG NO TABLE"; continue; }
  echo "=== [$TAG] eval n=$N $(date +%T)"
  SC_MP_TRACE=$OUT/${TAG}_trace.json python -u cls/eval.py \
    --mode sc --sc_prec 8 --halve 1 --chunk_d 128 --stoc_len 35 \
    --sc_ops mlp_fc1,mlp_fc2,qk,av,proj --qk_granularity per_row \
    --adaptive_mp 1 --adaptive_mp_table "results/mp_calib/${TAG}_wrapper.json" \
    --adaptive_mp_ops "$ALLOPS" --hybrid_config_json "$M32" \
    --data_root "$D" --max_images "$N" --batch_size 16 --workers 8 --seed 0 \
    --out_json "$OUT/$TAG.json" 2>&1 \
    | grep -viE "futurewarning|pynvml|warnings.warn" \
    | grep -E "^\[result\]|Error|Traceback|SystemExit"
done

echo "########## Lane D COMPLETE $(date +%T)"
