#!/usr/bin/env bash
# Knob ablation at the effective-128 budget (halved 64) — the budget where MP
# already WON before 2026-07-29 (+0.94 over 4 calibration seeds at
# W_g=uniform; W_g=macs was worse there, -0.38 paired). So these cells hold
# W_g=uniform and ask whether the new knobs add on top of a working cell.
#
# Uniform comparator: results/int_vs_sc sc_halved64 = 0.8330 for the per_head
# cells, and the per_row uniform cell for the attention cells.
set -uo pipefail
cd /nfs/turbo/coe-nbleier/yjrcs/HPCA/vit_sc || exit 1

NCAL=${NCAL:-256}
NEVAL=${NEVAL:-2048}
SPARSE=128,96,64,48,32
DENSE=128,112,96,80,64,48,40,32
ALLOPS=qkv_proj,out_proj,mlp_fc1,mlp_fc2,qk,av

# uniform SC at per_row qk — the matched comparator for the attention cells
# below. The published sc_halved64 = 0.8330 is a per_head run and cannot be
# differenced against a per_row MP cell.
D=${IMAGENET_ROOT:-/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data}
export TORCH_HOME=$PWD/.torch_cache
for SL in 64 32; do
  OUTJ=results/mp_eval/uni_perrow_$SL.json
  if [ ! -f "$OUTJ" ]; then
    SC_MP_TRACE=results/mp_eval/uni_perrow_${SL}_trace.json python -u cls/eval.py \
      --mode sc --sc_prec 8 --halve 1 --chunk_d 128 --stoc_len $SL \
      --sc_ops mlp_fc1,mlp_fc2,qk,av,proj --qk_granularity per_row \
      --data_root "$D" --max_images "$NEVAL" --batch_size 16 --workers 4 --seed 0 \
      --out_json "$OUTJ" 2>&1 \
      | grep -viE "futurewarning|pynvml|xFormers|warnings.warn" \
      | grep -E "^\[result\]|Error|Traceback"
  fi
done

bash cls/mp_cell2.sh s64_base 64 $SPARSE $NCAL $NEVAL
GLK=3 bash cls/mp_cell2.sh s64_glk 64 $DENSE $NCAL $NEVAL
ESC=2.0 bash cls/mp_cell2.sh s64_esc 64 $SPARSE $NCAL $NEVAL
PCF=0.02 bash cls/mp_cell2.sh s64_pc2 64 $SPARSE $NCAL $NEVAL
QKG=per_row MPOPS=$ALLOPS bash cls/mp_cell2.sh s64_attn 64 $SPARSE $NCAL $NEVAL
GLK=3 ESC=2.0 PCF=0.02 QKG=per_row MPOPS=$ALLOPS \
  bash cls/mp_cell2.sh s64_all 64 $DENSE $NCAL $NEVAL

echo "=== sweep_t64 COMPLETE $(date +%T)"
