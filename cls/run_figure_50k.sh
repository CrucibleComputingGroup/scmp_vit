#!/usr/bin/env bash
# Final figure cells on the FULL ImageNet val set.
#
#   ARM=fixed  bash cls/run_figure_50k.sh    # FP + INT + uniform SC  (no MP)
#   ARM=mp     bash cls/run_figure_50k.sh    # MP cells
#   ARM=hybrid bash cls/run_figure_50k.sh    # uniform+mask and MP+mask
#
# Split by arm so the two GPUs can take independent halves; the arms share no
# state beyond the calibration tables.
#
# DISJOINTNESS. On a 2048-image subset the calibrator's --eval_exclude_n keeps
# the two sets apart, because the eval subset is the first N of a known
# shuffle. On a full-val run it is not, so MP cells pass
# --exclude_calib_images 1 and eval.py subtracts the exact indices the table
# records. That makes the MP arms score ~49.7k images while the FP/INT/uniform
# arms score all 50k — the arms are NOT on identical image sets, and the
# collector reports each cell's own n so the Wilson intervals stay honest.
set -uo pipefail
cd /nfs/turbo/coe-nbleier/yjrcs/HPCA/vit_sc || exit 1
export TORCH_HOME=$PWD/.torch_cache
D=${IMAGENET_ROOT:-/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data}

ARM=${ARM:-fixed}
BS=${BS:-16}                 # set from cls/probe_batch.sh
N=${N:-50000}
QKG=${QKG:-per_row}
OUT=results/figure
mkdir -p $OUT

run_sc () {   # tag, stoc_len, extra args...
  local tag=$1 sl=$2; shift 2
  [ -f "$OUT/$tag.json" ] && { echo "[50k] $tag cached"; return; }
  echo "=== [$tag] sl=$sl bs=$BS n=$N $(date +%T)"
  SC_MP_TRACE=$OUT/${tag}_trace.json python -u cls/eval.py \
    --mode sc --sc_prec 8 --halve 1 --chunk_d 128 --stoc_len "$sl" \
    --sc_ops mlp_fc1,mlp_fc2,qk,av,proj --qk_granularity "$QKG" \
    --data_root "$D" --max_images "$N" --batch_size "$BS" --workers 8 --seed 0 \
    "$@" --out_json "$OUT/$tag.json" 2>&1 \
    | grep -viE "futurewarning|pynvml|warnings.warn" \
    | grep -E "^\[hybrid\]|^\[data\]|^\[result\]|Error|Traceback"
}

case "$ARM" in
fixed)
  if [ ! -f "$OUT/fp.json" ]; then
    python -u cls/eval.py --mode fp --data_root "$D" --max_images "$N" \
      --batch_size 64 --workers 8 --seed 0 --out_json "$OUT/fp.json" 2>&1 \
      | grep -E "^\[result\]|Error"
  fi
  for B in 8 7 6 4; do
    [ -f "$OUT/int${B}_asymm.json" ] && continue
    echo "=== [int${B}_asymm] $(date +%T)"
    python -u cls/eval.py --mode int --int_bits $B --int_sym 0 \
      --int_chunk 128 --int_attn_scope per_row \
      --sc_ops mlp_fc1,mlp_fc2,qk,av,proj \
      --data_root "$D" --max_images "$N" --batch_size 64 --workers 8 --seed 0 \
      --out_json "$OUT/int${B}_asymm.json" 2>&1 \
      | grep -E "^\[result\]|Error|Traceback"
  done
  for SL in 128 96 64 32; do run_sc "uni_$SL" $SL; done
  ;;
mp)
  # MP_CELLS: "tag:stoc_len:wrapper" triples, set by the caller from the
  # screening winner at each budget.
  for cell in ${MP_CELLS:?set MP_CELLS="tag:sl:wrapper ..."}; do
    IFS=: read -r tag sl wrap <<< "$cell"
    run_sc "$tag" "$sl" --adaptive_mp 1 --adaptive_mp_table "$wrap" \
      --adaptive_mp_ops "${MPOPS:-qkv_proj,out_proj,mlp_fc1,mlp_fc2,qk,av}" \
      --exclude_calib_images 1
  done
  ;;
hybrid)
  for cell in ${HYB_CELLS:?set HYB_CELLS="tag:sl:mask[:wrapper] ..."}; do
    IFS=: read -r tag sl mask wrap <<< "$cell"
    if [ -n "${wrap:-}" ]; then
      run_sc "$tag" "$sl" --hybrid_config_json "$mask" \
        --adaptive_mp 1 --adaptive_mp_table "$wrap" \
        --adaptive_mp_ops "${MPOPS:-qkv_proj,out_proj,mlp_fc1,mlp_fc2,qk,av}" \
        --exclude_calib_images 1
    else
      run_sc "$tag" "$sl" --hybrid_config_json "$mask"
    fi
  done
  ;;
*) echo "unknown ARM=$ARM"; exit 1;;
esac

echo "=== run_figure_50k ARM=$ARM COMPLETE $(date +%T)"
