#!/usr/bin/env bash
# The three controls, in ONE process. The standalone versions died in the
# 06:15 GPU-contention episode (each was an independent waiter; when the main
# queue was killed to fix the pricing bug they all fired at once onto an
# Exclusive_Process device). Nothing here waits on anything.
set -uo pipefail
cd /nfs/turbo/coe-nbleier/yjrcs/HPCA/vit_sc || exit 1
export TORCH_HOME=$PWD/.torch_cache
D=${IMAGENET_ROOT:-/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data}
N=1000; OUT=results/fig1k

run () { local tag=$1; shift
  [ -f "$OUT/$tag.json" ] && { echo "[ctl] $tag cached"; return; }
  echo "=== [$tag] $(date +%T)"
  SC_MP_TRACE=$OUT/${tag}_trace.json python -u cls/eval.py \
    --mode sc --sc_prec 8 --halve 1 --chunk_d 128 \
    --sc_ops mlp_fc1,mlp_fc2,qk,av,proj \
    --data_root "$D" --max_images $N --batch_size 16 --workers 8 --seed 0 \
    "$@" --out_json "$OUT/$tag.json" 2>&1 \
    | grep -viE "futurewarning|pynvml|warnings.warn" \
    | grep -E "^\[sc\] protected|^\[hybrid\]|^\[result\]|Error|Traceback|SystemExit"
}

# 1. per_head uniform — the decomposition chain's starting point
for SL in 32 64; do run uni_head_$SL --stoc_len $SL --qk_granularity per_head; done

# 2. protection WITHOUT MP — is it an MP feature or a general SC fix?
#    base 31 also run: 2% at 64 + 98% at 31 lands UNDER the uniform 32, so a
#    win there is conservative.
for B in 32 31; do
  run uniprot_$B --stoc_len $B --qk_granularity per_row \
      --protect_from_table results/mp_calib/r32_pc.json
done

# 3. random 10% mask matched per-op — did the sensitivity RANKING matter?
for T in 32 64; do
  run hyb_uni_rand_$T --stoc_len $T --qk_granularity per_row \
      --hybrid_config_json results/hybrid/maskrand10_t$T.json
done
echo "=== run_controls COMPLETE $(date +%T)"
