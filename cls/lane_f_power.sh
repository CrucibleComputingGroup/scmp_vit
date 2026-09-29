#!/usr/bin/env bash
# Lane F — INDEPENDENT replication of the eff-64 ladder at n=4000.
#
# WHY. The eff-64 ladder at n=1000 resolves the big steps easily but leaves the
# most contested one thin: hybrid-MP over MP is +2.10 pt with a paired CI of
# [+0.50, +3.70]. That excludes zero, so the ordering holds -- but a reviewer
# asking "how big is it" gets a factor-of-7 answer. Paired CI width scales as
# 1/sqrt(discordant) and discordance grows with n, so n=4000 should roughly
# halve it.
#
# WHY THIS IS A FRESH CALIBRATION, NOT A REUSE. The cached tables were built
# with --eval_exclude_n 2048, i.e. only the first 2048 images of shuffle(seed 0)
# were held out of the calibration pool. An n=4000 eval covers indices 0-3999,
# so 1952 of them WERE available to calibration -- the thresholds would have
# seen half the images they are scored on. Re-running eval alone cannot fix
# that. So lane F recalibrates at EXCL=4000, which keeps the split disjoint and
# makes this a genuine independent replication (new calibration draw + 4x the
# eval set) rather than a precision touch-up on the same draw.
#
# Consequence to keep in mind when reading it: BOTH the calibration draw and the
# eval set differ from the n=1000 ladder, so a disagreement cannot be attributed
# to one or the other. The n=1000 numbers stay the citable ones; this lane says
# whether the ORDERINGS survive an independent draw at 4x the sample.
#
#   bash cls/lane_f_power.sh
set -uo pipefail
cd /nfs/turbo/coe-nbleier/yjrcs/HPCA/vit_sc || exit 1
export TORCH_HOME=$PWD/.torch_cache
D=${IMAGENET_ROOT:-/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data}

N=${N:-4000}
NCAL=${NCAL:-128}
OUT=results/fig4k
mkdir -p "$OUT"
S32=64,48,32,24,16
ALLOPS=qkv_proj,out_proj,mlp_fc1,mlp_fc2,qk,av
M32=results/hybrid/mask10_t32.json

export QKG=per_row MPOPS=$ALLOPS PRICING=macs EXCL=4000

run () {   # tag, stoc_len, [extra eval args...]
  local tag=$1 sl=$2; shift 2
  [ -f "$OUT/$tag.json" ] && { echo "[F] $tag cached"; return; }
  echo "=== [$tag] n=$N $(date +%T)"
  SC_MP_TRACE=$OUT/${tag}_trace.json python -u cls/eval.py \
    --mode sc --sc_prec 8 --halve 1 --chunk_d 128 --stoc_len "$sl" \
    --sc_ops mlp_fc1,mlp_fc2,qk,av,proj --qk_granularity per_row "$@" \
    --data_root "$D" --max_images "$N" --batch_size 16 --workers 8 --seed 0 \
    --out_json "$OUT/$tag.json" 2>&1 \
    | grep -viE "futurewarning|pynvml|warnings.warn" \
    | grep -E "^\[result\]|Error|Traceback|SystemExit"
}

calib () {   # tag, budget   (EXCL=4000 tables get their own -x4k tag)
  local tag=$1 t=$2
  [ -f "results/mp_calib/${tag}_wrapper.json" ] && { echo "[F] table $tag cached"; return; }
  CLW=macs PCF=0.02 bash cls/mp_cell2.sh "$tag" "$t" $S32 "$NCAL" 16 2>&1 \
    | grep -E "^\[calib\] expected|^\[calib\] calib pool|^\[cell\]|Error|Traceback|SystemExit|CUDA" || true
}

echo "########## Lane F: independent n=$N replication  $(hostname) $(date +%T)"

calib r32_pc_x4k      32
calib hyb_mppc_32iso_x4k 35

run uni_32 32
run r32_pc 32 --adaptive_mp 1 \
    --adaptive_mp_table results/mp_calib/r32_pc_x4k_wrapper.json \
    --adaptive_mp_ops "$ALLOPS"
run hyb_uni_32iso 35 --hybrid_config_json "$M32"
run hyb_mppc_32iso 35 --adaptive_mp 1 \
    --adaptive_mp_table results/mp_calib/hyb_mppc_32iso_x4k_wrapper.json \
    --adaptive_mp_ops "$ALLOPS" --hybrid_config_json "$M32"

echo "########## Lane F COMPLETE $(date +%T)"
