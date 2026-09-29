#!/usr/bin/env bash
# Lane G — is the iso-hybrid gain the DATAPATH or just extra levels?
#
# THE CONFOUND. At the iso target 35 the SC groups carry 2*35 = 70 representable
# levels, while the masked groups run INT7 = 128. So the masked 8% of MACs got
# ~1.8x the resolution of everything else, and `hyb_mppc_32iso`'s +2.10 pt over
# `r32_pc` could be that alone -- "we upgraded 8% of the model" -- rather than
# any property of a mixed SC/INT datapath.
#
# THE CONTROL. Re-run the same mask at INT6 = 64 levels, which sits just BELOW
# the SC arm's 70. If the gain survives at INT6 it is about the datapath: a
# uniform fixed-point grid beats a stochastic stream of equal-or-more levels on
# those groups, which is the actual hybrid claim. If it collapses, the honest
# statement is "10% of groups at higher precision", and INT7-vs-SC level parity
# has to be reported alongside every hybrid number.
#
# Both arms reuse the tables already calibrated at 35 -- the mask changes only
# the backend of the masked groups, not the SC allocation, so no recalibration
# is needed and the calibration draw is held fixed against lane A.
#
#   bash cls/lane_g_levelmatch.sh
set -uo pipefail
cd /nfs/turbo/coe-nbleier/yjrcs/HPCA/vit_sc || exit 1
export TORCH_HOME=$PWD/.torch_cache
D=${IMAGENET_ROOT:-/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data}

N=${N:-1000}
OUT=results/fig1k
ALLOPS=qkv_proj,out_proj,mlp_fc1,mlp_fc2,qk,av
M6=results/hybrid/mask10_t32_int6.json

run () {   # tag, stoc_len, [extra args...]
  local tag=$1 sl=$2; shift 2
  [ -f "$OUT/$tag.json" ] && { echo "[G] $tag cached"; return; }
  echo "=== [$tag] n=$N $(date +%T)"
  SC_MP_TRACE=$OUT/${tag}_trace.json python -u cls/eval.py \
    --mode sc --sc_prec 8 --halve 1 --chunk_d 128 --stoc_len "$sl" \
    --sc_ops mlp_fc1,mlp_fc2,qk,av,proj --qk_granularity per_row "$@" \
    --data_root "$D" --max_images "$N" --batch_size 16 --workers 8 --seed 0 \
    --out_json "$OUT/$tag.json" 2>&1 \
    | grep -viE "futurewarning|pynvml|warnings.warn" \
    | grep -E "^\[hybrid\]|^\[result\]|Error|Traceback|SystemExit"
}

echo "########## Lane G: level-matched INT6 hybrid  $(hostname) $(date +%T)"

run hyb_uni_32iso_i6  35 --hybrid_config_json "$M6"
run hyb_mppc_32iso_i6 35 --adaptive_mp 1 \
    --adaptive_mp_table results/mp_calib/hyb_mppc_32iso_wrapper.json \
    --adaptive_mp_ops "$ALLOPS" --hybrid_config_json "$M6"

echo "########## Lane G COMPLETE $(date +%T)"
