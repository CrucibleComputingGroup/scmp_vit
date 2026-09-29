#!/usr/bin/env bash
# Lane H — decompose the +2.10 pt that the iso hybrid gains over plain MP.
#
# Going from r32_pc to hyb_mppc_32iso changes TWO things at once:
#   (a) 7.9% of MACs move from SC (64 levels @ ~32 cycles) to INT7 (128 levels)
#   (b) the surviving 92% get their budget raised 32 -> 35, paid for by (a)
# Both are required for the cell to be iso-compute, so the +2.10 is correctly
# attributed to the pair -- but it says nothing about which one carries it.
#
# This cell isolates (b): the SAME table, at the SAME raised target 35, with NO
# mask. It is deliberately NOT iso -- it spends ~9% more true cycles than
# r32_pc, since nothing pays for the raise -- so it is a MECHANISM probe, never
# a headline. Read it as: "if you simply gave plain MP the extra cycles instead
# of earning them, how much of the +2.10 would you get?" Whatever is left over
# is attributable to the INT format on the masked groups, and lane G's
# level-matched INT6 arm tests that half independently.
#
# No calibration: reuses hyb_mppc_32iso's table, so the draw is held fixed.
#
#   bash cls/lane_h_decomp.sh
set -uo pipefail
cd /nfs/turbo/coe-nbleier/yjrcs/HPCA/vit_sc || exit 1
export TORCH_HOME=$PWD/.torch_cache
D=${IMAGENET_ROOT:-/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data}

N=${N:-1000}
OUT=results/fig1k
ALLOPS=qkv_proj,out_proj,mlp_fc1,mlp_fc2,qk,av

echo "########## Lane H: MP@35 unmasked (over-budget mechanism probe)  $(date +%T)"

TAG=r35_pc_overbudget
if [ -f "$OUT/$TAG.json" ]; then
  echo "[H] $TAG cached"
else
  SC_MP_TRACE=$OUT/${TAG}_trace.json python -u cls/eval.py \
    --mode sc --sc_prec 8 --halve 1 --chunk_d 128 --stoc_len 35 \
    --sc_ops mlp_fc1,mlp_fc2,qk,av,proj --qk_granularity per_row \
    --adaptive_mp 1 --adaptive_mp_table results/mp_calib/hyb_mppc_32iso_wrapper.json \
    --adaptive_mp_ops "$ALLOPS" \
    --data_root "$D" --max_images "$N" --batch_size 16 --workers 8 --seed 0 \
    --out_json "$OUT/$TAG.json" 2>&1 \
    | grep -viE "futurewarning|pynvml|warnings.warn" \
    | grep -E "^\[result\]|Error|Traceback|SystemExit"
fi

echo "########## Lane H COMPLETE $(date +%T)"
