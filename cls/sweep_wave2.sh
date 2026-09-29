#!/usr/bin/env bash
# Wave 2: everything at qk per_row, plus the metric axis and the third budget.
#
# Wave 1 turned up a bigger effect than any MP knob: switching Q@K^T from
# per_head to per_row quantization is worth +3.22 pt at halved 32 and +0.59 at
# halved 64 on the UNIFORM arm, with no MP involved. So per_head cells are not
# the frontier and per_head uniform is not the comparator. Every cell here runs
# per_row and is differenced against uni_perrow_{32,64,96,128}.
#
# The other axis is the dispatch metric. Wave 1's rho diagnostic said it, not
# the ladder, is what limits MP: mlp_fc2 (a third of all MACs) ranked rows at
# rho=+0.13, av at -0.18 ranked them backwards. --metric-select auto scores
# amax / l2 / crest signed against the SAME objective the allocator uses.
set -uo pipefail
cd /nfs/turbo/coe-nbleier/yjrcs/HPCA/vit_sc || exit 1
export TORCH_HOME=$PWD/.torch_cache
D=${IMAGENET_ROOT:-/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data}

NCAL=${NCAL:-256}
NEVAL=${NEVAL:-2048}
S32=64,48,32,24,16
S64=128,96,64,48,32
S96=128,112,96,80,64,48
D32=64,56,48,40,32,24,20,16
ALLOPS=qkv_proj,out_proj,mlp_fc1,mlp_fc2,qk,av
export QKG=per_row MPOPS=$ALLOPS

# --- matched uniform comparators still missing -------------------------------
for SL in 96 128; do
  OUTJ=results/mp_eval/uni_perrow_$SL.json
  [ -f "$OUTJ" ] && continue
  SC_MP_TRACE=results/mp_eval/uni_perrow_${SL}_trace.json python -u cls/eval.py \
    --mode sc --sc_prec 8 --halve 1 --chunk_d 128 --stoc_len $SL \
    --sc_ops mlp_fc1,mlp_fc2,qk,av,proj --qk_granularity per_row \
    --data_root "$D" --max_images "$NEVAL" --batch_size 16 --workers 4 --seed 0 \
    --out_json "$OUTJ" 2>&1 \
    | grep -viE "futurewarning|pynvml|warnings.warn" \
    | grep -E "^\[result\]|Error|Traceback"
done

# Group ladders are deliberately absent: wave 1 put them 1.66 pt DOWN at this
# budget (occupancy-ranked rung selection capped mlp_fc1 at 24 cycles). The
# rule is fixed but unproven, and re-testing a knob that failed is worse value
# than the two axes below.
#
# The ESCAPE GATE is in, because wave 1 says it is the one knob that works:
# s32_esc 0.7681 vs s32_base 0.7573, +1.08 pt. It is now budget-COMPENSATED in
# the calibrator (the gate fires after the allocator has spent, so an ungated
# solve overshoots — s32_esc realized 32.81 against a nominal 32), so these
# cells are iso-compute rather than buying accuracy with cycles.

# --- eff 64 only; eff 128/192 run on the other node (sweep_wave2b.sh) -------
# per_row amax baseline is wave 1's s32_attn (same ops, same scope), so these
# isolate one knob each.
CLW=macs ESC=2.0             bash cls/mp_cell2.sh p32_esc      32 $S32 $NCAL $NEVAL
CLW=macs METRIC=auto         bash cls/mp_cell2.sh p32_auto     32 $S32 $NCAL $NEVAL
CLW=macs METRIC=auto ESC=2.0 bash cls/mp_cell2.sh p32_auto_esc 32 $S32 $NCAL $NEVAL

echo "=== sweep_wave2 COMPLETE $(date +%T)"
