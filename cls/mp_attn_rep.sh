#!/usr/bin/env bash
# Paired replicate of the effective-128 cell with qk/av ADDED to the MP set.
#
#   bash cls/mp_attn_rep.sh <SEED...>
#
# Uses the same calibration seeds as the proj+MLP-only baseline
# (results/mp_eval/r64_amax_s*.json), so the only difference between the two
# arms is the operator set — the comparison carries no calibration-sample
# noise. Attention MACs are in the budget either way, so it stays iso-compute.
set -uo pipefail
VIT_SC=/nfs/turbo/coe-nbleier/yjrcs/HPCA/vit_sc
for S in "$@"; do
  MPOPS=qkv_proj,out_proj,mlp_fc1,mlp_fc2,qk,av CALIB_SEED="$S" \
    bash "$VIT_SC/cls/mp_cell.sh" "a64_amax_s$S" 64 128,96,64,48,32 amax 64 256 2048
done
