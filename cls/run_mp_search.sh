#!/usr/bin/env bash
# MP budget-swap search at the halved-SC budgets used by the INT-vs-SC figure.
#
#   bash cls/run_mp_search.sh <TARGET_HALVED_SL> [N_SEARCH] [MAX_ITERS] [TAG] [FIXED_ATTN_SL]
#
# TARGET_HALVED_SL ∈ {32, 64, 96}  ->  effective levels {64, 128, 192}.
# (128 is the halved grid ceiling: nothing above it to trade with, so MP there
#  degenerates to uniform and is deliberately not searched.)
#
# Swaps run over proj / mlp_fc1 / mlp_fc2 only — those three have IDENTICAL
# per-block MACs (257*1024*4096), so an equal-|Δsl| swap is also equal-MAC and
# the budget is preserved exactly. qk/av are 15.9x cheaper; including them would
# silently move the MAC-weighted budget, which is why they stay fixed.
set -uo pipefail

T=${1:-64}
NS=${2:-32}
IT=${3:-8}
TAG=${4:-mp_t$T}
ATTN=${5:-$T}          # level qk/av are pinned to (default: same as target)

# The ladder MUST straddle the target. A zero-budget swap pairs one upgrade
# with one downgrade, so a target sitting on the ladder's lowest rung has
# nothing left to downgrade (and one on the highest has nothing to upgrade) —
# the search then finds no legal pair and exits at iteration 0 having moved
# nothing. This is why scmp_llm's canonical ladders always bracket their
# budget: int7 = [128,64,32] avg 64, len96 = [64,48,32] avg 48.
LEVELS=${LEVELS:-16,32,48,64,96,128}

VIT_SC=/nfs/turbo/coe-nbleier/yjrcs/HPCA/vit_sc
cd "$VIT_SC" || exit 1
export TORCH_HOME=$VIT_SC/.torch_cache
OUT=$VIT_SC/results/mp_search
mkdir -p "$OUT"

echo "### MP swap search | target_main_sl=$T (halved) = $((2*T)) levels | qk/av pinned $ATTN"
echo "### n_search=$NS max_iters=$IT | $(hostname) | $(date)"

# --op_min_levels is deliberately EMPTY. The script's defaults
# (proj=96, mlp_fc1=96, mlp_fc2=192) are non-halved values encoding the prior
# "fc2 is the fragile op"; translated to halved space (48/48/96) their mean is
# 64, which is infeasible at target 32 and leaves exactly zero slack at 64.
# Letting the ladder be the only constraint also keeps "fc2 must stay high" a
# FINDING of the search rather than an input to it.
python -u cls/experiments/mp_budget_swap_search.py \
  --sc_config all_ops \
  --target_main_sl "$T" \
  --levels "$LEVELS" \
  --search_ops proj,mlp_fc1,mlp_fc2 \
  --fixed_ops "qk=$ATTN,av=$ATTN" \
  --op_min_levels "" \
  --init_mode uniform_repair --init_level "$T" \
  --proxy raw_mse \
  --n_search "$NS" --max_iters "$IT" \
  --halve 1 --chunk_d 128 \
  --sc_prec 8 \
  --out_json "$OUT/$TAG.json" 2>&1 \
  | grep -viE "futurewarning|import pynvml"
