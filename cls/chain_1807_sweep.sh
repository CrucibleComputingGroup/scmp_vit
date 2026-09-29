#!/usr/bin/env bash
# gl1807 queue: wait for lane L to release the GPU, then run the eff-64 /
# eff-128 half of the ordering sweep end to end.
#
# BUDGETS is limited to "32 64" on purpose: eff-192 needs mask10_t96, which lane
# K is building on gl1805 right now. Two nodes calling build_hybrid_mask for the
# same tag would duplicate a 45-minute probe AND race on the same output file.
# eff-192 goes in a second pass once lane K has published the mask.
#
# The cells this actually adds (everything else is cache-hit):
#   hyb_mp_64iso   eff-128 hybrid MP with PCF=0, the recipe eff-128 is supposed
#                  to use. The only eff-128 hybrid-MP cell that exists is
#                  hyb_mppc_64iso2, which has protection ON -- fine as a cell,
#                  but it makes the eff-128 column of the figure use a different
#                  recipe from the row above it.
set -uo pipefail
cd /nfs/turbo/coe-nbleier/yjrcs/HPCA/vit_sc || exit 1

echo "[chain1807] waiting for lane L to release the GPU $(date +%T)"
while pgrep -f "cls/lane_l_denseladder.sh" > /dev/null; do sleep 30; done
echo "[chain1807] GPU free $(date +%T)"

BUDGETS="32 64" bash cls/sweep_orderings.sh
echo "=== chain_1807_sweep COMPLETE $(date +%T)"
