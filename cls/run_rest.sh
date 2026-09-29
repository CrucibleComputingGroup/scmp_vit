#!/usr/bin/env bash
# ONE process, stages strictly sequential. No background waiters.
#
# The previous arrangement chained four independently-launched queues, each
# polling `pgrep` for the previous one to exit. That is a race: killing the
# main queue to fix a bug released three waiters at once, they all grabbed the
# Exclusive_Process GPU, and every calibration in the restarted queue died with
# cudaErrorDevicesUnavailable — which surfaced only as "NO TABLE", because
# mp() sends calibration output to /dev/null. Sequential calls in a single
# process cannot race; the waiter loops inside each script become no-ops.
#
# Phase 0 (masks) and Phase A (references) are cached and skip in seconds.
set -uo pipefail
cd /nfs/turbo/coe-nbleier/yjrcs/HPCA/vit_sc || exit 1

echo "########## run_all_1k (Phase 0/A cached -> B, C, D) $(date +%T)"
bash cls/run_all_1k.sh

echo "########## mask-ranking control $(date +%T)"
bash cls/run_mask_control.sh

echo "########## protection-without-MP control $(date +%T)"
bash cls/run_protect_control.sh

echo "########## per_head decomposition references $(date +%T)"
bash cls/run_decomp.sh

# Repair pass: run_all_1k skips any cell whose result JSON already exists, so a
# cell whose artifacts were removed after it ran gets rebuilt here. Its
# calibration table is cached, so this is an eval only.
echo "########## repair pass $(date +%T)"
bash cls/run_all_1k.sh

echo "=== run_rest ALL COMPLETE $(date +%T)"
