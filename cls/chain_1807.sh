#!/usr/bin/env bash
# gl1807 queue. The GPUs are Exclusive_Process, so stages must serialize;
# this waits out the running sweep rather than racing it for the device.
set -uo pipefail
cd /nfs/turbo/coe-nbleier/yjrcs/HPCA/vit_sc || exit 1

echo "[chain1807] waiting for sweep_t64 to release the GPU $(date +%T)"
while pgrep -f "cls/sweep_t64.sh" > /dev/null; do sleep 30; done
echo "[chain1807] GPU free $(date +%T)"

# Batch-size probe FIRST: at 7 img/s a 50k cell is 2.1 h and the full figure
# does not fit the remaining allocation. Uniform SC is batch-invariant in
# accuracy (per_row derives each row's range from that row alone), so if
# occupancy is what is limiting, this is a free multiplier on every cell.
bash cls/probe_batch.sh 2>&1

echo "=== chain_1807 STAGE1 COMPLETE $(date +%T)"
