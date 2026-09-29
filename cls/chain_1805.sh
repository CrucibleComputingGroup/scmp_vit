#!/usr/bin/env bash
# gl1805 queue: wave-1 ablation -> wave-2 (everything at qk per_row).
# Exclusive_Process GPUs, so stages serialize rather than race for the device.
set -uo pipefail
cd /nfs/turbo/coe-nbleier/yjrcs/HPCA/vit_sc || exit 1

echo "[chain1805] waiting for sweep_t32 to release the GPU $(date +%T)"
while pgrep -f "cls/sweep_t32.sh" > /dev/null; do sleep 30; done
echo "[chain1805] GPU free $(date +%T)"

bash cls/sweep_wave2.sh 2>&1
echo "=== chain_1805 COMPLETE $(date +%T)"
