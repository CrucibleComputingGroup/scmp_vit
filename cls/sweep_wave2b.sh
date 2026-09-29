#!/usr/bin/env bash
# Wave 2b: eff 128 and eff 192, all at qk per_row (gl1807 half of wave 2).
#
# The eff-128 per_row amax BASELINE is built here rather than inherited: the
# wave-1 s64_attn cell was cut when the t64 queue was trimmed, because three
# more per_head ablation cells were not worth 45 minutes once per_row turned
# out to dominate. Without a per_row amax cell at this budget there is nothing
# to attribute the metric/gate deltas against.
set -uo pipefail
cd /nfs/turbo/coe-nbleier/yjrcs/HPCA/vit_sc || exit 1

NCAL=${NCAL:-256}
NEVAL=${NEVAL:-2048}
S64=128,96,64,48,32
S96=128,112,96,80,64,48
export QKG=per_row MPOPS=qkv_proj,out_proj,mlp_fc1,mlp_fc2,qk,av

# baselines (amax, no gate) — one per budget
bash cls/mp_cell2.sh p64_attn 64 $S64 $NCAL $NEVAL
bash cls/mp_cell2.sh p96_attn 96 $S96 $NCAL $NEVAL

# the wave-1 winner's recipe carried over
METRIC=auto ESC=2.0 bash cls/mp_cell2.sh p64_auto_esc 64 $S64 $NCAL $NEVAL
METRIC=auto ESC=2.0 bash cls/mp_cell2.sh p96_auto_esc 96 $S96 $NCAL $NEVAL

echo "=== sweep_wave2b COMPLETE $(date +%T)"
