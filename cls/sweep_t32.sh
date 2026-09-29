#!/usr/bin/env bash
# Knob ablation at the effective-64 budget (halved 32) — the budget where MP
# was NEGATIVE vs uniform SC before 2026-07-29 (MP_REPORT.md: -0.63 single
# draw, -1.22 over 4 seeds at W_g=uniform; +0.11 with W_g=macs).
#
# Every cell holds W_g=macs (the known fix at this budget) and varies ONE knob,
# so a win is attributable. Uniform comparator: results/int_vs_sc sc_halved32
# = 0.7505 for the per_head cells, and the per_row uniform cell run here for
# the attention cells (qk granularity changes the uniform arm too).
set -uo pipefail
cd /nfs/turbo/coe-nbleier/yjrcs/HPCA/vit_sc || exit 1

NCAL=${NCAL:-256}
NEVAL=${NEVAL:-2048}
SPARSE=64,48,32,24,16
DENSE=64,56,48,40,32,24,20,16
ALLOPS=qkv_proj,out_proj,mlp_fc1,mlp_fc2,qk,av

export CLW=macs

# 1. incumbent: W_g=macs on the sparse ladder, no new knobs
CLW=macs bash cls/mp_cell2.sh s32_base 32 $SPARSE $NCAL $NEVAL

# 2. per-group ladders: dense sigma, 3 deployed rungs per (op x block)
GLK=3 bash cls/mp_cell2.sh s32_glk 32 $DENSE $NCAL $NEVAL

# 3. absolute escape gate at k=2
ESC=2.0 bash cls/mp_cell2.sh s32_esc 32 $SPARSE $NCAL $NEVAL

# 4. protected input channels, 2% by activation amax
PCF=0.02 bash cls/mp_cell2.sh s32_pc2 32 $SPARSE $NCAL $NEVAL

# 5. attention gets MP tables, qk dispatched per query row
QKG=per_row MPOPS=$ALLOPS bash cls/mp_cell2.sh s32_attn 32 $SPARSE $NCAL $NEVAL

# 6. everything on
GLK=3 ESC=2.0 PCF=0.02 QKG=per_row MPOPS=$ALLOPS \
  bash cls/mp_cell2.sh s32_all 32 $DENSE $NCAL $NEVAL

echo "=== sweep_t32 COMPLETE $(date +%T)"
