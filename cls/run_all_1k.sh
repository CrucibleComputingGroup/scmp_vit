#!/usr/bin/env bash
# Single-GPU (gl1807) master queue, n=1000. Replaces the 50k plan.
#
# EVAL SUBSET. 1000 images = the first 1000 of shuffle(seed 0). Every cached
# calibration table was built with --eval_exclude_n 2048 off that same shuffle,
# and 1000 < 2048, so the eval set is a strict subset of what calibration
# already excluded — existing tables stay disjoint and only the EVAL has to be
# re-run. New cells keep --eval_exclude_n 2048 for the same reason.
#
# STATISTICS. At n=1000 the marginal Wilson half-width is +-2.6 pt at p~0.78,
# wider than the 1-3 pt effects here. Every cell therefore dumps per-image
# top-1 hits (<tag>_hits.npy) in loader order, so differences between cells can
# be tested PAIRED (McNemar on the discordant pairs) instead of by overlapping
# marginal intervals. Do not quote a marginal CI as evidence against a paired
# difference at this n.
set -uo pipefail
cd /nfs/turbo/coe-nbleier/yjrcs/HPCA/vit_sc || exit 1
export TORCH_HOME=$PWD/.torch_cache
D=${IMAGENET_ROOT:-/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data}

N=${N:-1000}
# 128 calibration images (was 256). Calibration dominates per-cell cost now, so
# this roughly halves it. Cost: MP_REPORT measured the calibration-sample sd at
# ~1 pt at n_calib=128 vs 0.33 pt at 256, i.e. comparable to the smaller
# effects here. Phase D therefore replicates the winning cell across
# calibration seeds rather than trusting a single draw.
NCAL=${NCAL:-128}
OUT=results/fig1k
mkdir -p $OUT

S32=64,48,32,24,16
S64=128,96,64,48,32
S96=128,112,96,80,64,48
ALLOPS=qkv_proj,out_proj,mlp_fc1,mlp_fc2,qk,av

ev () {   # tag, then eval.py args
  local tag=$1; shift
  [ -f "$OUT/$tag.json" ] && { echo "[1k] $tag cached"; return; }
  echo "=== [$tag] $(date +%T)"
  SC_MP_TRACE=$OUT/${tag}_trace.json python -u cls/eval.py \
    --data_root "$D" --max_images "$N" --batch_size 16 --workers 8 --seed 0 \
    --out_json "$OUT/$tag.json" "$@" 2>&1 \
    | grep -viE "futurewarning|pynvml|warnings.warn" \
    | grep -E "^\[hybrid\]|^\[result\]|^\[eval\] per-image|Error|Traceback|SystemExit"
}

sc () {   # tag, stoc_len, then extra args
  local tag=$1 sl=$2; shift 2
  ev "$tag" --mode sc --sc_prec 8 --halve 1 --chunk_d 128 --stoc_len "$sl" \
     --sc_ops mlp_fc1,mlp_fc2,qk,av,proj --qk_granularity per_row "$@"
}

# ---------------------------------------------------------------- Phase 0
# Hybrid masks, one per budget: SC fragility is budget-dependent, so a mask
# measured at one stream length does not transfer to another. Cut at qk
# per_row because that is what every deployed cell uses. Masks at different
# --frac off ONE sensitivity run are strictly NESTED, so 5/10/20% is a genuine
# dose axis and 10% is a point on it rather than a lone guess.
# COST: the probe is |ops|*blocks + 1 = 145 forwards. At 256 images and ~7
# img/s that measured 77 min PER BUDGET — three budgets would have cost 4 h,
# more than the rest of the plan combined. Cut to 128 probe images and to the
# two budgets the hybrid arms actually need (t96 is the mildest, where SC is
# already near FP and a mask has least to do).
#
# NOISE: at 256 images the mean dL was NEGATIVE for every operator, i.e. the
# tail of the ranking is sampling noise, not signal. Only the top ~14 groups
# sat 2-6x above that floor. The mask uses exactly those, so it survives the
# cut — but do NOT read the full ranking as meaningful, and do not compare
# small dL differences between low-ranked groups.
echo "########## Phase 0: hybrid sensitivity + masks"
for T in 32 64; do
  [ -f "results/hybrid/mask10_t$T.json" ] && { echo "[1k] mask t$T cached"; continue; }
  python -u cls/build_hybrid_mask.py --stoc_len $T --tag t$T \
    --frac 0.05,0.10,0.20 --int_bits 7 --int_sym 0 --qk_granularity per_row \
    --n_images 128 --batch_size 16 --data_root "$D" 2>&1 \
    | grep -viE "futurewarning|pynvml|warnings.warn" \
    | grep -E "^\[sens\]|Error|Traceback"
done

# ---------------------------------------------------------------- Phase A
# References. INT uses per_row attention scope so its scale scope matches SC's
# — the whole point of the levels axis is that the two families are otherwise
# identically configured.
echo "########## Phase A: references"
ev fp --mode fp
for B in 8 7 6 4; do
  ev "int${B}_asymm" --mode int --int_bits $B --int_sym 0 --int_chunk 128 \
     --int_attn_scope per_row --sc_ops mlp_fc1,mlp_fc2,qk,av,proj
done
for SL in 128 96 64 32; do sc "uni_$SL" $SL; done

# ---------------------------------------------------------------- Phase B
# MP, all at qk per_row. Recipe built up one knob at a time at the aggressive
# budget so each delta is attributable, then the winner carried to the other
# two budgets. Group ladders are excluded: they lost 1.66 pt in wave 1.
echo "########## Phase B: MP (qk per_row)"
# PRICING=macs is REQUIRED here, not a tuning choice. With qk/av in the table
# the default rep_g pricing charges an attention row 16x a linear row (B*H*N
# vs B*N rows, H=16), and the allocator starves attention to the ladder floor:
# measured qk at 16.6 cycles against a 32 budget, top-1 0.672 vs uniform 0.783.
# Attention is only 4% of MACs, so the starvation buys almost nothing and costs
# 11 pt. vit_sc's CLAUDE.md called this knob a no-op on ViT — true for a
# linear-only table, false the moment attention joins it.
export QKG=per_row MPOPS=$ALLOPS PRICING=macs
mp () {   # tag, budget, ladder  (calibrates if needed, then evals at n=1000)
  local tag=$1 t=$2 ladder=$3
  if [ ! -f "results/mp_calib/${tag}_wrapper.json" ]; then
    # Keep the calibration's own diagnostics in the log. Sending them to
    # /dev/null once turned a GPU-contention crash into a bare "NO TABLE" for
    # nine consecutive cells, which is indistinguishable from a config error.
    bash cls/mp_cell2.sh "$tag" "$t" "$ladder" "$NCAL" 16 2>&1 \
      | grep -E "^\[calib\]|^\[cell\]|rho=|Error|Traceback|SystemExit|CUDA" || true
  fi
  if [ ! -f "results/mp_calib/${tag}_wrapper.json" ]; then
    echo "[1k] $tag: NO TABLE — calibration failed, see above"; return
  fi
  sc "$tag" "$t" --adaptive_mp 1 \
     --adaptive_mp_table "results/mp_calib/${tag}_wrapper.json" \
     --adaptive_mp_ops "$ALLOPS"
}

CLW=macs                                     mp r32_base    32 $S32
CLW=macs PCF=0.02                            mp r32_pc      32 $S32
CLW=macs PCF=0.02 ESC=2.0                    mp r32_pc_esc  32 $S32
CLW=macs PCF=0.02 ESC=2.0 METRIC=auto        mp r32_full    32 $S32
CLW=macs PCF=0.05                            mp r32_pc5     32 $S32

                                             mp r64_base    64 $S64
             PCF=0.02 ESC=2.0                mp r64_pc_esc  64 $S64
                                             mp r96_base    96 $S96
             PCF=0.02 ESC=2.0                mp r96_pc_esc  96 $S96

# ---------------------------------------------------------------- Phase C
# 10% hybrid INT mask on BOTH arms. Masks come from cls/build_hybrid_mask.py
# (ranked by marginal knock-down dLoss at that budget). uniform+mask vs MP+mask
# holds the mask fixed, which is the only comparison the mask cannot flatter;
# against an unmasked cell it is an energy-axis claim.
echo "########## Phase C: 10% hybrid SC/INT"
for T in 32 64; do
  M=results/hybrid/mask10_t$T.json
  [ -f "$M" ] || { echo "[1k] no mask for t$T — skipped"; continue; }
  sc "hyb_uni_$T" $T --hybrid_config_json "$M"
  W=results/mp_calib/r${T}_pc_esc_wrapper.json
  [ -f "$W" ] || W=results/mp_calib/r${T}_base_wrapper.json
  [ -f "$W" ] && sc "hyb_mp_$T" $T --hybrid_config_json "$M" \
      --adaptive_mp 1 --adaptive_mp_table "$W" --adaptive_mp_ops "$ALLOPS"
done

# ---------------------------------------------------------------- Phase D
# Calibration-seed replication of the winner at the aggressive budget. At
# n_calib=128 a single draw carries ~1 pt of calibration noise, which is the
# size of the effects being claimed, so a one-draw win is not a result.
echo "########## Phase D: calibration-seed replication"
for SEED in 7 23 41; do
  CLW=macs PCF=0.02 ESC=2.0 CALIB_SEED=$SEED mp "r32_pc_esc_s$SEED" 32 $S32
done

echo "=== run_all_1k COMPLETE $(date +%T)"
