#!/usr/bin/env bash
# THE FORMAL RUN. n=10000, ImageNet val, seed-0 shuffle, first 10000 images.
#
#   HALF=a bash cls/run_final_10k.sh     # GPU A
#   HALF=b bash cls/run_final_10k.sh     # GPU B
#   HALF=all bash cls/run_final_10k.sh   # single GPU, everything in one queue
#   bash cls/run_final_10k.sh --report   # tables only, no GPU
#
# PREREQUISITE: cls/prep_final.sh must have run (both halves). It builds
# mask10_t48 / mask10_t128 and the EXCL=10000 tables for eff-96, and it settles
# whether protection is on at eff-96. Set PCF_96 below from its output.
#
# ============================================================== WHAT IT RUNS
#   references     FP, INT4/6/7/8 asymmetric
#   uniform SC     eff-64, 96, 128, 256      (halved T = 32, 48, 64, 128)
#   hybrid uniform same four budgets, 10% INT7
#   MP SC          eff-64, 96, 128
#   hybrid MP SC   eff-64, 96, 128, 10% INT7
#
# 21 cells. At n=10000 a uniform cell is ~21 min, an MP cell ~33 min, INT ~10,
# FP ~4 -- roughly 9.5 GPU-hours, or ~5 h split across two GPUs.
#
# ============================================================ THE PARAMETERS
# Every value below was measured. Changing one invalidates the cached table, so
# a modified cell needs its OWN tag.
#
# DATAPATH, byte-identical in the uniform and MP arms:
#   --sc_prec 8 --halve 1     halved cycles; effective levels = 2 x T
#   --chunk_d 128             CORRECTNESS, not a cache knob: chunk_d=0 on a
#                             D_in=4096 MLP gives top-1 0.0000
#   --qk_granularity per_row  worth +0.59 pt to the UNIFORM arm by itself, so
#                             both arms must use it or MP books it twice
#
# ALLOCATOR:
#   PRICING=macs   REQUIRED once qk/av are in the table. The default 'rep' prices
#                  an attention row 16x a linear row at H=16 and starves qk to
#                  16.6 cycles against a 32 budget: top-1 0.672 vs uniform 0.783.
#   CLW=macs       W_g=uniform pins the cheap out_proj (8% of MACs) at the ladder
#                  ceiling and starves mlp_fc1 (32%) to 64.8 cycles at t96.
#   METRIC=amax    --metric-select auto costs -1.20; it reorders rows WITHIN a
#                  group and cannot fix the cross-operator split.
#   GLK=0          per-group ladders: 4 measurements, -1.66/-0.70/-0.70/-0.80,
#                  4/4 negative, none significant, while the sigma-surrogate
#                  IMPROVES 0.45%. See LADDER_REPORT.md -- flat-to-slightly-
#                  negative axis, so the simplest ladder wins.
#   ESC unset      escape gate: +1.08 while the row metric is broken, -2.00 once
#                  protection repairs it.
#   NCAL=128       calibration-draw sd ~1 pt at n=1000 evals; at n=10000 the eval
#                  component of that spread largely disappears (measured sd 0.07
#                  pt across 4 seeds at n=4000).
#   EXCL=10000     the calibration pool must exclude every eval image.
#
# PROTECTION -- the one budget-dependent parameter. Increment over 4 seeds:
#   eff-64 +4.52 | eff-128 +0.15 (and 4x the calibration variance) | eff-192 -0.35
#   eff-96 is set from prep_final.sh's measurement, see PCF_96.
#
# HYBRID: 10% of (op, block) groups -> INT, ranked by marginal knock-down dLoss
# AT THAT BUDGET (the t32 and t64 masks share only 2 of 14 groups, so masks do
# not transfer). Re-budgeted to T/(1-f) with f the MAC-weighted fraction -- NOT
# the group fraction; attention groups are cheap, so 10% of groups is 7.9% of
# MACs at t32, 9.8% at t64, 10.6% at t96 -- then ITERATED against the trace,
# because T/(1-f) assumes the masked groups sat at the average stream length and
# an MP allocation makes that false (t64 needed 71 -> 73; found three times
# independently).
#
# INT WIDTH IS MATCHED TO THE SC LEVEL COUNT, per budget:
#
#   budget    SC levels   INT     INT levels   ratio
#   eff-64        64      INT6        64        1.00
#   eff-96        96      INT7       128        1.33   <- the exception; there is
#   eff-128      128      INT7       128        1.00      no 96-level INT format,
#   eff-256      256      INT8       256        1.00      and INT6's 64 would be
#                                                         a deficit
# Holding INT7 everywhere instead would drift the masked groups' resolution from
# 2x the SC groups' at eff-64 to 0.5x at eff-256, so "10% hybrid" would not mean
# the same thing at the two ends of the curve and the hybrid curve's shape would
# mix the budget axis with a resolution axis.
#
# CONSEQUENCE, so nobody reads it as a regression: the eff-64 hybrid numbers here
# will be SMALLER than the ones in ISO_REPORT.md, which used INT7 at every
# budget. Level parity was measured there (section 2b) and costs the MP arm
# +2.10 -> +1.70 and the uniform arm +3.20 -> +2.30. These are the honest
# numbers; the earlier ones gave the masked groups 2x everyone else's resolution.
#
# The INT width does NOT change which groups are masked -- it is not an input to
# the sensitivity probe -- so the level-matched masks are re-emitted from the
# same sensitivity runs by cls/remask_bits.py rather than re-probed.
#
# eff-256 IS SPECIAL. 128/(1-f) is ~143, past the hard ceiling
# 2**(sc_prec-1)=128, and the kernel CLAMPS instead of erroring -- the cell would
# silently run at 128, spend ~(1-f) of uni_128's cycles, and be plotted as its
# equal. So the hybrid stays at 128 and the REFERENCE drops to 128*(1-f).
# prep_final.sh builds that reference; this script only reports it.
#
# HONESTY CAVEAT that travels with every hybrid number: iso-SC-cycles counts the
# INT work as FREE, the optimistic bound. The repo has no energy model.
set -uo pipefail

# Capture the caller's overrides BEFORE sourcing. sc_lib.sh applies its own
# defaults (`N=${N:-1000}`, `OUT=${OUT:-results/fig1k}`), so sourcing it first
# and writing `export N=${N:-10000}` afterwards is a no-op: N is already 1000 by
# then and the fallback never fires. That failure ran the formal sweep at n=1000
# into results/fig1k, where the tags also COLLIDE with 2026-07-29 cells -- so
# `sc_cell hyb_uni_32 35` found a stoc_len=32 int7 cell from a different study
# and reported it "cached". Two silent faults composing into wrong data.
_WANT_N=${N:-10000}
_WANT_OUT=${OUT:-results/final10k}
source cls/sc_lib.sh
export N=$_WANT_N
export OUT=$_WANT_OUT
export EXCL=${EXCL:-10000}
export NCAL=${NCAL:-128}
mkdir -p "$OUT"

# Print the resolved configuration, and refuse to write the formal run into a
# directory that already holds cells from another study -- tag collisions there
# are silent and produce numbers from the wrong experiment.
echo "[10k] resolved: N=$N OUT=$OUT EXCL=$EXCL NCAL=$NCAL"
if [ "$OUT" = "results/fig1k" ] || [ "$OUT" = "results/fig4k" ]; then
  echo "[10k] FATAL: OUT=$OUT is an existing study's directory. The formal run"
  echo "[10k] needs its own, or cached tags from another n / mask / stream length"
  echo "[10k] will be silently reused. Set OUT=results/final10k."
  exit 1
fi

# Protection, per budget. eff-96 comes from prep_final.sh; override without
# editing by exporting PCF_96.
PCF_32=${PCF_32:-0.02}
PCF_48=${PCF_48:-${PCF_96:-0.02}}
PCF_64=${PCF_64:-0}

HALF=${HALF:-all}
[ "${1:-}" = "--report" ] && HALF=report

# ---------------------------------------------------------------- helpers
# Level-matched mask per budget. See the INT WIDTH note in the header.
mask_for () {   # T -> mask path
  case "$1" in
    32)  echo results/hybrid/mask10_t32_int6.json ;;
    48)  echo results/hybrid/mask10_t48.json ;;        # int7, the exception
    64)  echo results/hybrid/mask10_t64.json ;;        # int7
    128) echo results/hybrid/mask10_t128_int8.json ;;
    *)   echo results/hybrid/mask10_t$1.json ;;
  esac
}

int_cell () {   # bits
  local b=$1
  [ -f "$OUT/int${b}_asymm.json" ] && { echo "[10k] int$b cached"; return; }
  echo "=== [int${b}_asymm] n=$N $(date +%T)"
  python -u cls/eval.py --mode int --int_bits "$b" --int_sym 0 --int_chunk 128 \
    --int_attn_scope per_row --sc_ops mlp_fc1,mlp_fc2,qk,av,proj \
    --data_root "$DATA" --max_images "$N" --batch_size 16 --workers 8 --seed 0 \
    --out_json "$OUT/int${b}_asymm.json" 2>&1 \
    | grep -viE "futurewarning|pynvml|warnings.warn" \
    | grep -E "^\[result\]|^\[eval\] per-image|Error|Traceback"
}

# uniform + its hybrid partner at one budget
uni_pair () {   # T
  local t=$1
  LADDER= GLK=0 PCF=0 sc_cell "uni_$t" "$t"
  local mask; mask=$(mask_for "$t")
  [ -f "$mask" ] || { echo "[10k] no $mask -- run prep_final.sh"; return; }
  local f tiso
  f=$(mask_frac "$mask" "$OUT/uni_${t}_trace.json")
  tiso=$(python -c "print(round($t/(1-$f)))")
  if [ "$tiso" -gt "$SC_MAX_STOC_LEN" ]; then
    # Past the grid ceiling: the hybrid simply runs AT the cap. Nothing is wrong
    # with the cell -- it is 10% INT and 90% SC at 256 levels, exactly as asked.
    # What it is not is iso with uni_128: the masked groups' cycles are not
    # spent, so it costs about (1-f) of it. collect_figure.py plots every cell at
    # its TRUE cost, so on the figure it lands at its own x (~230 levels) and
    # needs no partner. A partner is only needed for a PAIRED test, and eff-256
    # is saturated (uniform is already within 0.4 pt of FP), so that test cannot
    # resolve anything. Off by default; ISO_PARTNER=1 to spend the ~21 min.
    local tref; tref=$(python -c "print(round($SC_MAX_STOC_LEN*(1-$f)))")
    echo "[10k] eff-$((t*2)): iso target $tiso > cap $SC_MAX_STOC_LEN -> hybrid runs AT the cap;"
    echo "[10k]   it costs ~$(python -c "print(f'{100*(1-$f):.1f}')")% of uni_$t, so it plots at its own x, not uni_$t's."
    LADDER= GLK=0 PCF=0 sc_cell "hyb_uni_$t" "$SC_MAX_STOC_LEN" --hybrid_config_json "$mask"
    if [ "${ISO_PARTNER:-0}" = "1" ]; then
      echo "[10k]   ISO_PARTNER=1 -> also running uni_$tref as its exact paired comparator"
      LADDER= GLK=0 PCF=0 sc_cell "uni_$tref" "$tref"
    fi
  else
    echo "[10k] eff-$((t*2)): mask f=$f -> hybrid uniform at $tiso"
    LADDER= GLK=0 PCF=0 sc_cell "hyb_uni_$t" "$tiso" --hybrid_config_json "$mask"
  fi
}

# MP + its hybrid partner at one budget
mp_pair () {   # T, PCF
  local t=$1 pcf=$2
  LADDER= GLK=0 PCF=$pcf mp_cell "mp_$t" "$t"
  local mask; mask=$(mask_for "$t")
  [ -f "$mask" ] || { echo "[10k] no $mask -- run prep_final.sh"; return; }
  local f tiso
  f=$(mask_frac "$mask" "$OUT/uni_${t}_trace.json")
  tiso=$(python -c "print(round($t/(1-$f)))")
  echo "[10k] eff-$((t*2)) MP: mask f=$f -> hybrid MP at $tiso, iso-fitted against mp_$t"
  LADDER= GLK=0 PCF=$pcf mp_hybrid_iso "hyb_mp_$t" "$tiso" "$mask" "mp_$t" 2
}

# ---------------------------------------------------------------- barrier
# Phase order is INT -> uniform -> MP, GLOBALLY, not just within one GPU. The
# cheap reference cells therefore land first and de-risk the whole run: if the
# datapath, the loader or the eval subset is wrong, INT4 exposes it in 10
# minutes instead of after a 33-minute MP cell.
#
# The barrier is a sentinel file in $OUT, so it works across two srun tasks with
# no shared parent. It TIMES OUT rather than hanging forever: if the other half
# dies (allocation expiry, OOM), this one proceeds alone after the wait and says
# so, because a partial run whose cells are all iso-audited is worth more than a
# process blocked until the allocation ends.
phase_done () { touch "$OUT/.phase_${1}_${HALF}.done"; }
phase_wait () {   # phase, minutes
  local ph=$1 mins=${2:-90} other waited=0
  [ "$HALF" = "all" ] && return 0
  other=$([ "$HALF" = "a" ] && echo b || echo a)
  while [ ! -f "$OUT/.phase_${ph}_${other}.done" ]; do
    [ "$waited" -ge $((mins*60)) ] && {
      echo "[10k] BARRIER TIMEOUT after ${mins}m waiting for half $other on phase $ph -- proceeding alone"
      return 0; }
    sleep 30; waited=$((waited+30))
  done
  echo "[10k] barrier: half $other cleared phase $ph after ${waited}s"
}

# ---------------------------------------------------------------- queues
echo "########## FINAL 10k [half $HALF]  $(hostname) N=$N  $(date +%T)"

# -------- PHASE 1: references. Cheap, and they anchor everything else.
if [ "$HALF" != "report" ]; then
  echo "########## PHASE 1/3: references  $(date +%T)"
  if [ "$HALF" = "a" ] || [ "$HALF" = "all" ]; then
    [ -f "$OUT/fp.json" ] || { echo "=== [fp] n=$N $(date +%T)"; python -u cls/eval.py --mode fp \
        --data_root "$DATA" --max_images "$N" --batch_size 16 --workers 8 --seed 0 \
        --out_json "$OUT/fp.json" 2>&1 | grep -E "^\[result\]|Error|Traceback"; }
    int_cell 6; int_cell 4
  fi
  [ "$HALF" = "b" ] && { int_cell 8; int_cell 7; }
  [ "$HALF" = "all" ] && { int_cell 8; int_cell 7; }
  phase_done int; phase_wait int 60
fi

# -------- PHASE 2: uniform SC and hybrid uniform SC.
# Must precede MP for a hard reason as well as the requested one: mask_frac and
# every iso number are computed against the UNIFORM cell's trace, which is the
# only unpadded per-(op, block) MAC count in the run.
if [ "$HALF" != "report" ]; then
  echo "########## PHASE 2/3: uniform + hybrid uniform  $(date +%T)"
  if [ "$HALF" = "a" ] || [ "$HALF" = "all" ]; then uni_pair 32; uni_pair 48; fi
  if [ "$HALF" = "b" ] || [ "$HALF" = "all" ]; then uni_pair 64; uni_pair 128; fi
  phase_done uni; phase_wait uni 180
fi

# -------- PHASE 3: MP SC and hybrid MP SC.
if [ "$HALF" != "report" ]; then
  echo "########## PHASE 3/3: MP + hybrid MP  $(date +%T)"
  if [ "$HALF" = "a" ] || [ "$HALF" = "all" ]; then
    mp_pair 32 "$PCF_32"
    mp_pair 48 "$PCF_48"
  fi
  if [ "$HALF" = "b" ] || [ "$HALF" = "all" ]; then
    mp_pair 64 "$PCF_64"
  fi
  phase_done mp
fi

# ---------------------------------------------------------------- report
echo ""
echo "################################################################"
echo "########## EVAL-SET AUDIT -- same images, or every paired delta is noise"
echo "################################################################"
# Runs FIRST because it invalidates everything downstream. Paired tests compare
# per-image hit vectors positionally, so a cell scored on a different image list
# produces a meaningless delta that looks exactly like a real effect.
if ! python cls/check_eval_sets.py "$OUT" --quiet; then
  echo "########## ABORTING THE REPORT: cells did not score the same images."
  echo "########## Fix the odd cells (or move the search window to its own dir)"
  echo "########## before reading any number below."
  exit 1
fi

echo ""
echo "################################################################"
echo "########## ISO AUDIT -- equal true SC cycles or the deltas mean nothing"
echo "################################################################"
CELLS=$(cd "$OUT" && ls *_trace.json 2>/dev/null | sed 's/_trace\.json//' | paste -sd, -)
[ -n "$CELLS" ] && python cls/iso_check.py "$OUT" --mac-ref uni_32 --cells "$CELLS" 2>&1 | tail -30

echo ""
echo "################################################################"
echo "########## THE THREE ORDERINGS"
echo "################################################################"
for t in 32 48 64; do
  eff=$((t*2))
  hmp=hyb_mp_$t; [ -f "$OUT/${hmp}2.json" ] && hmp=${hmp}2
  [ -f "$OUT/mp_$t.json" ] || continue
  echo ""
  echo "===== eff-$eff ============================================="
  echo "--- (1) MP > uniform"
  python cls/paired_stats.py "$OUT" --base "uni_$t" --against "mp_$t" 2>&1 | grep -E "^\S+\s+0\."
  echo "--- (2) hybrid MP > hybrid uniform"
  python cls/paired_stats.py "$OUT" --base "hyb_uni_$t" --against "$hmp" 2>&1 | grep -E "^\S+\s+0\."
  echo "--- (3) hybrid MP > MP"
  python cls/paired_stats.py "$OUT" --base "mp_$t" --against "$hmp" 2>&1 | grep -E "^\S+\s+0\."
done

echo ""
echo "########## SC vs INT at matched levels, and headroom"
python - "$OUT" <<'PY'
import json
from pathlib import Path
import sys
d = Path(sys.argv[1])
def top1(t):
    p = d / f"{t}.json"
    return json.loads(p.read_text())["top1"] if p.exists() else None
fp = top1("fp")
print(f"  FP {fp}" if fp else "  FP not run")
for b in (4, 6, 7, 8):
    v = top1(f"int{b}_asymm")
    if v: print(f"  INT{b} (2^{b}={2**b} levels)  {v:.4f}")
for t in (32, 48, 64, 128):
    u, m = top1(f"uni_{t}"), top1(f"mp_{t}")
    if u is None: continue
    line = f"  eff-{2*t:<4d} uniform {u:.4f}"
    if m: line += f"  MP {m:.4f}"
    if fp: line += f"  headroom {100*(fp-u):+.2f} pt"
    print(line)
PY

echo ""
echo "=== run_final_10k [$HALF] COMPLETE $(date +%T)"
