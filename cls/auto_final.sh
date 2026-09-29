#!/usr/bin/env bash
# Unattended driver for the formal n=10000 run. Started once per GPU:
#
#   HALF=a bash cls/auto_final.sh
#   HALF=b bash cls/auto_final.sh
#
# It waits for prep, makes the two decisions prep exists to inform, launches the
# phased run, and leaves a machine-readable verdict file behind. Everything is
# cache-keyed by tag, so re-running after an interruption resumes rather than
# repeats.
set -uo pipefail
cd /nfs/turbo/coe-nbleier/yjrcs/HPCA/vit_sc || exit 1
HALF=${HALF:-a}
LOG_TS () { date +%T; }

echo "########## AUTO-FINAL [half $HALF] $(hostname) $(LOG_TS)"

# ------------------------------------------------------------------ 1. prep
# Both halves need BOTH preps: half A needs mask10_t48, half B needs the t128
# sensitivity that the int8 mask is derived from.
#
# The eff-96 PCF cells are waited on TOO, and that is not redundant: prep A runs
# them AFTER its mask, so the t128 mask (prep B, ~15 min later) can land first.
# Waiting only on masks would let step 3 read a missing m48_pc.json, silently
# fall through to the PCF_96=0 default, and throw away the entire reason prep
# exists. Wait for the artifacts the DECISION needs, not the ones that happen to
# appear first.
for f in results/hybrid/mask10_t48.json results/hybrid/mask10_t128.json \
         results/fig1k/m48_base.json results/fig1k/m48_pc.json; do
  echo "[auto] waiting for $f $(LOG_TS)"
  waited=0
  while [ ! -f "$f" ]; do
    sleep 60; waited=$((waited+60))
    [ "$waited" -ge 7200 ] && { echo "[auto] TIMEOUT waiting for $f -- aborting"; exit 1; }
  done
  echo "[auto] $f present $(LOG_TS)"
done
# The JSON is written before the file is closed on some filesystems; make sure
# the decision reads a complete document rather than a truncated one.
for f in results/fig1k/m48_base.json results/fig1k/m48_pc.json; do
  for _ in 1 2 3 4 5; do
    python -c "import json,sys; json.load(open('$f'))" 2>/dev/null && break
    echo "[auto] $f not yet complete, retrying"; sleep 20
  done
done

# ------------------------------------------- 2. level-matched mask for eff-256
# int_bits is not an input to the sensitivity probe (it only lands in the emitted
# JSON), so the int8 mask is re-emitted from the int7 one rather than re-probed.
# Idempotent, and both halves may race here harmlessly -- same bytes either way.
if [ ! -f results/hybrid/mask10_t128_int8.json ]; then
  python cls/remask_bits.py results/hybrid/mask10_t128.json --int-bits 8
fi

# --------------------------------------------------- 3. the eff-96 PCF decision
# Protection is the one budget-dependent parameter (+4.52 at eff-64, +0.15 at
# eff-128, -0.35 at eff-192) and eff-96 has never been measured. prep_final.sh
# ran m48_base (PCF=0) and m48_pc (PCF=0.02) at n=1000 for exactly this.
#
# Rule, fixed in advance so it is not chosen to suit the answer: protection goes
# ON only if its increment clears +0.5 pt. It costs ~4x the calibration variance
# (sd 0.49 vs 0.13), so a coin-flip increment is a net loss even when positive --
# which is the reasoning that already put PC off at eff-128 (+0.15).
PCF_96=0
if [ -f results/fig1k/m48_base.json ] && [ -f results/fig1k/m48_pc.json ]; then
  INC=$(python - <<'PY'
import json
b = json.load(open('results/fig1k/m48_base.json'))['top1']
p = json.load(open('results/fig1k/m48_pc.json'))['top1']
print(f"{100*(p-b):.2f}")
PY
)
  echo "[auto] eff-96 protection increment = $INC pt (threshold +0.50)"
  [ "$(python -c "print(1 if $INC > 0.5 else 0)")" = "1" ] && PCF_96=0.02
else
  echo "[auto] WARNING: eff-96 PCF cells missing, defaulting PCF_96=0 (the eff-128 recipe)"
fi
echo "[auto] PCF_96=$PCF_96 $(LOG_TS)"
echo "$PCF_96" > results/final10k_PCF96.txt 2>/dev/null || true

# ------------------------------------------------------------------ 4. the run
mkdir -p results/final10k
export PCF_96
HALF=$HALF bash cls/run_final_10k.sh
echo "[auto] run_final_10k [half $HALF] returned $(LOG_TS)"

# ------------------------------------------------- 5. constraint check + rescue
# Only half A drives this, so the two halves do not both launch rescues.
if [ "$HALF" = "a" ]; then
  echo "########## AUTO-FINAL: constraint check $(LOG_TS)"
  # Wait for half B's MP phase so eff-128 is present before judging.
  waited=0
  while [ ! -f results/final10k/.phase_mp_b.done ] && [ "$waited" -lt 5400 ]; do
    sleep 60; waited=$((waited+60))
  done
  python cls/check_orderings.py results/final10k --json results/final10k/verdict.json
  for T in 32 48 64; do
    if python cls/check_orderings.py results/final10k --violated "$T" >/dev/null 2>&1; then
      echo "[auto] eff-$((T*2)) VIOLATES an ordering -> rescue search on the disjoint window $(LOG_TS)"
      bash cls/mp_rescue.sh "$T"
    fi
  done
  python cls/check_orderings.py results/final10k --json results/final10k/verdict.json
fi

echo "########## AUTO-FINAL [half $HALF] COMPLETE $(LOG_TS)"
