# QwT compensation — investigate the "loss" cases

**For:** a fresh Claude agent / research engineer picking up where we left off.
**Status:** opt3 recipe (`--r2_threshold 0.5 --last_block_r2_threshold 0.9
--lookahead_veto`) is committed and stable across every K=30 config — no
collapses. But QwT only provides a meaningful gain on **p7_uniform** (+2.16 pt
over raw SC at N=50k prod). Three configs where QwT either does nothing or
regresses at opt3 need a targeted investigation:

| config | raw SC (N=50k) | opt3 r²=0.5 la+last0.9 (N=1k) | Δ | notes |
|---|---:|---:|---:|---|
| p8_uniform     | 85.57 % | 84.90 % | **−0.67** | high-baseline; comp adds more noise than it removes |
| avg192_mp      | 84.62 % | 84.00 % | **−0.62** | same family; search-driven sl_map may interact |
| p7_mp          | 79.89 % | 79.90 % | +0.01 (neutral) | only 3 blocks enabled; tiny effect |

Your job: figure out *why* and, if possible, find a gate / recipe tweak that
moves these three closer to zero regression (or ideally a gain) **without
regressing the configs where opt3 already helps** (p7_uniform +2.16 pt,
avg192_uniform +0.43 pt).

---

## Where things stand

### Architecture
- Model: DINOv2 ViT-L/14 + linear head (`dinov2_vitl14_lc`). 24 blocks,
  d=1024, pre-norm.
- SC patching: per-(op, block) schedule at K=30 (30 worst-sensitivity (op,
  block) cells skipped to FP). See [cls/sensitivity/skip_worst30/](../cls/sensitivity/skip_worst30/).
- QwT-SC compensator: closed-form ridge LS per block, wrapped as
  `CompensationBlock` by [third_party/QwT-SC/QwT-vit-sc/qwt_sc/compensation.py](../third_party/QwT-SC/QwT-vit-sc/qwt_sc/compensation.py).
- Both uniform and MP paths now route the QwT comp itself through the SC
  kernel (`--comp_mode sc --comp_sc_prec 8`) for apples-to-apples hardware
  accounting.

### Gate knobs (already plumbed, CLI-accessible)
- `--r2_threshold T` → enable block if `r² > T`.
- `--last_block_r2_threshold T_last` → override for block 23 only.
- `--lookahead_veto` → 1-step binary lookahead: propagate `sc + comp` vs
  `sc` through `block_fp[i+1]`, veto apply when skip's downstream error
  is smaller.
- `--max_block_for_qwt N` → hard cap.
- Per-block calib report persisted in result JSON at
  `d["calib"]["per_block"][i]` with fields `r2`, `enabled`, `rmse_before`,
  `rmse_after`, `lookahead_decision ∈ {keep, veto, None}`,
  `lookahead_err_apply`, `lookahead_err_skip`, `r2_threshold_used`.

### What the "opt3" recipe does
At r²=0.5, block 23's fit r² empirically lands in [0.28, 0.35] in every
K=30 config, so the primary `r²_threshold=0.5` gate already skips it.
Mid-network blocks with r² > 0.5 pass; the 1-step lookahead sometimes
vetoes (e.g., p7_uniform has 3 keeps / 1 veto out of 4 that passed r²=0.5).

Per-block stats at opt3 r²=0.5:

| config | keep / veto / total-enabled |
|---|:-:|
| p7_uniform    | 3 / **1** / 3 |
| p7_mp         | 3 / 0 / 3     |
| avg192_mp     | 5 / 0 / 5     |
| p8_uniform    | 5 / 0 / 5     |
| avg192_uniform| 5 / 0 / 5     |

Only 3–5 blocks are enabled per config; the lookahead rarely fires.
That's the baseline you're working from.

---

## The three cases, each with its own hypothesis

### Case 1 — `p8_uniform` regression (−0.67 pt)
**Observation:** Raw SC is 85.57 % — *already within 0.3 pt of FP baseline* for
this tier. QwT with opt3 enables 5 blocks' comps and drops to 84.90 %.

**Hypotheses (ranked by likelihood):**
1. **Signal-to-noise inversion at high baseline.** When SC error is already
   small, the ridge LS fit captures mostly Sobol RNG variance, not
   deterministic quantization drift. Applying the comp then *adds*
   variance (the SC-kernel comp runs its own Sobol draw) rather than
   removing bias. Classic variance-vs-bias crossover.
2. **Over-regularization at ridge=1e-4.** The ridge value was chosen for
   the MP-noisy regime; at high baseline it may be too loose, letting W
   absorb the noise floor instead of shrinking.
3. **Comp SC precision mismatch.** `comp_sc_prec=8` (SL=256) but uniform
   p8 already runs at SL=256. The comp's own SC variance is the same
   order as the block's, so there's no "clean correction" headroom.
4. **Missing blocks matter less than enabled ones.** The 5 enabled
   blocks might not be the ones that most need fixing at p8. A
   per-block ablation (disable one at a time, measure) would reveal
   which are helping vs hurting.

**Experiments to run (at N=1000 first, N=50k once promising):**
```bash
# A. Ridge sweep — does tightening ridge help at high baseline?
for ridge in 1e-6 1e-4 1e-2 1e-1 1.0; do
  python experiments/qwt_sc_overnight.py \
    --sc_ops_per_block_json sensitivity/skip_worst30/skip_worst30_p8.json \
    --sc_prec 8 --batch_size 16 --workers 8 \
    --n_calib 1024 --n_eval 1000 \
    --comp_mode sc --comp_sc_prec 8 --ridge $ridge \
    --r2_threshold 0.5 --last_block_r2_threshold 0.9 --lookahead_veto \
    --skip_baseline \
    --out_json results/investigate/p8_uniform_ridge${ridge}.json
done

# B. Comp SC precision sweep — higher comp_sc_prec → cleaner correction?
for csp in 8 9 10 12; do
  # note: comp_sc_prec>8 may not be supported; check first.
  python experiments/qwt_sc_overnight.py \
    --sc_ops_per_block_json sensitivity/skip_worst30/skip_worst30_p8.json \
    --sc_prec 8 --batch_size 16 --workers 8 \
    --n_calib 1024 --n_eval 1000 \
    --comp_mode sc --comp_sc_prec $csp --ridge 1e-4 \
    --r2_threshold 0.5 --last_block_r2_threshold 0.9 --lookahead_veto \
    --skip_baseline \
    --out_json results/investigate/p8_uniform_csp${csp}.json
done

# C. FP comp as sanity — does the regression exist even with noise-free comp?
python experiments/qwt_sc_overnight.py \
  --sc_ops_per_block_json sensitivity/skip_worst30/skip_worst30_p8.json \
  --sc_prec 8 --batch_size 16 --workers 8 \
  --n_calib 1024 --n_eval 1000 \
  --comp_mode fp --ridge 1e-4 \
  --r2_threshold 0.5 --last_block_r2_threshold 0.9 --lookahead_veto \
  --skip_baseline \
  --out_json results/investigate/p8_uniform_fpcomp.json
# If FP comp also regresses → the fit itself is the problem (H1, H2).
# If FP comp gains → SC comp's own variance is the culprit (H3).

# D. Per-block leave-one-out (manual). For each of the 5 enabled blocks,
# force `--max_block_for_qwt=<one_less_than_that_block>` or patch the
# schedule to skip just that block. Measure top-1.
```

**What to report:** (a) ridge curve, (b) comp_sc_prec curve, (c) FP-comp
number, (d) per-block contribution. Conclusion: can we beat raw SC at p8
by adjusting any of these knobs, or is high-baseline QwT fundamentally
variance-dominated?

### Case 2 — `avg192_mp` regression (−0.62 pt)
**Observation:** Uses the search-driven sl_map at `main_sl=192`. Raw SC
84.62 %, opt3 QwT 84.00 %.

**Hypotheses:**
1. **Same as p8_uniform H1** (variance-dominated at ~85% baseline) —
   most likely.
2. **sl_map heterogeneity interaction.** Different blocks have different
   stoc_len (some at 256, some at 128), so the comp's per-block fit
   operates on different noise floors. The uniform ridge might be
   mis-calibrated for this mix.
3. **mlp_fc2=192 floor effect.** The search pins mlp_fc2 at 192 in every
   block; maybe fc2's comp at 192 is specifically noisy.

**Experiments:**
```bash
# A. Ridge sweep, same as p8 case (quick).
# B. FP comp variant.
# C. Compare p8_uniform vs avg192_mp per-block r². Is mlp_fc2's comp
#    systematically lower-quality at 192 vs block-op average?
python3 - <<'PY'
import json
for name in ("p8_uniform_qwton","avg192_mp_qwton"):
    p = f"/home/allenjin/Projects/vit_sc/cls/results/sweep_int678_k30_opt3_r2p5_lastblock_r2p9_n1000/{name}_r2p05.json"
    d = json.load(open(p))
    rep = d["calib"]["per_block"]
    print(f"\n=== {name} ===")
    for b in rep:
        print(f"  block {b['block']:2d}  r2={b['r2']:+.4f}  enabled={b['enabled']}  la={b.get('lookahead_decision')}")
PY
```

### Case 3 — `p7_mp` neutral (+0.01 pt)
**Observation:** Only 3 blocks pass r²=0.5. QwT has tiny effect.

**Hypotheses:**
1. The 3 enabled blocks genuinely don't help much at this budget. Lowering
   r²_threshold would admit more blocks — but we already know r²=0.3
   there sometimes collapses (opt2 run at r²=0.3 had p7_mp at 82.00 %,
   which was OK in that specific run; would need to retry with
   lookahead + last_block=0.9 to see if stable).
2. The search-driven sl_map at main_sl=128 might have a comp-hostile
   structure (short bitstreams on mlp_fc1/proj → noisy ridge fit).

**Experiments:**
```bash
# A. Try r²=0.3 with last_block=0.9 + lookahead for p7_mp specifically.
python experiments/eval_custom_sl_map_fpcomp.py \
  --sc_config skip_worst30 \
  --sl_map_json results/sweep_int678_k30/sl_maps/p7_mp_main128_sl_map.json \
  --sc_prec 8 --n_calib 1024 --n_eval 1000 \
  --batch_size 16 --workers 8 \
  --comp_mode sc --comp_sc_prec 8 --ridge 1e-4 \
  --r2_threshold 0.3 --last_block_r2_threshold 0.9 --lookahead_veto \
  --out_json results/investigate/p7_mp_r2p03_last0p9.json

# B. Try an sl_map that's less MP-aggressive (more blocks at higher SL).
# This tests hypothesis 2: if the comp works better on a gentler map,
# MP pushes us into a bad-comp regime.
```

---

## How to execute

**Environment** (matches the existing sweeps):
```bash
ssh gl1807 'cd /home/allenjin/Projects/vit_sc/cls && \
    source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh && \
    conda activate vit_sc && \
    export XFORMERS_DISABLED=1'
```
Shared scratch data at `/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data`.
4× RTX PRO 6000 Blackwell. **GPUs are in `EXCLUSIVE_PROCESS` compute mode** —
only one CUDA context per GPU. If a worker dies in <60 s with
`cudaErrorDevicesUnavailable`, another process is holding that GPU;
`nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r
kill -9` clears zombies.

**Suggested scaffold:** write a pool-scheduled driver (4 GPUs with a
3-MP cap, singleton launch via `nohup + ssh -n -f`, zombie-reap via
`pgrep` / `kill -0`) that writes to `results/investigate/<experiment>/*.json`
and `logs/investigate/<experiment>/`.

**Constraints for reproducibility:**
- Hold `--n_calib 1024 --n_eval 1000` for quick iteration; confirm
  promising results at `--n_eval 50000` before drawing conclusions.
- Keep the raw-SC baseline column in any comparison (the thing we're
  trying to beat is `raw SC`, not `raw SC + FP baseline`).
- Don't touch the existing `results/sweep_int678_k30*/` directories —
  those are reference data.

**Reference files:**
- [third_party/QwT-SC/QwT-vit-sc/qwt_sc/compensation.py](../third_party/QwT-SC/QwT-vit-sc/qwt_sc/compensation.py) — `calibrate_qwt` + `CompensationBlock` + the cross-seed cosine gate (see module docstring for the algorithm that superseded this investigation's r²-based gates).
- [cls/experiments/qwt_sc_overnight.py](../cls/experiments/qwt_sc_overnight.py) — uniform-path QwT driver.
- [cls/experiments/eval_custom_sl_map_fpcomp.py](../cls/experiments/eval_custom_sl_map_fpcomp.py) — sl_map-driven QwT driver (used by MP configs).
- [cls/experiments/mp_budget_swap_search.py](../cls/experiments/mp_budget_swap_search.py) — how the MP sl_maps are generated (for Case 2 hypothesis 3 if you need to regenerate a gentler map).

**Result JSONs for the cases being investigated:**
- `results/sweep_int678_k30_opt3_r2p5_lastblock_r2p9_n1000/p8_uniform_qwton_r2p05.json`
- `results/sweep_int678_k30_opt3_r2p5_lastblock_r2p9_n1000/avg192_mp_qwton_r2p05.json`
- `results/sweep_int678_k30_opt3_r2p5_lastblock_r2p9_n1000/p7_mp_qwton_r2p05.json`

Each has `calib.per_block[]` with the per-block fit r², enabled, and
lookahead decision — start there before rerunning anything.

**Deliverable:** a short markdown report in `docs/qwt_gate_findings.md`
with: (a) root cause for each of the 3 cases, (b) any recipe tweaks that
recover a net gain without breaking the p7_uniform / avg192_uniform wins
of opt3, (c) recommended next step (ship a refinement, park the
investigation, or escalate to full-chain lookahead).

---

## Quick-start command (drop-in launch)

```bash
ssh -f gl1807 'cd /home/allenjin/Projects/vit_sc/cls && \
    mkdir -p results/investigate logs/investigate && \
    source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh && \
    conda activate vit_sc && \
    export XFORMERS_DISABLED=1 && \
    nohup bash -c "
      # Phase 1 — baseline diagnostics (no new code needed):
      # Pull per-block r² + enabled + lookahead decisions for the 3 loss cases.
      python3 -c \"
import json
for name, path in [
  ('p8_uniform',     'results/sweep_int678_k30_opt3_r2p5_lastblock_r2p9_n1000/p8_uniform_qwton_r2p05.json'),
  ('avg192_mp',      'results/sweep_int678_k30_opt3_r2p5_lastblock_r2p9_n1000/avg192_mp_qwton_r2p05.json'),
  ('p7_mp',          'results/sweep_int678_k30_opt3_r2p5_lastblock_r2p9_n1000/p7_mp_qwton_r2p05.json'),
]:
    d = json.load(open(path))
    rep = d['calib']['per_block']
    print(f'=== {name} ===')
    for b in rep:
        la = b.get('lookahead_decision','-')
        print(f\\\"  block {b['block']:2d}  r2={b['r2']:+.4f}  enabled={b['enabled']}  la={la}\\\")
    print()
\"
    " </dev/null >logs/investigate/phase1_per_block_inspection.log 2>&1
'
```

Start with that — the per-block r² and enable pattern is the smoking gun
for which hypothesis is live. Everything else branches from there.
