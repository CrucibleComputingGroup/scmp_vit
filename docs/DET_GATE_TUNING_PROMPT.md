# Agent task — tune the det QwT gate to match the working r²>0.3 baseline

**You are an autonomous coding agent. Execute this task end-to-end.**
Read files, edit code, run smoke tests, launch GPU jobs, commit + push
without asking. Stop only on a genuinely ambiguous decision not covered
below — propose your answer and proceed unless the user objects.

The repo is `vit_sc`, on `main` at commit `edc1fa0`. The det side just
shipped the cls-style cross-seed cosine gate
([`docs/DET_AGENT_PROMPT.md`](DET_AGENT_PROMPT.md), commits 66f6274 →
edc1fa0). On a 2026-04-25 n=10 smoke matrix, the new gate **collapses
on every config except p8_uniform**:

| config | raw bbox | comp bbox | Δ | admitted |
|---|---:|---:|---:|---:|
| p7_uniform_qwton    | 67.24 | 55.77 | **−11.47** | 26/40 |
| p7_mp_qwton         | 67.24 | 55.77 | **−11.47** | 26/40 |
| avg192_uniform_qwton| 70.52 | 62.64 | **−7.89**  | 35/40 |
| avg192_mp_qwton     | 68.70 | 61.76 | **−6.94**  | 35/40 |
| p8_uniform_qwton    | 69.23 | 67.72 | −1.51      | 36/40 |

*(smoke at `n_calib=16, n_eval=10, sc_prec=8, comp_sc_prec=8,
head_aligned, comp_mode=sc, cos_threshold=0.5,
last_block_cos_threshold=0.8, lookahead_veto`)*

The user reports that the **legacy `min_r2 > 0.3` gate was
known-working on det** (see the now-obsolete table in
[`det/RESULTS.md`](../det/RESULTS.md) — `min_r2=0.2` showed
+20.90 bbox AP at n=10 — and the user's standing recollection
that r² > 0.3 worked across configs). The cross-seed gate change
itself is principled (see
[`third_party/QwT-SC/QwT-vit-sc/qwt_sc/compensation.py`](../third_party/QwT-SC/QwT-vit-sc/qwt_sc/compensation.py)
module docstring), but its calibration on cls
(`n_calib=1024, sc_prec=8, residual dim 1024, ViT-L`) doesn't
auto-translate to det's regime
(`n_calib=16-256, sc_prec ∈ {7,8}, residual dim 1408, ViT-g`).

**Your job: tune the cross-seed gate (τ, last_block_τ, norm_floor,
n_calib, lookahead_veto behavior) so it doesn't collapse on det,
ideally matching the admission set that `min_r2 > 0.3` produces.**
Don't roll back the cross-seed gate as the default — keep it as the
single recipe (consistency with cls); just find the right knob values
for det.

## Stop conditions

- Required artifact missing (e.g. `qwt_d2` env doesn't exist —
  unlikely; the previous session built it).
- An assumption in this prompt is contradicted by the codebase.
- A genuinely ambiguous design choice not covered below — propose your
  pick and proceed unless the user objects.

---

## Start here — three reads

1. [`third_party/QwT-SC/QwT-vit-sc/qwt_sc/compensation.py`](../third_party/QwT-SC/QwT-vit-sc/qwt_sc/compensation.py)
   module docstring — current cross-seed gate, derivation, why it
   replaces r².
2. [`det/experiments/qwt_det_compensate.py`](../det/experiments/qwt_det_compensate.py)
   — det driver; CLI knobs (`--cos_threshold`, `--last_block_cos_threshold`,
   `--norm_floor`, `--lookahead_veto`, `--n_calib`).
3. [`det/results/smoke_sweep_2026-04-25/p7_uniform_qwton.json`](../det/results/smoke_sweep_2026-04-25/p7_uniform_qwton.json)
   `calib.per_block` — the per-block `cos_ab`, `r2_a`, `r2_b`,
   `rmse_before/after`, `enabled` for the collapsed p7 run. **This
   JSON is the diagnostic ground truth for everything that follows.**

The legacy r²-based code (commit `c3d99d5` in the QwT-SC submodule)
also exists if you want to A/B compare admissions side-by-side. Don't
restore it as the default; you can read it via
`git show c3d99d5:QwT-vit-sc/qwt_sc/compensation.py` (in the
submodule) for reference only.

---

## Server setup

Identical to [`DET_AGENT_PROMPT.md` §1](DET_AGENT_PROMPT.md). Conda env
`qwt_d2` should already exist on `gl1807`. If not, see that doc for
the rebuild recipe.

```bash
# Always launch via fire-and-forget ssh to gl1807:
ssh -n -f gl1807 'nohup setsid bash -lc "
  cd /home/allenjin/Projects/vit_sc/det
  source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
  conda activate qwt_d2
  module load gcc/13.2.0
  CUDA_VISIBLE_DEVICES=0 python <script> ARGS \
    > <log>.log 2>&1
" > /tmp/<outer>.log 2>&1 < /dev/null &'
```

Det at n=10 + n_calib=16: each qwt-on cell is **~50 min** (calib is
the bottleneck, not eval). Don't gate on n=10 numbers as ground truth
— gate on **gate decisions** (which blocks admitted) and bbox-AP
sign+magnitude.

---

## Hypothesis to test

The cls cross-seed pilot found τ insensitive in [0.3, 0.65] **at
N_calib=1024**. At small N_calib, the per-block `W_A, W_B` ridge fits
overfit; their cosine becomes unreliable in ways that cluster around
0.5–0.7 even on noise-fit blocks. So:

  Admission = `cos(W_A, W_B) > τ ∧ min(||W_A||, ||W_B||) > ε`

is too permissive at det's `n_calib=16`. Three plausible fixes
(non-exclusive):

1. **Tighten τ for det.** Try `cos_threshold ∈ {0.65, 0.75, 0.85,
   0.90}` — find the smallest τ that doesn't admit collapse-prone
   blocks. cls's pilot suggested τ=0.5 was safely inside the bimodal
   noise/signal gap; det may sit at the boundary, requiring τ ≥ 0.7.
2. **Raise n_calib up to ~128, not beyond.** Try `n_calib ∈ {32, 64,
   128}`. The user's directive: **n_calib=1024 overfits on det;
   128 is the upper bound.** This may seem counter to cls's
   `n_calib=1024` recipe, but cls and det are different regimes:
   det's calib images at 1280² are highly correlated within a
   contiguous COCO val slice, so additional samples past ~128 add
   little new signal but let the LS fit pick up dataset-shared noise
   structure that doesn't generalize to the eval distribution. So
   the right operating point on det is between the 16-image floor
   (visible noise-fit / collapse) and the 128-image ceiling (where
   diminishing returns turn into overfitting).
3. **Activate `norm_floor`.** Block 0 / block N-1 typically have
   small ||W̄||; cls's `norm_floor=0` lets them through. Det may need
   `norm_floor > 0` to filter out near-zero-bias blocks where cosine
   is numerically unstable.

A 4th idea — **always-on lookahead_veto** — is already the smoke
default. If it's not catching the collapse-prone blocks at p7, the
veto's one-step horizon may be insufficient. Don't extend the
horizon yet; first see if τ alone solves it.

---

## Concrete plan

### Step 1: reproduce the r²>0.3 baseline as a reference

Spin up a one-off worktree on the QwT-SC submodule's `c3d99d5` and
restore the **legacy** det driver from a pre-rewrite git commit (or
write a thin shim). Run **only one cell** — the same p7_uniform
smoke that collapsed at −11.47 — with `--min_r2 0.3`. Expected (from
the user's hint): **doesn't collapse**.

```bash
# From repo root
cd third_party/QwT-SC && git worktree add /tmp/qwt_r2 c3d99d5 && cd -
# det/experiments/qwt_det_compensate.py at commit 66f6274~1 (b5af8742
# — the last commit before Task A) calls the r²-gate API. Restore via:
git show 66f6274~1:det/experiments/qwt_det_compensate.py > /tmp/qwt_det_compensate_r2.py
```

The legacy driver imports from `qwt_sc.calibrate_qwt` — point its
sys.path at `/tmp/qwt_r2/QwT-vit-sc` so the r²-version `compensation.py`
resolves, then run with `--min_r2 0.3 --n_calib 16 --n_eval 10
--sc_prec 7 --sc_ops_per_block_json sensitivity/skip/skip_worst_30_int7.json
--comp_mode sc --comp_sc_prec 8 --head_aligned`. Save to a clearly
labelled out tag like `ref_r2_0.3_p7`.

If this DOES collapse, the user's recollection was about a different
config — flag it and stop. If it DOESN'T collapse, proceed.

**Decision: don't bother running the r² baseline at every config.**
One reference cell is enough — its purpose is to confirm the user's
hint and produce a per-block admission map you can compare against.

### Step 2: sweep τ at det's collapsing config

Run the cross-seed gate at the same p7_uniform_qwton config but with
varied τ:

```bash
for tau in 0.65 0.75 0.85 0.90; do
  ssh -n -f gl1807 'nohup setsid bash -lc "
    cd /home/allenjin/Projects/vit_sc/det
    source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
    conda activate qwt_d2 && module load gcc/13.2.0
    CUDA_VISIBLE_DEVICES=$gpu python experiments/qwt_det_compensate.py \
      --sc_prec 7 \
      --sc_ops_per_block_json sensitivity/skip/skip_worst_30_int7.json \
      --n_calib 16 --n_eval 10 \
      --calib_seed 1 --calib_seed_b 2 \
      --cos_threshold $tau --last_block_cos_threshold 0.9 \
      --lookahead_veto \
      --comp_mode sc --comp_sc_prec 8 --head_aligned \
      --out_tag tau_${tau}_p7 \
      --out_json results/gate_tuning/tau_${tau}_p7.json
  ..."
done
```

(4 cells × 50 min ÷ 4 GPUs = 50 min wall.)

Pass criterion: at least one τ produces **bbox AP comp ≥ raw**, ideally
matching the r²>0.3 reference admission set ± 2 blocks.

### Step 3: sweep n_calib at the τ that worked (cap 128)

If Step 2 found a working τ, hold it fixed and sweep n_calib ∈ {32,
64, 128}. Goal: confirm where on the [16, 128] interval the gate
stops collapsing — do NOT extrapolate beyond 128. The user has
verified that n_calib=1024 overfits on det; we treat 128 as the
hard upper bound and look for the sweet spot below it.

(3 cells × ~70 min for n_calib=128 ÷ 3 GPUs = 70 min wall — n_calib
scales calib forward time roughly linearly.)

### Step 4: compare admission decisions

For the best config from Step 2/3, load its `calib.per_block` JSON
plus the r²>0.3 reference's per-block report, and produce a side-by-
side per-block table:

```
block | r²>0.3 admits | xseed admits | cos_ab | r2_a | r2_b | comment
  0   |  N            | N             | +0.01  | -0.5 | -0.3 | rejected by both
  1   |  Y            | Y             | +0.82  | +0.61| +0.55| both admit
  2   |  Y            | N             | +0.55  | +0.42| +0.40| xseed too strict
  ...
```

This is the diagnostic that tells you whether the gates are
making the same physical decisions (good — algorithmic parity) or
disagreeing on a subset (more analysis needed).

### Step 5: ship the tuning

Once you have a τ (and possibly n_calib bump) that works on at least
3 of the 4 collapsing configs (p7_uniform, p7_mp, avg192_uniform,
avg192_mp), update det's defaults:

- `det/experiments/qwt_det_compensate.py` argparse defaults
  (`--cos_threshold`, `--last_block_cos_threshold`).
- `det/experiments/sweep_2026-04-25.sh` env-var defaults
  (`COS_THR`, `LAST_COS_THR`).
- `det/RESULTS.md` — add a "Gate tuning for det (vs cls)" section
  documenting the τ shift and the rationale (smaller calib budget;
  per-block cos shifts upward).
- `docs/SC_COMP_ALGORITHM.md` "Tuning τ" subsection — append a
  paragraph noting that the τ floor on det at small n_calib is
  ~0.65–0.85 (or whatever you found) and link to the
  per-block diagnostic.

**Don't change the cls defaults.** cls is on the principled τ=0.5 with
N_calib=1024; the right τ for cls is unchanged.

### Step 6: rerun the matrix sweep with the new defaults

Once the tuning is locked in, re-fire the full matrix at n=10 for
sanity:

```bash
RES_DIR=results/sweep_2026-04-25_v2 \
LOG_DIR=logs/sweep_2026-04-25_v2 \
N_EVAL=10 N_CALIB=<chosen> \
HEAD_ALIGNED=1 HEAD_ALIGNED_HEADS=16 \
bash det/experiments/sweep_2026-04-25.sh
```

Pass criterion: bbox AP comp ≥ raw on every config, or within ~1 AP
of raw for the worst config (n=10 noise floor).

---

## Things to **not** do

- **Don't roll back to the legacy r² gate as default.** cls retired it
  for principled reasons; the cross-seed gate is the single recipe
  going forward. The r²>0.3 baseline is a reference, not a target
  for production.
- **Don't run the production n=5000 sweep** until the n=10 matrix is
  green at the new defaults.
- **Don't refactor cls** (the shared `sc_integration/mp_search.py`
  was added in commit `d1c68f9` without touching cls's driver — keep
  cls stable).
- **Don't change `comp_mode=sc` to `fp`** — FP comp reintroduces an
  FP multiplier and breaks the SC story. Production must be sc.

---

## Reference numbers

cls N=50k production (from
[`docs/SC_COMP_ALGORITHM.md`](SC_COMP_ALGORITHM.md)):

| config         | raw SC | B_ha+gate Δ |
|----------------|---:|---:|
| p7_uniform     | 78.94 | **+3.93** |
| p7_mp          | 79.89 | **+3.37** |
| p8_uniform     | 85.57 | +0.13     |
| avg192_uniform | 84.67 | **+0.75** |
| avg192_mp      | 84.62 | **+0.85** |

The det target is "qualitatively similar" — positive Δ on every
config, biggest gains at lowest sc_prec. The smoke n=10 numbers
won't match these magnitudes (sample size). Just need Δ ≥ 0 on most
configs.

---

## Existing data you can use

- `det/results/smoke_sweep_2026-04-25/*_qwton.json` — per-block
  `calib.per_block` with `cos_ab`, `r2_a`, `r2_b`, `rmse_*` for the
  collapsed configs. **Inspect these first** — they have the
  diagnostic ground truth without spending another GPU-hour.
- `det/results/smoke_sweep_2026-04-25/sl_maps/*.json` — search-derived
  per-(op, block) sl_maps for p7_mp / avg192_mp; useful if you want
  to compare admission decisions on the mp variants.
- `cls/results/sweep_2026-04-25/*_qwton.json` — cls reference at
  N=50k for the same gate params (`τ=0.5`, `last_τ=0.8`,
  `lookahead_veto`). Per-block `cos_ab` for cls is the bimodal-clean
  reference your det per-block diagnostic should approach.

## Deliverables

When done, you should have:

1. A reference run with `min_r2 > 0.3` confirming the user's hint
   (Step 1).
2. A τ-sweep diagnostic (Step 2) and possibly an n_calib-sweep (Step
   3).
3. A per-block admission comparison table (Step 4).
4. Updated det defaults + docs (Step 5).
5. A final n=10 matrix sweep showing no collapses (Step 6).
6. One commit per logical step, pushed to `origin/main`. Co-author
   tag the same one used in recent commits
   (`Claude Opus 4.7 (1M context) <noreply@anthropic.com>`).

If at any point the diagnostic shows the cross-seed gate is
fundamentally inappropriate for det in the n_calib ≤ 128 regime
(e.g. no τ in [0.3, 0.95] matches r²>0.3's admissions on >2/3
blocks at any n_calib ≤ 128), say so and propose a det-specific
gate (e.g. AND of cross-seed and r², or a det-tuned r² floor with
the cross-seed cosine as a tiebreaker). Do NOT propose raising
n_calib past 128 — the user has verified that's already the
overfitting regime on det.

## Quick re-check before you write code

If you're rereading this prompt mid-task and unsure whether to stop
or continue, the answer is **continue.** This document already
authorizes every action it describes. The only legitimate stop
conditions are listed at the top.

Begin with the three "Start here" reads.
