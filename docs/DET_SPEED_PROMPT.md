# Agent task — speed up det's QwT calibration + smoke inference

**You are an autonomous coding agent. Execute this task end-to-end.**
Read files, edit code, profile, run smoke tests, launch GPU jobs,
commit + push without asking. Stop only on a genuinely ambiguous
decision not covered below — propose your answer and proceed unless
the user objects.

The repo is `vit_sc`, on `main` at commit `270d436`. The det side just
shipped the cls-style cross-seed cosine gate + head-aligned comp + MP
search ([`docs/DET_AGENT_PROMPT.md`](DET_AGENT_PROMPT.md), commits
66f6274 → 270d436). All the algorithm pieces are in place — but the
**runtime is too slow** to iterate on smokes:

| stage | cls (ViT-L, 224²) | det (ViT-g, 1280²) | ratio |
|---|---:|---:|---:|
| forward / image (raw SC, sw30) | ~0.15 s | **~25-30 s** | ~180× |
| QwT calib (n_calib batch) | ~5 min @ N=1024 | ~30 min @ **N=16** | per-image ~120× |
| n=10 smoke wall (qwt-on cell) | n/a | **~50 min** | — |

The structural reason is real (det's per-token / per-matmul compute is
way larger), but the asymmetry is bigger than the structural ratio
implies. **The smoke sweep at n=10 should not take an hour.** Your
job: cut the iteration loop to **≤ 5-10 min wall per qwt-on smoke
cell** so the gate-tuning loop in
[`docs/DET_GATE_TUNING_PROMPT.md`](DET_GATE_TUNING_PROMPT.md) becomes
practical.

## Stop conditions

- Required artifact missing (e.g. `qwt_d2` env doesn't exist —
  unlikely; the previous session built it).
- An assumption in this prompt is contradicted by the codebase.
- A genuinely ambiguous design choice not covered below — propose your
  pick and proceed unless the user objects.

---

## Start here — three reads + one profile

1. [`det/experiments/qwt_det_compensate.py`](../det/experiments/qwt_det_compensate.py)
   — current driver; note `inference_on_dataset` for the eval phase
   and `BackboneCalibLoader` + `calibrate_qwt` for the calib phase.
2. [`third_party/QwT-SC/QwT-vit-sc/qwt_sc/compensation.py`](../third_party/QwT-SC/QwT-vit-sc/qwt_sc/compensation.py)
   `calibrate_qwt` body — note the per-block `_forward_batched`
   pattern and the two block-0 input collections (`_collect_x0` for
   A and B).
3. [`det/sc_patch/sc_model_eva.py`](../det/sc_patch/sc_model_eva.py)
   — det's SC patcher. The SC matmul / linear hot path lives here
   (and in `sc_integration/sc_linear.py` / `sc_matmul.py` /
   `head_aligned_comp.py`).
4. **Profile** (`nsys` or just python `time.perf_counter()` around
   each phase) one fresh `qwt_det_compensate.py --n_calib 16
   --n_eval 10 --head_aligned` cell to get a phase-level breakdown:
   model load / SC patch / raw-SC eval (RCNN inference + COCO API) /
   block-0 input capture (×2) / per-block A+B ridge fits / comp eval
   (RCNN inference). The numbers will tell you where the real time
   goes.

---

## Diagnosis hypotheses (rank by likely impact)

The underlying structural slowdown (det's matmuls are 30-50× heavier
per image) is unavoidable. **The fixable slowdowns are around it.**

1. **`inference_on_dataset` is overkill for smokes.** At n=10 the per-
   image RCNN post-processing (RPN, ROI heads, NMS, COCO encode) adds
   meaningful overhead vs. raw backbone forward. Real AP at n=10 is
   noise anyway (±2-3 AP). For gate-tuning smokes, what we actually
   need is a **backbone-output MSE proxy**: load FP and SC backbones,
   forward both on the same batch, return mean ‖Y_fp − Y_sc‖²
   (averaged over admitted-block-affecting positions). This is the
   same proxy the MP search uses internally
   ([`sc_integration/mp_search.py:_compute_proxy_score`](../sc_integration/mp_search.py))
   and is **already implemented**; we just need to expose it as an
   eval mode in the driver.
2. **Block-0 input collection runs twice.** `_collect_x0` in
   `calibrate_qwt` does `model_sc(imgs)` for batch A and again for
   batch B — that's 2 × n_calib full-backbone forwards just to
   capture the first block's inputs. The blocks-1..N-1 forwards on
   those captured inputs are smaller. If the backbone is the bottleneck
   (it is on det), this 2× wastes a lot. Fix: capture block-0 inputs
   for A and B in **one fused pass** that runs `model_sc(cat([A, B]))`,
   then splits.
3. **`fwd_chunk=2` is conservative.** Det defaults to `fwd_chunk=2`
   because cls's `fwd_chunk=32` was OOM-ing at det's 1280². Profile
   peak GPU memory at chunk=4, 8 — RTX PRO 6000 Blackwell has 96 GB,
   so chunk=4 is probably safe and would halve the calib loop time.
4. **SC kernel overhead per launch.** Det's per-block has ~6 SC ops
   × 40 blocks = 240 Triton launches per forward. Each launch is
   ~50-100 µs of CPU+CUDA overhead. At 240 launches × 100 µs = 24 ms
   per image of pure dispatch overhead. Worth checking if
   `torch.compile` (or CUDA graphs) on the SC backbone can fuse these
   launches. Caveat: SC kernels may not be compile-friendly because
   they capture random Sobol state.
5. **FP backbone runs in fp32 during calib.** The FP reference
   forward in `calibrate_qwt` does its block forwards at fp32 (cast
   inside `_forward_batched`). Switching to fp16 / bf16 for the FP
   reference would ~2× the FP forward speed — but might shift the
   ridge LS targets enough to matter. Profile the eval-time impact
   before committing.
6. **Detectron2 model load at every search iteration.** The MP
   search reloads the FP model on each `iterative_swap_search`
   iteration. On det that's a ~30 s reload × `max_iters` = wasted
   minutes. Cache the FP model across iterations (CPU-resident, swap
   to GPU per per-block fp_block_dev). cls also has this redundancy
   but its model loads in <5 s so it's invisible.

---

## Concrete plan

### Step 1: profile

Add a quick `--profile` flag to `qwt_det_compensate.py` that wraps
each phase in `time.perf_counter()` and writes a single-line summary
to the log. Run **one cell** (`--n_calib 16 --n_eval 10 --head_aligned
--sc_prec 8 --sc_ops_per_block_json sensitivity/skip/skip_worst_30_int7.json`)
and extract:

```
[profile] model_load_fp=Xs  model_load_sc=Xs  patch=Xs
[profile] raw_sc_eval=Xs (incl. inference_on_dataset)
[profile] collect_x0_a=Xs  collect_x0_b=Xs
[profile] per_block_calib=Xs (40 blocks, A+B)
[profile] comp_eval=Xs
[profile] total=Xs
```

Stash the numbers in a comment at the top of `det/RESULTS.md` for
posterity. **Don't optimize blind** — this is the truth source for
which fixes matter.

### Step 2: fast-smoke proxy mode

Add `--smoke_proxy` to `qwt_det_compensate.py`. When set, replace the
two `inference_on_dataset` calls (raw-SC eval + comp eval) with:

```python
# Build a small held-out batch (n_eval images, e.g. via load_model_and_loader)
# but DON'T run RCNN — only the backbone.
y_fp_ref  = forward_backbone(model_fp.backbone.net, batch)
y_sc_raw  = forward_backbone(model_sc.backbone.net, batch)  # before comp
y_sc_comp = forward_backbone(model_sc.backbone.net, batch)  # after calib
metric = {
    "raw_mse": ((y_fp - y_sc_raw)**2).mean().item(),
    "comp_mse": ((y_fp - y_sc_comp)**2).mean().item(),
}
```

The `metric` reports MSE-to-FP, not COCO AP. **For gate tuning this
is sufficient** — comp MSE < raw MSE means the comp helped; comp
MSE > raw MSE means collapse. The diagnostic question
"which blocks did the gate admit?" is answered by `calib.per_block`
regardless of the eval metric.

Pass criterion: `--smoke_proxy` cell wall ≤ **3 min total** at
`n_calib=16 --n_eval 10` on det.

Document the trade-off: real bbox AP comparisons still need
`inference_on_dataset` (i.e. omit `--smoke_proxy` for the final
production sweep).

### Step 3: fuse the two block-0 input collections

In a private det wrapper around `calibrate_qwt` (or by extending the
`calibrate_qwt` API in the QwT-SC submodule with a backwards-
compatible `calib_loaders=(loader_a, loader_b)` tuple), pass A and B
through the SC backbone in **one** forward (concatenated) and split
the captured block-0 inputs by index. Halves the wall of the
"collecting block-0 inputs for batch {A, B}" phase.

If extending the submodule is too disruptive, do the same
optimization purely on det's side: run `model_sc.backbone.net(cat)`
yourself, capture block-0 inputs via a forward_pre_hook, split, then
hand the SC + FP block-level forwards to a re-exported lower-level
helper from `qwt_sc.compensation`. **Don't** try to refactor the
whole `calibrate_qwt` body unless the submodule extension is clean.

Pass criterion: the "collect_x0" total time should drop ~50 %.

### Step 4: bump fwd_chunk, profile memory

Test `fwd_chunk ∈ {2, 4, 8}` for the per-block forwards. Record peak
GPU memory at each. Change the det default to the largest chunk that
fits comfortably under 80 GB (RTX PRO 6000 has 96 GB; leave headroom
for the model + cuDNN workspace).

### Step 5: cache the FP model across MP-search iterations

In `det/experiments/mp_budget_swap_search.py`, `DetBackend.load_model`
gets called per `iterative_swap_search` iteration in
`sc_integration/mp_search.py`. Add an optional cache (lazy property)
so the FP model loads once per backend instance, not once per
iteration. Risk: PyTorch state mutates across iterations; verify by
running 2 iterations of the search and confirming results match
no-cache baseline.

### Step 6 (optional, bigger swing): torch.compile or CUDA graphs

Try wrapping the SC backbone forward in `torch.compile(model_sc,
mode="reduce-overhead")` and see if Triton-launch overhead drops.
Caveat: SC matmul kernels may use Python-level Sobol RNG which
breaks compilation. If `torch.compile` doesn't apply cleanly within
~1 hour, drop it; not core to the prompt.

### Step 7: validate

Re-run the gate-tuning Step 2 from
[`DET_GATE_TUNING_PROMPT.md`](DET_GATE_TUNING_PROMPT.md) using
`--smoke_proxy`. The gate decisions (which blocks `enabled=True`)
must match the slow-eval baseline; the per-block `cos_ab` and
`r2_a/b` are independent of the eval metric. Pass criterion:
admission set is identical between fast and slow modes for the same
seed.

### Step 8: update docs

- `det/README.md` "What's new" — add a "Fast-smoke mode" subsection.
- `docs/DET_GATE_TUNING_PROMPT.md` — add a note that smokes default
  to `--smoke_proxy` now; expected per-cell wall is ~5-10 min, not
  50 min.
- `det/experiments/sweep_2026-04-25.sh` — add `SMOKE_PROXY` env var
  that the harness propagates to `--smoke_proxy` for the qwt-on
  cells when set.

---

## Things to **not** do

- **Don't change the comp/calib algorithm.** Only the runtime path.
  The cross-seed gate, head-aligned comp, ridge fit etc. are out of
  scope for this prompt.
- **Don't optimize at the cost of cls.** The shared
  `qwt_sc.compensation.calibrate_qwt` and `sc_integration/mp_search.py`
  are used by both cls and det. If your fix is det-specific, put it
  in the det driver. If it's general, verify cls's smoke
  (`bash cls/experiments/sweep_int678_skip30.sh` with smoke env vars)
  still produces the same numerics.
- **Don't merge `--smoke_proxy` into the production sweep harness's
  default qwt-on cells.** Production must use real COCO inference.
  `SMOKE_PROXY` is opt-in.
- **Don't try to make det match cls's per-image timing.** The 30×
  structural gap is real and not fixable without changing the SC
  kernel architecture (out of scope).

---

## Reference timings to beat

Fresh n=10 qwt-on smoke at sw30+int8+head_aligned:

| phase | observed | target post-fix |
|---|---:|---:|
| model load (×2 — fp + sc) | ~60 s | ~60 s (unchanged) |
| SC patch | ~5 s | ~5 s |
| raw-SC eval (`inference_on_dataset`) | ~250 s | ~5 s with `--smoke_proxy` |
| block-0 capture (×2) | ~700 s | ~350 s after fusion |
| per-block calib (40 × A+B) | ~600 s | ~300 s with chunk=4 |
| comp eval (`inference_on_dataset`) | ~250 s | ~5 s with `--smoke_proxy` |
| **total wall** | **~50 min** | **~12-15 min** |

Production cell (n=5000, real AP) stays at ~6-8 h — out of scope.

---

## Existing data you can use

- `det/results/smoke_sweep_2026-04-25/*_qwton.json` — full reports
  from the slow-eval baseline. Compare your fast-mode admissions and
  per-block diagnostics against these. The `calib.per_block` field
  is what your fast-mode runs must reproduce.
- `det/logs/smoke_sweep_2026-04-25/*_qwton.log` — phase timings via
  the existing flush-print tags (search "elapsed=" etc.).
- The cls speed numbers are visible in
  `cls/results/sweep_2026-04-25/*_qwton.json` under
  `calib.elapsed_s` — useful for sanity-checking that your det
  optimizations don't have an obvious lower bound that cls already
  hit.

## Deliverables

When done, you should have:

1. A profile in `det/RESULTS.md` (or a top-of-file comment in the
   driver) showing pre/post timings per phase.
2. `--smoke_proxy` flag working in `det/experiments/qwt_det_compensate.py`
   with documented MSE-not-AP semantics.
3. Block-0 input capture fused (Step 3).
4. `fwd_chunk` default raised in `det/experiments/qwt_det_compensate.py`
   and `det/experiments/sweep_2026-04-25.sh`.
5. FP model cached across MP-search iterations
   (`det/experiments/mp_budget_swap_search.py`).
6. (Optional) torch.compile applied if it helped.
7. A validation run showing fast-mode and slow-mode produce the
   same gate decisions (admission set identical, `cos_ab` close to
   numerical equality).
8. One commit per logical optimization, pushed to `origin/main`.
   Co-author tag the same one used in recent commits.

If at any point the profile shows that >70 % of the wall time is in
SC matmul kernels (i.e. the structural lower bound), say so —
further optimization requires kernel-level work that's out of scope
for this prompt.

## Quick re-check before you write code

If you're rereading this prompt mid-task and unsure whether to stop
or continue, the answer is **continue.** This document already
authorizes every action it describes. The only legitimate stop
conditions are listed at the top.

Begin with the profile run.
