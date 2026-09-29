# SC-Vision Task: migrate vit_sc onto scmp_kernels, adapt cls

**Date:** 2026-07-25  **Node:** gl1804 (RTX PRO 6000 Blackwell, 98 GB)
**Seeds:** `--seed 0` everywhere (torch / numpy / random)

## Setup

- Task: ImageNet-1k classification, DINOv2 ViT-L/14 + linear head, 24 blocks.
- Data: `/scratch/nbleier_owned_root/nbleier_owned1/shared_data/imagenet/data`
  (50 000 val images, 14 parquet shards); smoke runs use a seeded 512-image subset.
- SC config: `sc_prec=8`, bipolar, `chunk_d=128`, `halve=0` (nominal cycles),
  `stoc_len ∈ {256, 64}`.

## What Was Done

1. **vit_sc now maintains its own kernel package.** `vit_sc/kernels` is a git
   submodule of `CrucibleComputingGroup/scmp_kernels`, branch
   `sc/precision-trace`, pinned at `01f83579` — the same commit `scmp_llm`
   pins, but independently updatable. `.gitmodules` records the branch as
   `sc/precision-trace` (scmp_llm's says `main` while being checked out at
   precision-trace, so `--remote` there would move it backwards; ours is
   self-consistent).
2. **Resolved a live shadowing hazard.** `pip show scmp_kernels` reports an
   editable install pointing at `scmp_cnn/kernels` (branch `main` + uncommitted
   edits), so a bare `import scmp_kernels` picked up the *CNN* copy, which has
   no `trace.py` and none of the precision-trace MP features.
   `model/_bootstrap.py` prepends `vit_sc/kernels` to `sys.path`, and every run
   prints which copy won (`[env] scmp_kernels vit_sc-local: …`).
3. **New integration layer `model/sc_vit.py`** — `SCLinear`, SC attention, and
   `patch_model`, all on the unified `scmp_kernels.sc_matmul`.
4. **Retired the in-tree SC stack.** `sc/` and `sc_integration/` moved to
   `deprecated/` via `git mv` (history preserved), with an API mapping table.
5. **`cls/eval.py` adapted**, gaining `--stoc_len`, `--chunk_d`, `--halve`.
6. Tests + docs: `tests/test_sc_vit_smoke.py`, `cls/run_smoke.sh`, `CLAUDE.md`,
   `deprecated/README.md`.

### Kernel call mapping

| retired | now |
|---|---|
| `sc_matmul_enable_batched_bipolar(q,k,q_max,q_min,k_max,k_min,…)` | `sc_matmul(q, k, granularity="per_head")` — ranges computed in-kernel |
| `sc_matmul_grouped_enable_triton(a,b,group_a,group_b,…)` | `sc_matmul(a, b, granularity="per_row", …)` |
| `sc_matmul_enable_triton_mlp(…, chunk_d)` | `sc_matmul(…, granularity="per_row", chunk_d=…)` |
| `sc_integration.sc_linear.SCLinear` | `model.sc_vit.SCLinear` |
| `sc.mp_config.*` / `sc_integration.mp_linear.*` | `scmp_kernels.mp.*` |

`attn@V` is now a **single batched 3D call** for all (B,H); the legacy path
looped `B*H` times in Python.

## Files Created/Modified

- `model/sc_vit.py` — SC integration (new)
- `model/_bootstrap.py` — kernel path pinning (new)
- `cls/eval.py` — migrated to `model.sc_vit`; new budget flags
- `cls/sc_attention_patch.py` — reduced to a compat shim
- `cls/run_smoke.sh`, `tests/test_sc_vit_smoke.py` — new
- `CLAUDE.md`, `deprecated/README.md` — new
- `.gitmodules`, `.gitignore` — kernels submodule, `.torch_cache/`
- `sc/` → `deprecated/sc/`, `sc_integration/` → `deprecated/sc_integration/`

## Results

### Unit-level (synthetic ViT, `tests/test_sc_vit_smoke.py`) — PASS

Vendored kernels confirmed in use; `patch_model` swapped 12 `SCLinear` + 3
attention forwards; all-SC rel-err **0.0104** at `stoc_len=256` with 100 %
argmax agreement; error monotone in `stoc_len`
(256→0.0080, 128→0.0249, 64→0.0379, 32→0.0583, 16→0.1070); fixed / adaptive /
range MP paths all execute; trace recorded 12 groups, 103.8 M MACs.

### End-to-end (DINOv2 ViT-L/14, 512-image ImageNet subset)

| config | ops on SC | stoc_len | top-1 | top-5 | img/s |
|---|---|---|---|---|---|
| fp | — | — | 0.8438 | 0.9863 | 177.3 |
| sc_qk_sl256 | qk | 256 | 0.8438 | 0.9863 | 96.4 |
| sc_attn_sl256 | qk, av | 256 | 0.8340 | 0.9844 | 72.7 |
| sc_all_sl256 | qk, av, mlp_fc1/2, proj | 256 | 0.8379 | 0.9824 | 5.6 |
| sc_all_sl64 | qk, av, mlp_fc1/2, proj | 64 | 0.8184 | 0.9707 | 6.9 |

Per-op spot checks at N=256, `stoc_len=256`: av-only 0.8359, mlp_fc2-only 0.8438.

**These deltas are NOT yet measurements.** At N=512 and top-1≈0.84 the binomial
standard error is ±1.6 pt, so every SC row above sits within ~1σ of FP except
`sc_all_sl64` (−2.5 pt ≈ 1.6σ). `sc_all_sl256` scoring *above* `sc_attn_sl256`
is noise, not a finding. The purpose here was pipeline validation; quoting
accuracy requires a full 50 k val run.

Throughput note: `stoc_len` barely moves wall-clock (5.6 vs 6.9 img/s for
256 vs 64) — consistent with the sibling repos, where SC cost is dominated by
`cum_indicator` build + Triton launch overhead, not stream length. `stoc_len`
is an energy/area knob, not a GPU-latency knob.

### The old "SC on MLP collapses to ~0%" result — resolved

`sc_all_sl256` (all 5 ops × all 24 blocks, no skipping) scoring 0.8379
contradicted two documented results: README's `--sc_mlp 1` "**collapses to ~0%
accuracy** at uniform prec=8" and the `everything | 120 ops | 0.000 |
catastrophic` row of the fine-grained scheduling table. That was worth
disproving rather than explaining away, since a silent FP fallback would look
exactly like a fix.

**Coverage is real** — `SC_MP_TRACE` over the all-ops run shows every op in
every block issuing SC matmuls, with MAC counts matching the analytic values
(mlp_fc1: 257 × 1024 × 4096 × 32 imgs × 24 blocks = 827.9 G ✓):

| op | calls | blocks | MACs |
|---|---:|---:|---:|
| av | 48 | 24 | 51,943,047,168 |
| mlp_fc1 | 48 | 24 | 827,854,946,304 |
| mlp_fc2 | 48 | 24 | 827,854,946,304 |
| out_proj | 48 | 24 | 206,963,736,576 |
| qk | 48 | 24 | 51,943,047,168 |
| qkv_proj | 48 | 24 | 620,891,209,728 |

**The cause is `chunk_d`.** A/B at N=256, SC on `mlp_fc1`+`mlp_fc2` only:

| `chunk_d` | top-1 | top-5 |
|---|---|---|
| 128 (new default) | **0.8438** | 0.9805 |
| 0 (legacy behaviour) | **0.0000** | 0.0156 |

The legacy `SCLinear` defaulted to `chunk_d=0` and the legacy `patch_model`
never passed the argument, so every historical MLP-SC run used the unchunked
path. The collapse is a property of **unchunked SC on wide layers**, not of SC
on MLP — and it reproduces exactly on the new kernel when chunking is off, so
this is a config difference, not an upstream bug fix.

Consequence: **the skip / per-block-scheduling conclusions were partly
compensating for an unchunked MLP path.** The `sc_skip_worst*` curve and the
"knee at ~100 SC ops" should be re-derived with `chunk_d=128` before being
carried into any writeup. Attention-only results are unaffected (old
`attn_full` 0.832 vs new `sc_attn_sl256` 0.8340 — consistent).

`SCLinear` now raises a `RuntimeWarning` when built with `chunk_d=0` on a layer
of width ≥ 1024, since the failure is otherwise silent.

The Owen scramble was ruled out as a second collapse channel. At the ViT level
`SC_OWEN_MODE=bitrev` and `off` both give 0.8359 (N=256, all ops); at the kernel
level (`tests/check_owen_knob.sh`) `bitrev`/`off`/`random` are **bit-identical**
at `stoc_len=256` — the Sobol sequence is fully enumerated so scrambling is a
no-op — and only diverge at short streams (`sl=16`: bitrev 0.3195, off 0.4841,
random 0.3113). The knob is live, it just does not act at full stream length.
It will matter for MP ladders with low rungs.

## Issues & Notes

- **gl1810 was unavailable.** Both nodes answered at session start, but the
  gl1810 job ended mid-session and `pam_slurm_adopt` then refused SSH. Only
  gl1804 (job 54927808) remained, so everything ran single-GPU and serial. I
  did not request a new allocation — that is the user's call. With gl1810 back,
  `cls/run_smoke.sh` cells are independent and can be split across both.
- **Fixed a misleading log line** in `cls/eval.py`: it echoed the raw
  `--sc_av` / `--sc_mlp` flags, so a `--sc_ops qk,av,mlp_fc1,mlp_fc2,proj` run
  printed `av=False mlp=False` while actually patching all of them. It now
  prints the resolved op set. The first smoke log
  (`results/smoke/driver.log`) predates the fix — its `[sc]` lines understate
  the surface; the `patched={…}` dict in the same line was always correct.
- **Semantic change users must know:** `scmp_kernels`'s `AdaptiveMPConfig` is
  data-driven only; upstream removed the `threshold = α·progress + β`
  closed-form mode. `--adaptive_mp_alpha/beta` are gone, replaced by
  `--adaptive_mp_table` (calibrated) or `--adaptive_mp_fractions` (quantile).
- **Not migrated:** `det/`, top-level `experiments/`, and 5 scripts under
  `cls/experiments/`. They need `HeadAlignedSCLinear`, `sc_prec_for_stoc_len`,
  `set_noise_corrections`, which have no scmp_kernels equivalent. They still
  run via `PYTHONPATH=deprecated:deprecated/sc` (documented in both
  `CLAUDE.md` and `deprecated/README.md`).
- Nothing has been committed or pushed; the restructure sits staged/untracked
  in the working tree for review.

## Next Steps

1. Full 50 k-image FP and SC runs to turn the table above into real numbers.
2. Calibrate an MP threshold table for ViT (`calibrate_mp_thresholds.py`
   equivalent) — this also unlocks the per-op ρ-selected dispatch metrics
   (`amax`/`l2`/`crest` + sign), worth +4.3 pt top-1 on scmp_cnn at an
   iso-compute int7 budget.
3. Decide whether ViT adopts `--halve 1` as canonical (scmp_llm does), which
   would halve the cycle budget at equal quality but changes what a quoted
   budget number means.
4. Migrate `det/` onto `model/sc_vit.py`, then delete `deprecated/`.
