# Auto-MP Calibration — Implementation Assumptions

Decisions made during implementation of phases 1-5 (commit: upcoming).
Each bullet is reversible with one edit.

## Defaulted choices

- **Codex advisor (Step 3.5 / 5.5 of research-implement skill)**: skipped —
  `mcp__codex__codex` tool was not available in this session's deferred
  tools. Proceed-without-Codex is an explicit skill fallback.
- **Ridge for oracle scoring**: `1e-2` (matches QwT default in
  `third_party/QwT-SC/QwT-vit-sc/qwt_sc/compensation.py`). Configurable via
  `--ridge` (shared with QwT calib).
- **Candidate grid size**: `n_candidates=10` per 1D line search
  (`--auto_mp_n_candidates`). Coord descent with 10 points × 2 rounds × 4
  ops × 24 blocks ≈ 2k forwards ≈ 10-15 min wall on ViT-L.
- **Outer rounds**: `outer_per_op=2`, `outer_block=1`. Phase-2 smoke shows
  second outer round still improves, so leaving at 2. Block-level coord
  descent disabled by default (1 pass) to keep cost tractable; increase
  with `--auto_mp_outer_block` if ops interact strongly.
- **QK per-head boundaries**: in auto mode, QK also goes through
  `FreeBoundaryMPConfig` (H=16 is small but the same k-1 boundaries
  mechanism works). Non-auto mode keeps the static `MPConfig` quantile
  for QK as before.
- **Pre-hook retention**: the block-idx pre-hooks installed by
  `auto_calibrate_mp` are **intentionally left live** after calibration so
  runtime forward reads the correct `_CURRENT_BLOCK_IDX`. They are idempotent
  (set an int, no side effect) and add negligible overhead.
- **Smoke target**: unit-level only in this run — synthetic 2-24 block
  transformer-like stacks in 3 test scripts, all on CPU, each < 30s.
  Full DINOv2 ViT-L/14 smoke via `experiments/test_auto_mp_calibrate.sh`
  is the scale-up command; not run here because it requires GPU +
  ImageNet parquet data and is >> 60s.
- **enabled flag semantics**: aligned to QwT —
  `enabled=True` iff `block_idx >= start_block` and `r2 > 0`.

## Scope cuts (deferred)

- **MLP prediction head (v2)**: per plan, no MLP in v1. Boundaries are
  stored as a static `{(block, op): tensor}` dict in the
  `FreeBoundaryMPConfig`; MLP that predicts these from per-batch features
  is future work.
- **Compute-budget constraint**: auto-MP now supports a target
  `--auto_mp_budget_ratio` via blockwise carried budget. Candidate boundaries
  are ranked by feasibility first (actual compute <= carried block target),
  then by residual score within the feasible set.
- **Caching calibrated boundaries**: config object is not yet pickled to
  disk. For repeat runs with same (sc_config, n_calib) the search runs
  again (~10-15 min). Add `--auto_mp_cache_path` when needed.
- **Per-candidate SC comp factory**: oracle scoring uses closed-form ridge
  residual (fast, dense FP comp math). The *final* installed comp uses the
  user-configured `comp_factory` (which may be SC). Per-candidate SC comp
  would 10× the cost for marginal signal — skipped per plan.
- **QwT parity**: `comp_factory_variants` and `comp_refit_iters` are threaded
  into `auto_calibrate_mp`, so the auto-MP path now supports per-block comp
  variant selection and SC-comp noise-aware refit like the standard QwT path.

## Follow-up (next steps in suggested order)

1. Run `experiments/test_auto_mp_calibrate.sh` on the real DINOv2 pipeline
   with `N_CALIB=128 N_EVAL=100`, compare top1 vs the fractions baseline.
2. If auto < fractions baseline, increase `--auto_mp_n_candidates 16` or
   `--auto_mp_outer_block 2`.
3. Persist boundaries per (sc_config, seed, n_calib) to skip re-search.
4. If dynamic adaptation becomes useful, move to v2 MLP that predicts
   boundaries from per-batch metric statistics (trained on these oracle
   outputs as imitation targets).
