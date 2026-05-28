# Deferred from research-implement: combined min(range_mp, adaptive_mp) for ViT + det

Landed 2026-04-19. Context: `SCLinear._sc_linear_mp` already did `min(weight_sl, row_sl)` for `MPConfig` (static quantile). This pass added `AdaptiveMPConfig` as a third orthogonal input and an external timestep setter for ViT callers without a native timestep.

## Deferred — not needed for the core wiring

1. **CLI flags in `cls/eval.py` / `det/sc_eval.py`** — right now callers have to
   build `linear_mp_spec` dicts in Python. Adding `--adaptive-mp-alpha`,
   `--adaptive-mp-beta`, `--adaptive-mp-levels`, `--range-mp-threshold`,
   `--range-mp-levels`, `--vit-timestep`, `--vit-total-timesteps` would let
   shell scripts drive it. One-shot, ~40 LoC per script.

2. **QK / AV adaptive MP inside cls's `sc_attention_patch`** — the current
   attn dispatch classifies per-head with `MPConfig` only (see
   `_classify_per_head` at `cls/sc_attention_patch.py:47`). Adapting that to
   `AdaptiveMPConfig` would involve the same `(alpha, beta, progress)` math
   but over the per-head/per-row metric, not per-token. Separate PR.

3. **Adaptive MP for det's `SCMatMul`** — `det/sc_patch/sc_matmul.py` has its
   own QK/AV matmul and doesn't go through SCLinear. Same separate change.

4. **Training/calibrating (α, β) for ViT** — `experiments/eval_adaptive_mp.py`
   already learns per-(op, block) sensitivity allocators for diffusion. Porting
   that to ViT needs an ImageNet-style classification loss surrogate. Not
   touched here.

5. **Per-block adaptive_mp_cfg override** — the current plumb passes one
   AdaptiveMPConfig per operator to *every* block. If you want per-block (op,
   block) specialization, either extend `_normalize_linear_mp_spec` to accept
   nested block-indexed dicts, or wire through the per-block schedule already
   used by `sc_ops_per_block`.

## Assumptions the implementation baked in

- **Adaptive wins over fixed**: when both `mp_cfg` and `adaptive_mp_cfg` are set
  on the same `SCLinear`, adaptive is used (the fixed MPConfig is ignored).
  Rationale: adaptive is strictly more expressive and matches the paper's
  reported gains. User can reverse by passing only one of them.
- **Timestep state is module-level, not per-instance**. Every SCLinear with an
  adaptive config reads from the same `(t, T)` pair in
  `sc_integration.sc_linear`. Fine for a single-stream forward; if you start
  running two models with different schedules in the same process, that's when
  you'd need instance-level state.
- **Diffusion path unchanged**: Q-DiT's `SCController` / `SCMlp` / `SCAttention`
  untouched. The new adaptive-MP-in-SCLinear is orthogonal and only activates
  when a caller opts in via `linear_mp_spec`.
- **`(0, 1)` is the neutral setting** for ViT callers that don't care about
  timestep — progress=0, so thresholds are just β.
