# deprecated/ — retired in-tree SC stack

Archived 2026-07-25. These two packages were vit_sc's own fork of the SC
kernels and the model-integration layer:

| dir | what it was |
|---|---|
| `sc/` | in-tree SC primitives — `sc_triton.py` (154 KB of Triton kernels), `mp_config.py`, `mp_auto_calibrator.py`, `rng.py`, `sng.py`, `config_helpers.py`, `lfsr_taps.py` |
| `sc_integration/` | model-side wrappers — `sc_linear.py`, `sc_attention.py`, `sc_mlp.py`, `mp_search.py`, `mp_ilp.py`, `noise_matmul.py`, … |

## Replaced by

`vit_sc/kernels/` (our own `scmp_kernels` checkout, branch `sc/precision-trace`)
plus `vit_sc/model/sc_vit.py`. The kernel package is the shared, maintained
version used by `scmp_llm` and `scmp_cnn`; `sc/mp_config.py` here was an older
copy of `scmp_kernels/mp/config.py` that had drifted.

Mapping of the entry points:

| retired | replacement |
|---|---|
| `sc_triton.sc_matmul_grouped_enable_triton(a, b, group_a, group_b, …)` | `sc_matmul(a, b, granularity="per_row", group_a=…, group_b=…)` |
| `sc_triton.sc_matmul_enable_batched_bipolar(q, k, q_max, q_min, k_max, k_min, …)` | `sc_matmul(q, k, granularity="per_head")` — ranges computed inside |
| `sc_triton.sc_matmul_enable_triton_mlp(…, chunk_d=…)` | `sc_matmul(…, granularity="per_row", chunk_d=…)` |
| `sc_integration.sc_linear.SCLinear` | `model.sc_vit.SCLinear` |
| `sc.mp_config.*`, `sc_integration.mp_linear.*` | `scmp_kernels.mp.*` |

## Behaviour that did NOT carry over

`AdaptiveMPConfig` in `scmp_kernels` is **data-driven only** — a calibrated
threshold table, quantile `target_fractions`, or free boundaries. The old
closed-form `threshold = alpha * progress + beta` mode was deliberately removed
upstream, so `--adaptive_mp_alpha` / `--adaptive_mp_beta` are gone from
`cls/eval.py`; use `--adaptive_mp_table` or `--adaptive_mp_fractions`.

## Still importing this code

`det/` and the top-level `experiments/` have not been migrated yet — they still
`import sc_integration` / `from sc_triton import …`. To run them unchanged:

```bash
export PYTHONPATH="$PWD/deprecated:$PWD/deprecated/sc:$PYTHONPATH"
```

Note that path puts the OLD kernels back in play for those scripts. Do not mix
it into a `cls/` run.
