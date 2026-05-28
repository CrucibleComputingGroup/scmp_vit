"""Mixed-precision SC support for Linear ops, ported from ../scmp_llm.

Two orthogonal axes of MP, both consumable by SCLinear:

  * Fixed MP (``MPConfig``) — classifies *input rows* of the activation by
    per-row abs-max and buckets them into ``stoc_len`` levels. Higher-magnitude
    rows get longer bit-streams, colder rows get shorter. Classification runs
    each forward pass.

  * Range MP (``RangeMPConfig``) — classifies *output row groups* of the weight
    by per-group (max - min) range. Built once at patch time from the
    already-quantized weight, baked into a list of ``DispatchEntry``. Each
    entry holds a contiguous weight slice and the output indices it produces.

Both can be combined: per (input-row, output-group) pair the effective
``stoc_len`` is ``min(row_sl, weight_group_sl)``.

The canonical ``MPConfig`` / ``RangeMPConfig`` / classifier implementations
live in ``sc/mp_config.py`` and are re-exported here.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch

# Canonical dataclasses live in sc/mp_config.py (local copy of scmp_llm's MP).
HERE = Path(__file__).resolve().parent
SC_PATH = HERE.parent / "sc"
if str(SC_PATH) not in sys.path:
    sys.path.insert(0, str(SC_PATH))

from scmp_kernels.mp import (  # type: ignore
    MPConfig,
    RangeMPConfig,
    AdaptiveMPConfig,
    FreeBoundaryMPConfig,
    RowAssignment,
    classify_rows_by_metric,
    classify_groups_by_range,
    adaptive_classify_rows,
    set_current_block_idx,
    get_current_block_idx,
)

__all__ = [
    "MPConfig",
    "RangeMPConfig",
    "AdaptiveMPConfig",
    "FreeBoundaryMPConfig",
    "DispatchEntry",
    "build_range_dispatch",
    "classify_input_rows",
    "classify_input_rows_adaptive",
    "sc_prec_for_stoc_len",
    "set_current_block_idx",
    "get_current_block_idx",
]


def sc_prec_for_stoc_len(stoc_len: int) -> int:
    """Smallest ``sc_prec`` such that ``2**sc_prec >= stoc_len``."""
    if stoc_len <= 1:
        return 1
    return int(math.ceil(math.log2(stoc_len)))


@dataclass
class DispatchEntry:
    """One (stoc_len, weight-slice, output-indices) dispatch bucket."""
    stoc_len: int
    sc_prec: int
    weight: torch.Tensor        # (n_out_in_bucket, D_in), contiguous
    out_indices: torch.Tensor   # LongTensor of output row indices into the full weight


def build_range_dispatch(
    weight: torch.Tensor,
    cfg: RangeMPConfig,
    operator: Optional[str] = None,
    group_size: int = 0,
) -> List[DispatchEntry]:
    """Split the output rows of ``weight`` into contiguous groups, assign each
    a ``stoc_len`` from its (max - min) range via ``classify_groups_by_range``.

    Groups that share a ``stoc_len`` are merged into a single
    ``DispatchEntry`` so the forward pass launches one kernel per distinct
    level rather than per group.
    """
    out_features = weight.shape[0]
    g = group_size if 0 < group_size < out_features else out_features
    num_groups = out_features // g

    group_stoc_lens = classify_groups_by_range(
        weight, group_size=g, config=cfg, operator=operator,
    )
    assert len(group_stoc_lens) == num_groups, (
        f"classify_groups_by_range returned {len(group_stoc_lens)} levels "
        f"for {num_groups} groups")

    by_level: dict[int, list[int]] = {}
    for gi, sl in enumerate(group_stoc_lens):
        rows = list(range(gi * g, (gi + 1) * g))
        by_level.setdefault(int(sl), []).extend(rows)

    entries: List[DispatchEntry] = []
    for sl, rows in by_level.items():
        idx = torch.as_tensor(rows, dtype=torch.long, device=weight.device)
        entries.append(DispatchEntry(
            stoc_len=int(sl),
            sc_prec=sc_prec_for_stoc_len(int(sl)),
            weight=weight.index_select(0, idx).contiguous(),
            out_indices=idx,
        ))
    return entries


def classify_input_rows(x_flat: torch.Tensor, cfg: MPConfig) -> RowAssignment:
    """Bucket flattened-activation rows by abs-max into MP levels."""
    metric = x_flat.float().abs().amax(dim=-1)  # (M,)
    return classify_rows_by_metric(
        metric, cfg.stoc_len_levels, cfg.level_fractions,
    )


def classify_input_rows_adaptive(
    x_flat: torch.Tensor,
    cfg: AdaptiveMPConfig,
    timestep: int,
    total_timesteps: int,
    operator: Optional[str] = None,
) -> RowAssignment:
    """Adaptive (timestep-aware) per-row classification for SCLinear.

    Same row metric (abs-max) as ``classify_input_rows`` but routes through
    ``adaptive_classify_rows`` with per-operator (α, β) and a progress term
    ``t / max(T-1, 1)``. ViT callers that have no native timestep can set
    ``(timestep, total_timesteps)`` externally via ``sc_linear.set_vit_timestep``.
    """
    metric = x_flat.float().abs().amax(dim=-1)  # (M,)
    return adaptive_classify_rows(
        metric, timestep, total_timesteps, cfg, operator=operator,
    )
