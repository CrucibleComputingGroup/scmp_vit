"""Lightweight objective transforms for SC mixed-precision calibration.

Copied verbatim (numerical contract included) from
``scmp_llm/benchmark/ppl/mp_objectives.py`` so an allocation objective means the
same thing in both repos. Deliberately free of torch / model imports, so the
contract is unit-testable without the SC runtime.
"""

from __future__ import annotations

import numpy as np


def objective_errors(errors: np.ndarray, objective: str) -> np.ndarray:
    """Map measured relative-L2 sigma to the allocator's objective currency.

    ``errors[:, 0]`` is the error at the highest stream length because MP
    levels are stored in strictly descending order. ``delta_sigma2`` prices
    only the additional error introduced by shortening a row, then squares it
    to penalize the low-precision cliff superlinearly::

        max(0, sigma(level) - sigma(max_level)) ** 2

    Preserves the input dtype so the ``sigma`` path stays byte-identical to the
    pre-objective implementation (notably float32 threshold ties).
    """
    e = np.asarray(errors)
    if e.ndim != 2:
        raise ValueError(
            f"objective errors must be 2D [rows, levels], got {e.shape}")
    if objective == "sigma":
        return e
    if objective == "sigma2":
        return e ** 2
    if objective == "delta_sigma2":
        delta = np.maximum(e - e[:, :1], 0.0)
        return delta ** 2
    raise ValueError(f"unknown allocation objective: {objective!r}")
