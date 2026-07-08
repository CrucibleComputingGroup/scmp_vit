"""
Per-block, per-operator, per-group SC precision configuration.

This module provides the SCPrecisionMap data structure that controls
stochastic computing precision at fine granularity:
- Per block: each transformer block can have different SC settings
- Per operator: qk, av, proj, mlp_fc1, mlp_fc2, input_proj
- Per group: each weight quantization group (linear) or head (BMM) can
  have its own stoc_len for early termination

The core mechanism is early termination: shorter stoc_len = fewer loop
iterations = proportional speedup, with the guarantee that every group's
stoc_len <= max stoc_len (so total work <= uniform max precision).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Optional

OPERATORS = ("qk", "av", "proj", "mlp_fc1", "mlp_fc2", "input_proj")


@dataclass
class OperatorConfig:
    """Configuration for a single operator in a single block.

    Attributes:
        enabled: Whether SC is enabled for this operator.
        stoc_len: Default stochastic stream length for this operator.
        timewise: Per-operator timewise fraction (0-1). SC is used for
                  the first timewise * total_timesteps diffusion steps.
        group_stoc_lens: Per-group stoc_lens for mixed precision.
            - For linear ops: one stoc_len per weight quantization group.
            - For BMM ops (qk, av): one per attention head.
            - If None, all groups use self.stoc_len (uniform precision).
    """
    enabled: bool = False
    stoc_len: int = 256
    timewise: float = 0.0
    group_stoc_lens: Optional[list[int]] = None

    def sc_prec_for_stoc_len(self, stoc_len: int) -> int:
        """Get sc_prec (ceil(log2(stoc_len))) for a given stoc_len."""
        if stoc_len <= 1:
            return 1
        return math.ceil(math.log2(stoc_len))

    @property
    def sc_prec(self) -> int:
        return self.sc_prec_for_stoc_len(self.stoc_len)

    def distinct_stoc_lens(self) -> list[int]:
        """Return sorted list of distinct stoc_len values used by this operator."""
        if self.group_stoc_lens is None:
            return [self.stoc_len]
        return sorted(set(self.group_stoc_lens), reverse=True)


class SCPrecisionMap:
    """Per-block, per-operator, per-group SC precision config.

    Args:
        total_blocks: Number of transformer blocks in the model.
        default_stoc_len: Default stoc_len for all operators (256 = int8).
        default_timewise: Default timewise fraction for all operators.
    """

    def __init__(
        self,
        total_blocks: int,
        default_stoc_len: int = 256,
        default_timewise: float = 0.0,
    ):
        self.total_blocks = total_blocks
        self.default_stoc_len = default_stoc_len
        self.default_timewise = default_timewise
        # _configs[block_idx][operator] = OperatorConfig
        self._configs: list[dict[str, OperatorConfig]] = [
            {op: OperatorConfig(stoc_len=default_stoc_len, timewise=default_timewise)
             for op in OPERATORS}
            for _ in range(total_blocks)
        ]

    # --- Core API ---

    def get(self, block_idx: int, operator: str) -> OperatorConfig:
        """Get config for a specific block and operator."""
        assert 0 <= block_idx < self.total_blocks, f"block_idx {block_idx} out of range [0, {self.total_blocks})"
        assert operator in OPERATORS, f"Unknown operator '{operator}', expected one of {OPERATORS}"
        return self._configs[block_idx][operator]

    def set(
        self,
        block_idx: int,
        operator: str,
        *,
        enabled: Optional[bool] = None,
        stoc_len: Optional[int] = None,
        timewise: Optional[float] = None,
        group_stoc_lens: Optional[list[int]] = None,
    ):
        """Set config fields for a specific block and operator."""
        cfg = self.get(block_idx, operator)
        if enabled is not None:
            cfg.enabled = enabled
        if stoc_len is not None:
            cfg.stoc_len = stoc_len
        if timewise is not None:
            cfg.timewise = timewise
        if group_stoc_lens is not None:
            cfg.group_stoc_lens = group_stoc_lens

    # --- Batch API for algorithms ---

    def enable_operator(
        self,
        operator: str,
        block_indices: list[int],
        stoc_len: Optional[int] = None,
        timewise: Optional[float] = None,
    ):
        """Enable an operator for a list of blocks."""
        for idx in block_indices:
            self.set(idx, operator, enabled=True,
                     stoc_len=stoc_len, timewise=timewise)

    def set_group_stoc_lens(
        self,
        block_idx: int,
        operator: str,
        group_stoc_lens: list[int],
    ):
        """Set per-group stoc_lens for an operator in a block."""
        self.set(block_idx, operator, group_stoc_lens=group_stoc_lens)

    # --- Convenience (backward compat) ---

    def enable_operator_fraction(
        self,
        operator: str,
        fraction: float,
        stoc_len: Optional[int] = None,
        timewise: Optional[float] = None,
        reverse: bool = False,
    ):
        """Enable operator for a fraction of blocks (first N or last N).

        Args:
            operator: Operator name.
            fraction: Fraction of blocks to enable (0-1).
            stoc_len: Stochastic stream length (None = keep default).
            timewise: Per-operator timewise (None = keep default).
            reverse: If True, enable last N blocks instead of first N.
        """
        n = int(fraction * self.total_blocks)
        if n == 0:
            return
        if reverse:
            indices = list(range(self.total_blocks - n, self.total_blocks))
        else:
            indices = list(range(n))
        self.enable_operator(operator, indices, stoc_len=stoc_len, timewise=timewise)

    def get_enabled_blocks(self, operator: str) -> list[int]:
        """Return list of block indices where operator is enabled."""
        return [i for i in range(self.total_blocks) if self._configs[i][operator].enabled]

    # --- Serialization ---

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return {
            "total_blocks": self.total_blocks,
            "default_stoc_len": self.default_stoc_len,
            "default_timewise": self.default_timewise,
            "blocks": [
                {op: asdict(cfg) for op, cfg in block.items()}
                for block in self._configs
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> SCPrecisionMap:
        """Deserialize from a plain dict."""
        pm = cls(
            total_blocks=d["total_blocks"],
            default_stoc_len=d.get("default_stoc_len", 256),
            default_timewise=d.get("default_timewise", 0.0),
        )
        for i, block_dict in enumerate(d["blocks"]):
            for op, cfg_dict in block_dict.items():
                if op not in OPERATORS:
                    continue
                pm._configs[i][op] = OperatorConfig(**cfg_dict)
        return pm

    def to_json(self, path: str):
        """Serialize to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> SCPrecisionMap:
        """Deserialize from JSON file."""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_legacy_args(cls, args, total_blocks: int) -> SCPrecisionMap:
        """Build a SCPrecisionMap from legacy CLI arguments.

        Maps the old flat layerwise fractions + global timewise/sc_prec
        into the per-block, per-operator structure.
        """
        sc_prec = getattr(args, "sc_prec", 8)
        stoc_len = 2 ** sc_prec
        timewise = getattr(args, "timewise", 0.0)
        reverse = getattr(args, "reverse_layerwise", False)

        skip_str = getattr(args, "sc_skip_blocks", "")
        skip_blocks = set()
        if skip_str:
            skip_blocks = {int(x.strip()) for x in skip_str.split(",") if x.strip()}

        pm = cls(total_blocks=total_blocks, default_stoc_len=stoc_len, default_timewise=timewise)

        # Map legacy layerwise fractions to enabled blocks
        op_fractions = {
            "qk": getattr(args, "qklayerwise", 0.0),
            "av": getattr(args, "avlayerwise", 0.0),
            "proj": getattr(args, "projlayerwise", 0.0),
            "mlp_fc1": getattr(args, "mlplayerwise", 0.0),
            "mlp_fc2": getattr(args, "mlplayerwise", 0.0),
            "input_proj": getattr(args, "inputprojlayerwise", 0.0),
        }

        for op, frac in op_fractions.items():
            if frac <= 0:
                continue
            pm.enable_operator_fraction(
                op, frac, stoc_len=stoc_len, timewise=timewise, reverse=reverse,
            )

        # Remove skip blocks
        for block_idx in skip_blocks:
            if 0 <= block_idx < total_blocks:
                for op in OPERATORS:
                    pm._configs[block_idx][op].enabled = False

        return pm

    # --- Analysis ---

    def summary(self) -> str:
        """Human-readable summary of the precision map."""
        lines = [f"SCPrecisionMap(total_blocks={self.total_blocks}, default_stoc_len={self.default_stoc_len})"]
        for op in OPERATORS:
            enabled = self.get_enabled_blocks(op)
            if not enabled:
                lines.append(f"  {op}: disabled")
                continue
            stoc_lens = set()
            has_mixed = False
            for idx in enabled:
                cfg = self._configs[idx][op]
                stoc_lens.add(cfg.stoc_len)
                if cfg.group_stoc_lens is not None:
                    has_mixed = True
                    stoc_lens.update(cfg.group_stoc_lens)
            lines.append(
                f"  {op}: enabled in {len(enabled)}/{self.total_blocks} blocks, "
                f"stoc_lens={sorted(stoc_lens, reverse=True)}, "
                f"mixed_groups={'yes' if has_mixed else 'no'}"
            )
        return "\n".join(lines)

    def total_stoc_budget(self) -> int:
        """Sum of all enabled stoc_lens across all blocks and operators.

        Useful for comparing against uniform int8 budget (total_blocks * 6 * 256).
        """
        total = 0
        for i in range(self.total_blocks):
            for op in OPERATORS:
                cfg = self._configs[i][op]
                if not cfg.enabled:
                    continue
                if cfg.group_stoc_lens is not None:
                    total += sum(cfg.group_stoc_lens)
                else:
                    total += cfg.stoc_len
        return total

    def __repr__(self) -> str:
        return self.summary()
