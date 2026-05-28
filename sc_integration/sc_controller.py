"""
SC Controller for managing timewise and layerwise stochastic computing control.

This module provides the SCController class that determines when and where
to use stochastic computing based on:
- timewise: fraction of timesteps to use SC (early timesteps with high noise)
- Per-op layerwise: fraction of blocks to use SC for each operator type
  (qk, av, output projection, mlp_fc1, mlp_fc2, input projection)
- Per-group precision: mixed stoc_len per weight group or attention head
"""
from __future__ import annotations

import math
import torch
from typing import Optional

from .sc_precision_map import SCPrecisionMap, OperatorConfig, OPERATORS
from .mp_config import MPConfig, AdaptiveMPConfig, RangeMPConfig


class SCController:
    """
    Controller for stochastic computing in diffusion transformer.

    Manages two dimensions of SC control:
    1. Timewise: Use SC for early diffusion timesteps (more noise-tolerant)
    2. Layerwise (per-op): Use SC for specific operations in specific blocks

    Can be initialized with either:
    - Legacy flat args (timewise, qklayerwise, etc.) for backward compat
    - A SCPrecisionMap for fine-grained per-group control

    Args:
        timewise: Float (0-1), fraction of timesteps to use SC.
        qklayerwise: Float (0-1), fraction of blocks to use SC for q@k^T.
        avlayerwise: Float (0-1), fraction of blocks to use SC for attn@v.
        projlayerwise: Float (0-1), fraction of blocks to use SC for output projection.
        mlplayerwise: Float (0-1), fraction of blocks to use SC for MLP (fc1, fc2).
        inputprojlayerwise: Float (0-1), fraction of blocks to use SC for QKV input projection.
        total_timesteps: Total number of diffusion timesteps.
        total_blocks: Total number of transformer blocks in the model.
        sc_prec: Stochastic computing precision (stoc_len = 2^sc_prec).
        precision_map: Optional SCPrecisionMap for fine-grained control.
                       If provided, overrides layerwise args.
    """

    def __init__(
        self,
        timewise: float,
        qklayerwise: float,
        total_timesteps: int,
        total_blocks: int,
        sc_prec: int = 8,
        avlayerwise: float = 0.0,
        projlayerwise: float = 0.0,
        mlplayerwise: float = 0.0,
        inputprojlayerwise: float = 0.0,
        reverse_layerwise: bool = False,
        sc_skip_blocks: Optional[set] = None,
        sc_enable: bool = False,
        noise_model: bool = False,
        noise_local_correction: float = 0.15,
        noise_global_correction: float = 0.60,
        precision_map: Optional[SCPrecisionMap] = None,
    ):
        # Validate parameters
        assert 0.0 <= timewise <= 1.0, f"timewise must be in [0, 1], got {timewise}"
        assert 0.0 <= qklayerwise <= 1.0, f"qklayerwise must be in [0, 1], got {qklayerwise}"
        assert 0.0 <= avlayerwise <= 1.0, f"avlayerwise must be in [0, 1], got {avlayerwise}"
        assert 0.0 <= projlayerwise <= 1.0, f"projlayerwise must be in [0, 1], got {projlayerwise}"
        assert 0.0 <= mlplayerwise <= 1.0, f"mlplayerwise must be in [0, 1], got {mlplayerwise}"
        assert 0.0 <= inputprojlayerwise <= 1.0, f"inputprojlayerwise must be in [0, 1], got {inputprojlayerwise}"
        assert total_timesteps > 0, f"total_timesteps must be positive, got {total_timesteps}"
        assert total_blocks > 0, f"total_blocks must be positive, got {total_blocks}"
        assert sc_prec > 0, f"sc_prec must be positive, got {sc_prec}"

        # Legacy attributes (kept for backward compat)
        self.timewise = timewise
        self.qklayerwise = qklayerwise
        self.avlayerwise = avlayerwise
        self.projlayerwise = projlayerwise
        self.mlplayerwise = mlplayerwise
        self.inputprojlayerwise = inputprojlayerwise
        self.total_timesteps = total_timesteps
        self.total_blocks = total_blocks
        self.reverse_layerwise = reverse_layerwise
        self.sc_skip_blocks = sc_skip_blocks or set()

        # Derived from sc_prec
        self.sc_prec = sc_prec
        self.stoc_len = 2 ** sc_prec                # 256 for sc_prec=8
        self.quant_max = 2 ** (sc_prec - 1) - 1     # 127 for sc_prec=8
        self.sc_enable = sc_enable                   # Use enable-signal SC multiplication
        self.noise_model = noise_model               # Closed-form SC noise surrogate (fast sim)
        self.noise_local_correction = noise_local_correction
        self.noise_global_correction = noise_global_correction
        if noise_model:
            from .noise_matmul import set_noise_corrections
            set_noise_corrections(noise_local_correction, noise_global_correction)

        # Current timestep (set by diffusion loop)
        self.current_timestep: Optional[int] = None

        # Debug mode: run both FP and SC, log comparison, use FP downstream
        self.debug = False
        self._debug_log = []

        # Precision map: the source of truth for per-block, per-op, per-group config
        if precision_map is not None:
            self.precision_map = precision_map
        else:
            # Build from legacy args for backward compat
            # Create a minimal args-like object
            class _LegacyArgs:
                pass
            la = _LegacyArgs()
            la.sc_prec = sc_prec
            la.timewise = timewise
            la.qklayerwise = qklayerwise
            la.avlayerwise = avlayerwise
            la.projlayerwise = projlayerwise
            la.mlplayerwise = mlplayerwise
            la.inputprojlayerwise = inputprojlayerwise
            la.reverse_layerwise = reverse_layerwise
            la.sc_skip_blocks = ','.join(str(b) for b in (sc_skip_blocks or set()))
            self.precision_map = SCPrecisionMap.from_legacy_args(la, total_blocks)

        # Pre-compute legacy thresholds (for backward compat with old code paths)
        self._timestep_threshold = int(self.timewise * self.total_timesteps)
        self._qk_block_threshold = int(self.qklayerwise * self.total_blocks)
        self._av_block_threshold = int(self.avlayerwise * self.total_blocks)
        self._proj_block_threshold = int(self.projlayerwise * self.total_blocks)
        self._mlp_block_threshold = int(self.mlplayerwise * self.total_blocks)
        self._input_proj_block_threshold = int(self.inputprojlayerwise * self.total_blocks)

        # Mixed precision (per-token-row) config
        self.mp_config: Optional[MPConfig] = None

        # Adaptive mixed precision config (timestep-aware, inspired by APT)
        self.adaptive_mp_config: Optional[AdaptiveMPConfig] = None

        # Range-based mixed precision config (weight min/max range)
        self.range_mp_config: Optional[RangeMPConfig] = None

    # =================================================================
    # Per-operator precision API (new)
    # =================================================================

    def get_stoc_len(self, block_idx: int, operator: str) -> int:
        """Get the stoc_len for a specific block and operator."""
        return self.precision_map.get(block_idx, operator).stoc_len

    def get_group_stoc_lens(self, block_idx: int, operator: str) -> Optional[list[int]]:
        """Get per-group stoc_lens for a specific block and operator.

        Returns None if uniform precision (all groups use the same stoc_len).
        """
        return self.precision_map.get(block_idx, operator).group_stoc_lens

    @staticmethod
    def get_sc_prec_for_stoc_len(stoc_len: int) -> int:
        """Get sc_prec = ceil(log2(stoc_len)) for a given stoc_len."""
        if stoc_len <= 1:
            return 1
        return math.ceil(math.log2(stoc_len))

    # =================================================================
    # Timestep control
    # =================================================================

    def set_timestep(self, t: int):
        """Set the current diffusion timestep."""
        self.current_timestep = t

    def _in_sc_timestep(self) -> bool:
        """Check if current timestep should use SC (global timewise)."""
        if self.current_timestep is None:
            return False
        if self._timestep_threshold == 0:
            return False
        cutoff = self.total_timesteps - self._timestep_threshold
        return self.current_timestep >= cutoff

    def _in_sc_timestep_for(self, block_idx: int, operator: str) -> bool:
        """Check if current timestep should use SC for a specific operator.

        Uses per-operator timewise from precision map if available,
        falls back to global timewise.
        """
        if self.current_timestep is None:
            return False
        cfg = self.precision_map.get(block_idx, operator)
        tw = cfg.timewise
        if tw <= 0.0:
            return False
        threshold = int(tw * self.total_timesteps)
        if threshold == 0:
            return False
        cutoff = self.total_timesteps - threshold
        return self.current_timestep >= cutoff

    # =================================================================
    # Per-operator SC decision (uses precision map)
    # =================================================================

    def _use_sc_for(self, block_idx: int, operator: str) -> bool:
        """Generic check: should SC be used for this operator in this block?"""
        cfg = self.precision_map.get(block_idx, operator)
        if not cfg.enabled:
            return False
        return self._in_sc_timestep_for(block_idx, operator)

    def use_sc_for_qk(self, block_idx: int) -> bool:
        """Check if q@k^T should use SC for the given block."""
        return self._use_sc_for(block_idx, "qk")

    def use_sc_for_av(self, block_idx: int) -> bool:
        """Check if attn@v should use SC for the given block."""
        return self._use_sc_for(block_idx, "av")

    def use_sc_for_proj(self, block_idx: int) -> bool:
        """Check if output projection should use SC for the given block."""
        return self._use_sc_for(block_idx, "proj")

    def use_sc_for_mlp(self, block_idx: int) -> bool:
        """Check if MLP (both fc1 and fc2) should use SC for the given block.

        Legacy API: returns True if either fc1 or fc2 is enabled.
        """
        return (self._use_sc_for(block_idx, "mlp_fc1")
                or self._use_sc_for(block_idx, "mlp_fc2"))

    def use_sc_for_mlp_fc1(self, block_idx: int) -> bool:
        """Check if MLP fc1 should use SC for the given block."""
        return self._use_sc_for(block_idx, "mlp_fc1")

    def use_sc_for_mlp_fc2(self, block_idx: int) -> bool:
        """Check if MLP fc2 should use SC for the given block."""
        return self._use_sc_for(block_idx, "mlp_fc2")

    def use_sc_for_input_proj(self, block_idx: int) -> bool:
        """Check if QKV input projection should use SC for the given block."""
        return self._use_sc_for(block_idx, "input_proj")

    # =================================================================
    # Mixed precision (per-token-row) API
    # =================================================================

    def init_mp(self, mp_config: MPConfig):
        """Initialize mixed precision configuration."""
        self.mp_config = mp_config

    def init_adaptive_mp(self, adaptive_mp_config: AdaptiveMPConfig):
        """Initialize adaptive mixed precision (timestep-aware).

        When set, this takes priority over mp_config for dynamic MP paths.
        """
        self.adaptive_mp_config = adaptive_mp_config

    def init_range_mp(self, range_mp_config: RangeMPConfig):
        """Initialize range-based mixed precision (weight min/max range).

        When set, per-group stoc_lens are computed from weight ranges
        after quantization. Can be combined with adaptive/fixed MP.
        """
        self.range_mp_config = range_mp_config

    # =================================================================
    # Legacy helpers (kept for backward compat)
    # =================================================================

    def _block_in_range(self, block_idx: int, threshold: int) -> bool:
        """Check if block_idx falls in the SC range for a given threshold."""
        if threshold == 0:
            return False
        if block_idx in self.sc_skip_blocks:
            return False
        if self.reverse_layerwise:
            return block_idx >= self.total_blocks - threshold
        return block_idx < threshold

    # =================================================================
    # Debug
    # =================================================================

    def enable_debug(self):
        """Enable debug mode: compute both FP and SC, log error, use FP downstream."""
        self.debug = True
        self._debug_log = []

    def log_debug(self, block_idx: int, operator: str, fp_result, sc_result):
        """Log comparison between FP and SC results for one operator call."""
        import torch
        with torch.no_grad():
            diff = (sc_result.float() - fp_result.float())
            mse = diff.pow(2).mean().item()
            max_ae = diff.abs().max().item()
            fp_norm = fp_result.float().norm().item()
            cos_sim = torch.nn.functional.cosine_similarity(
                fp_result.float().reshape(1, -1),
                sc_result.float().reshape(1, -1),
            ).item()

        entry = {
            "timestep": self.current_timestep,
            "block": block_idx,
            "operator": operator,
            "mse": mse,
            "max_ae": max_ae,
            "fp_norm": fp_norm,
            "cos_sim": cos_sim,
        }
        self._debug_log.append(entry)
        print(f"  [DEBUG] t={entry['timestep']:3d} blk={block_idx:2d} "
              f"{operator:12s}  MSE={mse:.6e}  MaxAE={max_ae:.6e}  "
              f"CosSim={cos_sim:.6f}  FP_norm={fp_norm:.4f}")

    def save_debug_log(self, path: str):
        """Save debug log to CSV."""
        import csv
        if not self._debug_log:
            return
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._debug_log[0].keys())
            writer.writeheader()
            writer.writerows(self._debug_log)
        print(f"Debug log saved to {path} ({len(self._debug_log)} entries)")

    def get_sc_config(self) -> dict:
        """Get SC configuration parameters (legacy API)."""
        return {
            'sc_prec': self.sc_prec,
            'stoc_len': self.stoc_len,
            'quant_max': self.quant_max,
            'timewise': self.timewise,
            'qklayerwise': self.qklayerwise,
            'avlayerwise': self.avlayerwise,
            'projlayerwise': self.projlayerwise,
            'mlplayerwise': self.mlplayerwise,
            'inputprojlayerwise': self.inputprojlayerwise,
        }

    def __repr__(self) -> str:
        return (
            f"SCController("
            f"timewise={self.timewise}, "
            f"qklayerwise={self.qklayerwise}, "
            f"avlayerwise={self.avlayerwise}, "
            f"projlayerwise={self.projlayerwise}, "
            f"mlplayerwise={self.mlplayerwise}, "
            f"inputprojlayerwise={self.inputprojlayerwise}, "
            f"reverse_layerwise={self.reverse_layerwise}, "
            f"total_timesteps={self.total_timesteps}, "
            f"total_blocks={self.total_blocks}, "
            f"sc_prec={self.sc_prec}, "
            f"sc_enable={self.sc_enable}, "
            f"quant_max={self.quant_max})"
        )
