"""
SC-enabled MLP module for Q-DiT.

This module provides SCMlp, which extends QuantMlp with stochastic
computing support for fc1 and fc2 linear layers.

Supports:
- Independent fc1/fc2 SC control (one can be SC, the other FP)
- Per-group mixed precision via dispatch tables
- Early termination with per-group stoc_len
"""
from __future__ import annotations

import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from copy import deepcopy

from ..quant import Quantizer
from ..qLinearLayer import QLinearLayer
from .sc_controller import SCController
from .mp_config import classify_rows_by_metric, adaptive_classify_rows, MPDistributionLogger, MetricProfiler

# Add SC folder to path for imports
SC_PATH = Path(__file__).parent.parent.parent.parent / "SC"
if str(SC_PATH) not in sys.path:
    sys.path.insert(0, str(SC_PATH))
from scmp_kernels.sc.config_helpers import make_sobol_simple_config
from scmp_kernels.sc import sc_matmul  # unified granularity dispatcher


def _sc_mlp_enable_per_row(a, b, max_fp_a=0.0, min_fp_a=0.0,
                           max_fp_b=None, min_fp_b=None,
                           mode="bipolar", sc_prec=8, config=None,
                           group_a=1, group_b=1, chunk_d=0, stoc_len=None):
    """Enable-signal MLP matmul, migrated to scmp_kernels.sc.sc_matmul.

    Mirrors the legacy ``sc_matmul_enable_triton_mlp`` signature so the
    polymorphic call sites (which sometimes pass per-tensor max/min
    positionally) are unchanged. The per-row path computes its own per-row
    scales, so ``max_fp_*`` are accepted-and-ignored (verified bit-exact vs
    the old kernel on GPU)."""
    return sc_matmul(a, b, granularity="per_row",
                     mode=mode, sc_prec=sc_prec, config=config,
                     group_a=group_a, group_b=group_b, chunk_d=chunk_d,
                     stoc_len=stoc_len)


@dataclass
class DispatchEntry:
    """Pre-computed dispatch entry for one stoc_len level."""
    stoc_len: int
    sc_prec: int
    weight: torch.Tensor
    bias: Optional[torch.Tensor]
    out_indices: torch.Tensor


class SCMlp(nn.Module):
    """
    MLP module with stochastic computing support for fc1 and fc2.

    Supports independent sc1/fc2 control and per-group mixed precision
    via dispatch tables built once after weight quantization.

    Args:
        mlp: Original MLP module (timm Mlp)
        args: Quantization arguments
        block_idx: Index of the block this MLP belongs to
        sc_controller: SCController instance for SC decisions
    """

    def __init__(self, mlp, args, block_idx, sc_controller):
        super().__init__()
        self.block_idx = block_idx
        self.sc_controller = sc_controller

        # SC mode: match Q-DiT's quantization symmetry
        self.sc_mode = "bipolar" if args.w_sym else "unipolar"

        self.input_quant = Quantizer(args=deepcopy(args))
        self.fc1 = QLinearLayer(mlp.fc1, deepcopy(args))
        self.act = mlp.act
        self.drop1 = mlp.drop1
        self.norm = mlp.norm
        self.act_quant = Quantizer(args=deepcopy(args))
        self.fc2 = QLinearLayer(mlp.fc2, deepcopy(args))
        self.drop2 = mlp.drop2
        self.register_buffer("reorder_index_fc1", None)

        # SC config cache keyed by (D, sc_prec), created on first use
        self._sc_configs = {}

        # Dispatch tables for mixed precision (built after weight quantization)
        self._dispatch_tables: dict[str, list[DispatchEntry]] = {}

    def _get_sc_config(self, D, sc_prec=None):
        """Get or create cached SC config for the given (D, sc_prec)."""
        if sc_prec is None:
            sc_prec = self.sc_controller.sc_prec
        key = (D, sc_prec)
        if key not in self._sc_configs:
            self._sc_configs[key] = make_sobol_simple_config(D, D, sc_prec)
        return self._sc_configs[key]

    def _get_sc_matmul_fn(self):
        """Return the SC matmul function (per-row) or noise surrogate.

        The legacy packed path (sc_enable=False) is gone — scmp has no packed
        kernel — so both flags route to the scmp per_row path.
        """
        if self.sc_controller.noise_model:
            from .noise_matmul import noisy_sc_matmul_mlp
            return noisy_sc_matmul_mlp
        return _sc_mlp_enable_per_row

    # =================================================================
    # Dispatch table construction (called once after weight quantization)
    # =================================================================

    def _build_dispatch_tables(self):
        """Build dispatch tables for mixed-precision forward.

        Called once after weight quantization from sc_modelutils.py.
        """
        for operator, weight, bias in [
            ("mlp_fc1", self.fc1.weight, self.fc1.bias),
            ("mlp_fc2", self.fc2.weight, self.fc2.bias),
        ]:
            group_stoc_lens = self.sc_controller.get_group_stoc_lens(
                self.block_idx, operator)
            if group_stoc_lens is None:
                continue

            stoc_len_to_rows: dict[int, list[int]] = defaultdict(list)
            group_size = weight.shape[0] // len(group_stoc_lens)
            for g, sl in enumerate(group_stoc_lens):
                start = g * group_size
                end = min((g + 1) * group_size, weight.shape[0])
                stoc_len_to_rows[sl].extend(range(start, end))

            entries = []
            for sl in sorted(stoc_len_to_rows, reverse=True):
                rows = stoc_len_to_rows[sl]
                idx = torch.tensor(rows, dtype=torch.long, device=weight.device)
                sp = SCController.get_sc_prec_for_stoc_len(sl)
                entries.append(DispatchEntry(
                    stoc_len=sl,
                    sc_prec=sp,
                    weight=weight[idx].contiguous(),
                    bias=None,
                    out_indices=idx,
                ))
                # Pre-warm config cache
                self._get_sc_config(weight.shape[1], sp)

            self._dispatch_tables[operator] = entries

    # =================================================================
    # SC linear with dispatch support
    # =================================================================

    def _sc_linear_uniform(self, x, weight, bias, sc_prec, stoc_len, chunk_d=0):
        """Uniform precision SC linear — single kernel launch."""
        orig_shape = x.shape
        D = x.shape[-1]
        x_flat = x.reshape(-1, D)

        # Noise-model fast path: skip chunk_d, call core directly.
        if self.sc_controller.noise_model:
            from .noise_matmul import _noisy_matmul_core
            result = _noisy_matmul_core(
                x_flat, weight, L=stoc_len, mode=self.sc_mode,
                per_row_scale=True,
            )
            if bias is not None:
                result = result + bias
            return result.reshape(*orig_shape[:-1], -1)

        matmul_fn = self._get_sc_matmul_fn()

        if chunk_d > 0 and D > chunk_d:
            config = self._get_sc_config(chunk_d, sc_prec)
            result = matmul_fn(
                x_flat, weight,
                mode=self.sc_mode, sc_prec=sc_prec, config=config,
                group_a=1, group_b=1, chunk_d=chunk_d,
                stoc_len=stoc_len)
        else:
            config = self._get_sc_config(D, sc_prec)
            result = matmul_fn(x_flat, weight,
                               x_flat.max().item(), x_flat.min().item(),
                               weight.max().item(), weight.min().item(),
                               mode=self.sc_mode, sc_prec=sc_prec, config=config,
                               group_a=1, group_b=1,
                               stoc_len=stoc_len)

        if bias is not None:
            result = result + bias

        return result.reshape(*orig_shape[:-1], -1)

    def _sc_linear_dynamic_mp(self, x, weight, bias, operator, chunk_d=0):
        """Dynamic per-token-row mixed precision SC linear.

        Classifies input rows by magnitude, assigns each row a stoc_len level,
        and computes SC matmul per level. Supports both fixed-fraction MP and
        timestep-adaptive MP (when adaptive_mp_config is set).
        """
        orig_shape = x.shape
        D = x.shape[-1]
        x_flat = x.reshape(-1, D)
        M = x_flat.shape[0]

        # Per-row metric: max absolute value
        row_metric = x_flat.float().abs().amax(dim=-1)  # [M]
        MetricProfiler.record(row_metric, self.sc_controller.current_timestep,
                              self.block_idx, operator)

        if self.sc_controller.adaptive_mp_config is not None:
            assignment = adaptive_classify_rows(
                row_metric,
                self.sc_controller.current_timestep,
                self.sc_controller.total_timesteps,
                self.sc_controller.adaptive_mp_config,
                operator=operator,
)
        else:
            mp_config = self.sc_controller.mp_config
            assignment = classify_rows_by_metric(
                row_metric, mp_config.stoc_len_levels, mp_config.level_fractions)

        MPDistributionLogger.log(
            self.sc_controller.current_timestep, self.block_idx,
            operator, assignment, M)

        matmul_fn = self._get_sc_matmul_fn()
        out_features = weight.shape[0]
        result = torch.zeros(M, out_features,
                             device=x.device, dtype=torch.float32)

        dynamic_levels = (self.sc_controller.adaptive_mp_config.stoc_len_levels
                          if self.sc_controller.adaptive_mp_config is not None
                          else self.sc_controller.mp_config.stoc_len_levels)
        max_stoc_len = max(dynamic_levels)
        compute_baseline = 0
        compute_actual = 0.0

        for sl, rows in assignment.level_row_indices.items():
            if len(rows) == 0 or sl == 0:
                continue  # pruned rows: result already zeroed
            n_rows = len(rows)
            compute_baseline += n_rows * out_features * max_stoc_len
            compute_actual += n_rows * out_features * sl

            sp = SCController.get_sc_prec_for_stoc_len(sl)
            x_sub = x_flat[rows].contiguous()  # [len(rows), D]

            if chunk_d > 0 and D > chunk_d:
                config = self._get_sc_config(chunk_d, sp)
                sub = matmul_fn(
                    x_sub, weight,
                    mode=self.sc_mode, sc_prec=sp, config=config,
                    group_a=1, group_b=1, chunk_d=chunk_d,
                    stoc_len=sl)
            else:
                config = self._get_sc_config(D, sp)
                sub = matmul_fn(
                    x_sub, weight,
                    x_sub.max().item(), x_sub.min().item(),
                    weight.max().item(), weight.min().item(),
                    mode=self.sc_mode, sc_prec=sp, config=config,
                    group_a=1, group_b=1,
                    stoc_len=sl)
            result[rows] = sub

        MPDistributionLogger.log_compute(
            self.sc_controller.current_timestep, self.block_idx,
            operator, compute_baseline, compute_actual)

        if bias is not None:
            result = result + bias

        return result.reshape(*orig_shape[:-1], -1)

    def _sc_linear_combined_mp(self, x, weight, bias, operator, dispatch,
                               chunk_d=0):
        """Combined range-based + dynamic mixed precision SC linear.

        Iterates dispatch entries (range-based weight groups), and within each
        entry, further classifies input rows by dynamic MP metric.  The
        effective stoc_len for each (weight_group, input_row) pair is
        min(range_stoc_len, dynamic_stoc_len).
        """
        orig_shape = x.shape
        D = x.shape[-1]
        x_flat = x.reshape(-1, D)
        M = x_flat.shape[0]

        # Classify input rows by dynamic MP
        row_metric = x_flat.float().abs().amax(dim=-1)  # [M]
        MetricProfiler.record(row_metric, self.sc_controller.current_timestep,
                              self.block_idx, operator)

        if self.sc_controller.adaptive_mp_config is not None:
            assignment = adaptive_classify_rows(
                row_metric,
                self.sc_controller.current_timestep,
                self.sc_controller.total_timesteps,
                self.sc_controller.adaptive_mp_config,
                operator=operator,
            )
        else:
            mp_config = self.sc_controller.mp_config
            assignment = classify_rows_by_metric(
                row_metric, mp_config.stoc_len_levels, mp_config.level_fractions)

        MPDistributionLogger.log(
            self.sc_controller.current_timestep, self.block_idx,
            operator, assignment, M)

        # Per-row dynamic stoc_len
        dynamic_levels = self.sc_controller.adaptive_mp_config.stoc_len_levels \
            if self.sc_controller.adaptive_mp_config is not None \
            else self.sc_controller.mp_config.stoc_len_levels
        max_stoc_len = max(dynamic_levels)
        # Baseline: all M rows * all out_features at max_stoc_len
        # (includes pruned rows/groups for accurate savings)
        compute_baseline = M * weight.shape[0] * max_stoc_len
        compute_actual = 0.0

        row_stoc_lens = torch.zeros(M, dtype=torch.long, device=x.device)
        for sl, rows in assignment.level_row_indices.items():
            if len(rows) > 0:
                row_stoc_lens[rows] = sl

        matmul_fn = self._get_sc_matmul_fn()
        result = torch.zeros(M, weight.shape[0],
                             device=x.device, dtype=torch.float32)

        for entry in dispatch:
            weight_sl = entry.stoc_len
            if weight_sl == 0:
                continue
            n_out = len(entry.out_indices)

            effective_sl = torch.minimum(
                row_stoc_lens,
                torch.tensor(weight_sl, device=x.device))

            unique_sls = effective_sl.unique()
            for eff_sl in unique_sls:
                eff_sl_val = eff_sl.item()
                if eff_sl_val == 0:
                    continue
                rows = torch.where(effective_sl == eff_sl)[0]
                if len(rows) == 0:
                    continue

                n_rows = len(rows)
                compute_actual += n_rows * n_out * eff_sl_val

                sp = SCController.get_sc_prec_for_stoc_len(eff_sl_val)
                x_sub = x_flat[rows].contiguous()

                if chunk_d > 0 and D > chunk_d:
                    config = self._get_sc_config(chunk_d, sp)
                    sub = matmul_fn(
                        x_sub, entry.weight,
                        mode=self.sc_mode, sc_prec=sp, config=config,
                        group_a=1, group_b=1, chunk_d=chunk_d,
                        stoc_len=eff_sl_val)
                else:
                    config = self._get_sc_config(D, sp)
                    sub = matmul_fn(
                        x_sub, entry.weight,
                        x_sub.max().item(), x_sub.min().item(),
                        entry.weight.max().item(), entry.weight.min().item(),
                        mode=self.sc_mode, sc_prec=sp, config=config,
                        group_a=1, group_b=1,
                        stoc_len=eff_sl_val)
                result[rows.unsqueeze(1), entry.out_indices.unsqueeze(0)] = sub

        MPDistributionLogger.log_compute(
            self.sc_controller.current_timestep, self.block_idx,
            operator, compute_baseline, compute_actual)

        if bias is not None:
            result = result + bias

        return result.reshape(*orig_shape[:-1], -1)

    def _sc_linear(self, x, weight, bias, operator=None, chunk_d=0):
        """y = x @ W^T + bias, using SC matmul.

        If a dispatch table exists for this operator, iterates precomputed
        dispatch entries (static mixed precision). If mp_config is set and
        no static dispatch, uses dynamic per-token-row mixed precision.
        When both dispatch table (range-based) and dynamic MP are active,
        uses combined mode. Otherwise, single kernel launch (uniform).
        """
        dispatch = self._dispatch_tables.get(operator) if operator else None
        has_dynamic_mp = (self.sc_controller.adaptive_mp_config is not None
                          or self.sc_controller.mp_config is not None)

        if dispatch is not None and has_dynamic_mp:
            return self._sc_linear_combined_mp(x, weight, bias, operator,
                                                dispatch, chunk_d)

        if dispatch is None and has_dynamic_mp:
            return self._sc_linear_dynamic_mp(x, weight, bias, operator, chunk_d)
        if dispatch is None:
            stoc_len = self.sc_controller.get_stoc_len(self.block_idx, operator) if operator else self.sc_controller.stoc_len
            sc_prec = SCController.get_sc_prec_for_stoc_len(stoc_len)
            return self._sc_linear_uniform(x, weight, bias, sc_prec, stoc_len, chunk_d)

        # Mixed precision — iterate pre-built dispatch entries (range-based only)
        orig_shape = x.shape
        D = x.shape[-1]
        x_flat = x.reshape(-1, D)
        M = x_flat.shape[0]

        matmul_fn = self._get_sc_matmul_fn()

        max_stoc_len = max(e.stoc_len for e in dispatch)
        compute_baseline = 0
        compute_actual = 0.0

        result = torch.empty(x_flat.shape[0], weight.shape[0],
                             device=x.device, dtype=torch.float32)

        for entry in dispatch:
            n_out = len(entry.out_indices)
            compute_baseline += M * n_out * max_stoc_len
            compute_actual += M * n_out * entry.stoc_len

            if chunk_d > 0 and D > chunk_d:
                config = self._get_sc_config(chunk_d, entry.sc_prec)
                sub = matmul_fn(
                    x_flat, entry.weight,
                    mode=self.sc_mode, sc_prec=entry.sc_prec, config=config,
                    group_a=1, group_b=1, chunk_d=chunk_d,
                    stoc_len=entry.stoc_len)
            else:
                config = self._get_sc_config(D, entry.sc_prec)
                sub = matmul_fn(x_flat, entry.weight,
                                x_flat.max().item(), x_flat.min().item(),
                                entry.weight.max().item(), entry.weight.min().item(),
                                mode=self.sc_mode, sc_prec=entry.sc_prec,
                                config=config, group_a=1, group_b=1,
                                stoc_len=entry.stoc_len)
            result[:, entry.out_indices] = sub

        MPDistributionLogger.log_compute(
            self.sc_controller.current_timestep, self.block_idx,
            operator, compute_baseline, compute_actual)

        if bias is not None:
            result = result + bias

        return result.reshape(*orig_shape[:-1], -1)

    # =================================================================
    # Logging
    # =================================================================

    _compare_log = []  # class-level list to collect stats across blocks

    def _log_compare(self, name, x_fp, x_sc):
        """Collect SC vs FP comparison stats for CSV output."""
        diff = (x_sc - x_fp).abs()
        fp_mean = x_fp.mean().item()
        sc_mean = x_sc.mean().item()
        SCMlp._compare_log.append({
            "block": self.block_idx,
            "layer": name,
            "fp_min": x_fp.min().item(),
            "fp_max": x_fp.max().item(),
            "fp_mean": fp_mean,
            "sc_min": x_sc.min().item(),
            "sc_max": x_sc.max().item(),
            "sc_mean": sc_mean,
            "abs_err_mean": diff.mean().item(),
            "abs_err_max": diff.max().item(),
            "rel_err": abs(sc_mean - fp_mean) / max(abs(fp_mean), 1e-8),
        })

    @classmethod
    def dump_compare_csv(cls, path="debug_sc_mlp.csv"):
        """Write collected comparison stats to CSV and clear."""
        if not cls._compare_log:
            return
        import csv
        keys = cls._compare_log[0].keys()
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(cls._compare_log)
        print(f"[SCMlp] Wrote {len(cls._compare_log)} rows to {path}")
        cls._compare_log.clear()

    # =================================================================
    # Forward
    # =================================================================

    def forward(self, x):
        debug = self.sc_controller.debug

        if self.reorder_index_fc1 is not None:
            x = torch.index_select(x, 2, self.reorder_index_fc1)
        x = self.input_quant(x)

        # Force FP for the last 2 blocks
        force_fp = self.block_idx >= self.sc_controller.total_blocks - 2

        # FC1: use SC matmul with D-chunking (D=1152 -> chunks of 72)
        if not force_fp and self.sc_controller.use_sc_for_mlp_fc1(self.block_idx):
            x_sc = self._sc_linear(x, self.fc1.weight, self.fc1.bias,
                                   operator="mlp_fc1", chunk_d=72)
            if debug:
                x_fp = self.fc1(x)
                self._log_compare("fc1", x_fp, x_sc)
                self.sc_controller.log_debug(self.block_idx, "mlp_fc1", x_fp, x_sc)
            x = x_sc
        else:
            x = self.fc1(x)

        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.act_quant(x)

        # FC2: use SC matmul with D-chunking (D=4608 -> chunks of 72)
        if not force_fp and self.sc_controller.use_sc_for_mlp_fc2(self.block_idx):
            x_sc = self._sc_linear(x, self.fc2.weight, self.fc2.bias,
                                   operator="mlp_fc2", chunk_d=72)
            if debug:
                x_fp = self.fc2(x)
                self._log_compare("fc2", x_fp, x_sc)
                self.sc_controller.log_debug(self.block_idx, "mlp_fc2", x_fp, x_sc)
            x = x_sc
        else:
            x = self.fc2(x)

        x = self.drop2(x)
        return x

    def to(self, *args, **kwargs):
        super(SCMlp, self).to(*args, **kwargs)
        self.fc1 = self.fc1.to(*args, **kwargs)
        self.act = self.act.to(*args, **kwargs)
        self.drop1 = self.drop1.to(*args, **kwargs)
        self.norm = self.norm.to(*args, **kwargs)
        self.fc2 = self.fc2.to(*args, **kwargs)
        self.drop2 = self.drop2.to(*args, **kwargs)
        self.act_quant = self.act_quant.to(*args, **kwargs)
        self.input_quant = self.input_quant.to(*args, **kwargs)
        if self.reorder_index_fc1 is not None:
            self.reorder_index_fc1 = self.reorder_index_fc1.to(*args, **kwargs)
        return self
