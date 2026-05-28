"""
SC-enabled Attention module for Q-DiT.

This module provides SCAttention, which extends QuantAttention with
stochastic computing support for input projection, q@k^T, attn@v,
and output projection operations.

Supports per-group mixed precision via dispatch tables: each weight
quantization group (linear) or attention head (BMM) can use a different
stoc_len for early-termination speedup.
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
# All SC matmuls route through the unified scmp dispatcher.
from scmp_kernels.sc import sc_matmul as scmp_sc_matmul


# --- SC matmul adapters: mirror the legacy sc_triton signatures (so the
# polymorphic getters / call sites are unchanged) but route to
# scmp_kernels.sc.sc_matmul. The per_row / per_head ones are verified
# bit-exact vs the old kernels on GPU (sc_prec=8); the per_tensor one backs a
# dead path (sc_enable=False / unipolar QK, never used here). ---------------
def _sc_mlp_enable_per_row(a, b, max_fp_a=0.0, min_fp_a=0.0,
                           max_fp_b=None, min_fp_b=None,
                           mode="bipolar", sc_prec=8, config=None,
                           group_a=1, group_b=1, chunk_d=0, stoc_len=None):
    """Legacy ``sc_matmul_enable_triton_mlp`` -> per_row. ``max_fp_*`` ignored
    (per-row computes its own scales)."""
    return scmp_sc_matmul(a, b, granularity="per_row",
                          mode=mode, sc_prec=sc_prec, config=config,
                          group_a=group_a, group_b=group_b, chunk_d=chunk_d,
                          stoc_len=stoc_len)


def _sc_grouped_enable_per_row(a, b, group_a=1, group_b=1, mode="unipolar",
                               sc_prec=8, config=None, stoc_len=None):
    """Legacy ``sc_matmul_grouped_enable_triton`` -> per_row."""
    return scmp_sc_matmul(a, b, granularity="per_row",
                          group_a=group_a, group_b=group_b, mode=mode,
                          sc_prec=sc_prec, config=config, stoc_len=stoc_len)


def _sc_qk_perhead_bipolar(q_flat, k_flat, q_maxs, q_mins, k_maxs, k_mins,
                           sc_prec, config, stoc_len=None):
    """Legacy ``sc_matmul_enable_batched_bipolar`` -> per_head. The per-head
    max/min are recomputed internally by the dispatcher (== the passed-in
    amax/amin), so the explicit range args are accepted-and-ignored."""
    return scmp_sc_matmul(q_flat, k_flat, granularity="per_head",
                          sc_prec=sc_prec, config=config, stoc_len=stoc_len)


def _sc_pertensor(a, b, max_fp_a=0.0, min_fp_a=0.0, max_fp_b=None,
                  min_fp_b=None, mode="bipolar", sc_prec=8,
                  config=None, stoc_len=None):
    """Legacy per-tensor ``sc_matmul_enable_triton`` / packed ``sc_matmul``
    -> scmp per_tensor. Dead path: only reached for sc_enable=False / unipolar
    QK, which this project never uses. Not bit-identical to the old per-tensor
    kernel, but never executed in the default bipolar + sc_enable path."""
    return scmp_sc_matmul(a, b, granularity="per_tensor", mode=mode,
                          sc_prec=sc_prec, config=config, stoc_len=stoc_len)


@dataclass
class DispatchEntry:
    """Pre-computed dispatch entry for one stoc_len level."""
    stoc_len: int
    sc_prec: int
    weight: torch.Tensor        # contiguous sub-weight for this level
    bias: Optional[torch.Tensor]  # None; bias is added after merge
    out_indices: torch.Tensor   # LongTensor of output row indices


class SCAttention(nn.Module):
    """
    Attention module with stochastic computing support for all operators:
    input projection (QKV), q@k^T, attn@v, and output projection.

    Supports per-group mixed precision via dispatch tables that are built
    once after weight quantization and used in every forward pass.

    Args:
        attn: Original attention module (timm Attention)
        args: Quantization arguments
        block_idx: Index of the block this attention belongs to
        sc_controller: SCController instance for SC decisions
    """

    def __init__(
        self,
        attn,  # timm Attention module
        args,
        block_idx: int,
        sc_controller: SCController,
    ):
        super().__init__()

        self.args = args
        self.block_idx = block_idx
        self.sc_controller = sc_controller

        # Copy attention parameters
        self.quantize_bmm_input = args.quantize_bmm_input
        self.abits = args.abits
        self.num_heads = attn.num_heads
        self.head_dim = attn.head_dim
        self.scale = attn.scale
        self.fused_attn = False  # Disable fused attention to use our SC path
        self.q_norm = attn.q_norm
        self.k_norm = attn.k_norm
        self.attn_drop = attn.attn_drop
        self.proj_drop = attn.proj_drop

        # SC mode: match Q-DiT's quantization symmetry
        self.sc_mode = "bipolar" if args.w_sym else "unipolar"

        # Quantizers
        self.input_quant = Quantizer(args=deepcopy(args))
        self.qkv = QLinearLayer(attn.qkv, deepcopy(args))
        if self.quantize_bmm_input:
            self.q_quant = Quantizer(args=deepcopy(args))
            self.k_quant = Quantizer(args=deepcopy(args))
            self.v_quant = Quantizer(args=deepcopy(args))
        self.act_quant = Quantizer(args=deepcopy(args))
        self.proj = QLinearLayer(attn.proj, deepcopy(args))

        # Reorder indices (for activation reordering optimization)
        self.register_buffer("reorder_index_qkv", None)
        self.register_buffer("reorder_index_proj", None)

        # AV per-row-group quantization parameters
        self.av_attn_group_size = getattr(args, 'av_attn_group_size', 1)
        self.av_v_group_size = getattr(args, 'av_v_group_size', 1)

        # SC config cache keyed by (D, sc_prec), created on first use
        self._sc_configs = {}

        # Dispatch tables for mixed precision (built after weight quantization)
        # Keys: operator name ("input_proj", "proj")
        # Values: list[DispatchEntry] or None (uniform)
        self._dispatch_tables: dict[str, list[DispatchEntry]] = {}

    _compare_log = []  # class-level list to collect stats across blocks

    def _log_compare(self, name, x_fp, x_sc):
        """Collect SC vs FP comparison stats for CSV output."""
        diff = (x_sc - x_fp).abs()
        fp_mean = x_fp.mean().item()
        sc_mean = x_sc.mean().item()
        SCAttention._compare_log.append({
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
    def dump_compare_csv(cls, path="debug_sc_proj.csv"):
        """Write collected comparison stats to CSV and clear."""
        if not cls._compare_log:
            return
        import csv
        keys = cls._compare_log[0].keys()
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(cls._compare_log)
        print(f"[SCAttention] Wrote {len(cls._compare_log)} rows to {path}")
        cls._compare_log.clear()

    def _get_sc_config(self, D, sc_prec=None):
        """Get or create cached SC config for the given (D, sc_prec)."""
        if sc_prec is None:
            sc_prec = self.sc_controller.sc_prec
        key = (D, sc_prec)
        if key not in self._sc_configs:
            self._sc_configs[key] = make_sobol_simple_config(D, D, sc_prec)
        return self._sc_configs[key]

    def _get_sc_matmul_fn(self):
        """Return the SC matmul function (per-tensor) or noise surrogate.

        The legacy packed path (sc_enable=False) is gone — scmp has no packed
        AND/XNOR kernel — so both flags route to the scmp per_tensor path.
        """
        if self.sc_controller.noise_model:
            from .noise_matmul import noisy_sc_matmul
            return noisy_sc_matmul
        return _sc_pertensor

    def _get_mlp_matmul_fn(self):
        """Return the per-row-grouped (MLP-style) SC matmul function."""
        if self.sc_controller.noise_model:
            from .noise_matmul import noisy_sc_matmul_mlp
            return noisy_sc_matmul_mlp
        return _sc_mlp_enable_per_row

    def _get_av_grouped_fn(self):
        """Return the AV 'grouped' SC matmul function."""
        if self.sc_controller.noise_model:
            from .noise_matmul import noisy_sc_matmul_grouped
            return noisy_sc_matmul_grouped
        return _sc_grouped_enable_per_row

    def _get_batched_bipolar_fn(self):
        """Return the QK-batched bipolar SC matmul function."""
        if self.sc_controller.noise_model:
            from .noise_matmul import noisy_sc_matmul_enable_batched_bipolar
            return noisy_sc_matmul_enable_batched_bipolar
        return _sc_qk_perhead_bipolar

    # =================================================================
    # Dispatch table construction (called once after weight quantization)
    # =================================================================

    def _build_dispatch_table_for(self, table_key, weight, group_stoc_lens):
        """Build a single dispatch table for the given weight and group stoc_lens.

        Args:
            table_key: Key for self._dispatch_tables (e.g. "proj", "input_proj_q").
            weight: Weight tensor [out_features, in_features].
            group_stoc_lens: Per-group stoc_len list.
        """
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
                bias=None,  # bias added after merge
                out_indices=idx,
            ))
            # Pre-warm config cache for this (D, sc_prec) pair
            self._get_sc_config(weight.shape[1], sp)

        self._dispatch_tables[table_key] = entries

    def _build_dispatch_tables(self):
        """Build dispatch tables for mixed-precision forward.

        Called once after weight quantization from sc_modelutils.py.
        For each linear operator with per-group stoc_lens, pre-slices the
        weight into contiguous sub-matrices grouped by stoc_len level.
        """
        # Output projection
        proj_stoc_lens = self.sc_controller.get_group_stoc_lens(
            self.block_idx, "proj")
        if proj_stoc_lens is not None:
            self._build_dispatch_table_for(
                "proj", self.proj.weight, proj_stoc_lens)

        # Input projection: group_stoc_lens is stored as concatenation of
        # Q, K, V portions (each with groups_per_sub entries).
        # Build 3 separate dispatch tables keyed "input_proj_q/k/v".
        input_proj_stoc_lens = self.sc_controller.get_group_stoc_lens(
            self.block_idx, "input_proj")
        if input_proj_stoc_lens is not None:
            w = self.qkv.weight  # [3*C, C]
            C = w.shape[0] // 3
            groups_per_sub = len(input_proj_stoc_lens) // 3
            for suffix, w_start, gs_start in [
                ("q", 0, 0),
                ("k", C, groups_per_sub),
                ("v", 2 * C, 2 * groups_per_sub),
            ]:
                sub_weight = w[w_start:w_start + C]
                sub_stoc_lens = input_proj_stoc_lens[gs_start:gs_start + groups_per_sub]
                self._build_dispatch_table_for(
                    f"input_proj_{suffix}", sub_weight, sub_stoc_lens)

    # =================================================================
    # SC linear with dispatch support
    # =================================================================

    def _sc_linear_uniform(self, x, weight, bias, sc_prec, stoc_len,
                           chunk_d=0, grouped=False):
        """Uniform precision SC linear — single kernel launch."""
        orig_shape = x.shape
        D = x.shape[-1]
        x_flat = x.reshape(-1, D)

        # Noise-model fast path: skip chunk_d (memory mgmt for real SC) and
        # call the core surrogate directly to avoid the adapter indirection.
        if self.sc_controller.noise_model:
            from .noise_matmul import _noisy_matmul_core
            result = _noisy_matmul_core(
                x_flat, weight, L=stoc_len, mode=self.sc_mode,
                per_row_scale=grouped,
            )
            if bias is not None:
                result = result + bias
            return result.reshape(*orig_shape[:-1], -1)

        if grouped:
            matmul_fn = self._get_mlp_matmul_fn()
        else:
            matmul_fn = self._get_sc_matmul_fn()

        if chunk_d > 0 and D > chunk_d:
            result = None
            for start in range(0, D, chunk_d):
                end = min(start + chunk_d, D)
                x_chunk = x_flat[:, start:end].contiguous()
                w_chunk = weight[:, start:end].contiguous()
                chunk_size = end - start
                config = self._get_sc_config(chunk_size, sc_prec)

                kwargs = dict(
                    mode=self.sc_mode, sc_prec=sc_prec, config=config,
                    stoc_len=stoc_len)
                if grouped:
                    kwargs.update(group_a=1, group_b=1)

                chunk_result = matmul_fn(
                    x_chunk, w_chunk,
                    x_chunk.max().item(), x_chunk.min().item(),
                    w_chunk.max().item(), w_chunk.min().item(),
                    **kwargs)

                if result is None:
                    result = chunk_result
                else:
                    result = result + chunk_result
        else:
            config = self._get_sc_config(D, sc_prec)
            kwargs = dict(
                mode=self.sc_mode, sc_prec=sc_prec, config=config,
                stoc_len=stoc_len)
            if grouped:
                kwargs.update(group_a=1, group_b=1)

            result = matmul_fn(x_flat, weight,
                               x_flat.max().item(), x_flat.min().item(),
                               weight.max().item(), weight.min().item(),
                               **kwargs)

        if bias is not None:
            result = result + bias

        return result.reshape(*orig_shape[:-1], -1)

    def _sc_linear_dynamic_mp(self, x, weight, bias, operator, chunk_d=0, grouped=False):
        """Dynamic per-token-row mixed precision SC linear for attention projections."""
        orig_shape = x.shape
        D = x.shape[-1]
        x_flat = x.reshape(-1, D)
        M = x_flat.shape[0]

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

        if grouped:
            matmul_fn = self._get_mlp_matmul_fn()
        else:
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
                continue
            n_rows = len(rows)
            compute_baseline += n_rows * out_features * max_stoc_len
            compute_actual += n_rows * out_features * sl

            sp = SCController.get_sc_prec_for_stoc_len(sl)
            x_sub = x_flat[rows].contiguous()

            if chunk_d > 0 and D > chunk_d:
                sub_result = None
                for start in range(0, D, chunk_d):
                    end = min(start + chunk_d, D)
                    x_chunk = x_sub[:, start:end].contiguous()
                    w_chunk = weight[:, start:end].contiguous()
                    config = self._get_sc_config(end - start, sp)

                    kwargs = dict(mode=self.sc_mode, sc_prec=sp, config=config,
                                  stoc_len=sl)
                    if grouped:
                        kwargs.update(group_a=1, group_b=1)

                    chunk_out = matmul_fn(
                        x_chunk, w_chunk,
                        x_chunk.max().item(), x_chunk.min().item(),
                        w_chunk.max().item(), w_chunk.min().item(),
                        **kwargs)

                    if sub_result is None:
                        sub_result = chunk_out
                    else:
                        sub_result = sub_result + chunk_out
                result[rows] = sub_result
            else:
                config = self._get_sc_config(D, sp)
                kwargs = dict(mode=self.sc_mode, sc_prec=sp, config=config,
                              stoc_len=sl)
                if grouped:
                    kwargs.update(group_a=1, group_b=1)

                sub = matmul_fn(x_sub, weight,
                                x_sub.max().item(), x_sub.min().item(),
                                weight.max().item(), weight.min().item(),
                                **kwargs)
                result[rows] = sub

        MPDistributionLogger.log_compute(
            self.sc_controller.current_timestep, self.block_idx,
            operator, compute_baseline, compute_actual)

        if bias is not None:
            result = result + bias

        return result.reshape(*orig_shape[:-1], -1)

    def _sc_linear_combined_mp(self, x, weight, bias, operator, dispatch,
                               chunk_d=0, grouped=False):
        """Combined range-based + dynamic mixed precision SC linear.

        Iterates dispatch entries (range-based weight groups), and within each
        entry, further classifies input rows by dynamic MP metric.  The
        effective stoc_len for each (weight_group, input_row) pair is
        min(range_stoc_len, dynamic_stoc_len).

        Args:
            dispatch: Pre-built dispatch entries from range-based MP.
            Other args: same as _sc_linear.
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

        # Per-row dynamic stoc_len lookup
        dynamic_levels = self.sc_controller.adaptive_mp_config.stoc_len_levels \
            if self.sc_controller.adaptive_mp_config is not None \
            else self.sc_controller.mp_config.stoc_len_levels
        row_stoc_lens = torch.zeros(M, dtype=torch.long, device=x.device)
        for sl, rows in assignment.level_row_indices.items():
            if len(rows) > 0:
                row_stoc_lens[rows] = sl

        if grouped:
            matmul_fn = self._get_mlp_matmul_fn()
        else:
            matmul_fn = self._get_sc_matmul_fn()

        max_stoc_len = max(dynamic_levels)
        # Baseline: all M rows * all out_features at max_stoc_len
        # (includes pruned rows/groups for accurate savings)
        compute_baseline = M * weight.shape[0] * max_stoc_len
        compute_actual = 0.0

        result = torch.zeros(M, weight.shape[0],
                             device=x.device, dtype=torch.float32)

        for entry in dispatch:
            weight_sl = entry.stoc_len
            if weight_sl == 0:
                continue
            n_out = len(entry.out_indices)

            # For each dispatch entry, compute effective stoc_len per input row
            effective_sl = torch.minimum(
                row_stoc_lens,
                torch.tensor(weight_sl, device=x.device))

            # Group input rows by effective stoc_len
            unique_sls = effective_sl.unique()
            for eff_sl in unique_sls:
                eff_sl_val = eff_sl.item()
                if eff_sl_val == 0:
                    continue
                row_mask = effective_sl == eff_sl
                rows = torch.where(row_mask)[0]
                if len(rows) == 0:
                    continue

                n_rows = len(rows)
                compute_actual += n_rows * n_out * eff_sl_val

                sp = SCController.get_sc_prec_for_stoc_len(eff_sl_val)
                x_sub = x_flat[rows].contiguous()

                if chunk_d > 0 and D > chunk_d:
                    sub_result = None
                    for start in range(0, D, chunk_d):
                        end = min(start + chunk_d, D)
                        x_chunk = x_sub[:, start:end].contiguous()
                        w_chunk = entry.weight[:, start:end].contiguous()
                        config = self._get_sc_config(end - start, sp)

                        kwargs = dict(mode=self.sc_mode, sc_prec=sp,
                                      config=config, stoc_len=eff_sl_val)
                        if grouped:
                            kwargs.update(group_a=1, group_b=1)

                        chunk_out = matmul_fn(
                            x_chunk, w_chunk,
                            x_chunk.max().item(), x_chunk.min().item(),
                            w_chunk.max().item(), w_chunk.min().item(),
                            **kwargs)

                        if sub_result is None:
                            sub_result = chunk_out
                        else:
                            sub_result = sub_result + chunk_out
                    # Place results at correct output positions
                    result[rows.unsqueeze(1), entry.out_indices.unsqueeze(0)] = sub_result
                else:
                    config = self._get_sc_config(D, sp)
                    kwargs = dict(mode=self.sc_mode, sc_prec=sp,
                                  config=config, stoc_len=eff_sl_val)
                    if grouped:
                        kwargs.update(group_a=1, group_b=1)

                    sub = matmul_fn(
                        x_sub, entry.weight,
                        x_sub.max().item(), x_sub.min().item(),
                        entry.weight.max().item(), entry.weight.min().item(),
                        **kwargs)
                    result[rows.unsqueeze(1), entry.out_indices.unsqueeze(0)] = sub

        MPDistributionLogger.log_compute(
            self.sc_controller.current_timestep, self.block_idx,
            operator, compute_baseline, compute_actual)

        if bias is not None:
            result = result + bias

        return result.reshape(*orig_shape[:-1], -1)

    def _sc_linear(self, x, weight, bias, operator=None, chunk_d=0,
                    grouped=False, dispatch_key=None):
        """y = x @ W^T + bias, using SC matmul.

        If a dispatch table exists for this operator, iterates precomputed
        dispatch entries (mixed precision). Otherwise, single kernel launch.
        When both dispatch table (range-based) and dynamic MP are active,
        uses combined mode.

        Args:
            operator: Operator name for MP classification, logging, and
                precision_map lookup.
            chunk_d: If > 0, split D dimension into chunks.
            grouped: If True, use per-row-group quantization.
            dispatch_key: Override key for dispatch table lookup.
                Defaults to operator if not provided.  Useful when the
                dispatch table key differs from the operator name
                (e.g. "input_proj_q" vs "input_proj").
        """
        dk = dispatch_key if dispatch_key is not None else operator
        dispatch = self._dispatch_tables.get(dk) if dk else None
        has_dynamic_mp = (self.sc_controller.adaptive_mp_config is not None
                          or self.sc_controller.mp_config is not None)

        if dispatch is not None and has_dynamic_mp:
            # Combined mode: range-based dispatch + dynamic MP
            return self._sc_linear_combined_mp(x, weight, bias, operator,
                                                dispatch, chunk_d, grouped)

        if dispatch is None and has_dynamic_mp:
            return self._sc_linear_dynamic_mp(x, weight, bias, operator,
                                              chunk_d, grouped)
        if dispatch is None:
            # Uniform precision — single kernel launch
            stoc_len = self.sc_controller.get_stoc_len(self.block_idx, operator) if operator else self.sc_controller.stoc_len
            sc_prec = SCController.get_sc_prec_for_stoc_len(stoc_len)
            return self._sc_linear_uniform(x, weight, bias, sc_prec, stoc_len,
                                           chunk_d, grouped)

        # Mixed precision — iterate pre-built dispatch entries (range-based only)
        orig_shape = x.shape
        D = x.shape[-1]
        x_flat = x.reshape(-1, D)
        M = x_flat.shape[0]

        if grouped:
            matmul_fn = self._get_mlp_matmul_fn()
        else:
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
                sub_result = None
                for start in range(0, D, chunk_d):
                    end = min(start + chunk_d, D)
                    x_chunk = x_flat[:, start:end].contiguous()
                    w_chunk = entry.weight[:, start:end].contiguous()
                    chunk_size = end - start
                    config = self._get_sc_config(chunk_size, entry.sc_prec)

                    kwargs = dict(
                        mode=self.sc_mode, sc_prec=entry.sc_prec,
                        config=config, stoc_len=entry.stoc_len)
                    if grouped:
                        kwargs.update(group_a=1, group_b=1)

                    chunk_out = matmul_fn(
                        x_chunk, w_chunk,
                        x_chunk.max().item(), x_chunk.min().item(),
                        w_chunk.max().item(), w_chunk.min().item(),
                        **kwargs)

                    if sub_result is None:
                        sub_result = chunk_out
                    else:
                        sub_result = sub_result + chunk_out
                result[:, entry.out_indices] = sub_result
            else:
                config = self._get_sc_config(D, entry.sc_prec)
                kwargs = dict(
                    mode=self.sc_mode, sc_prec=entry.sc_prec,
                    config=config, stoc_len=entry.stoc_len)
                if grouped:
                    kwargs.update(group_a=1, group_b=1)

                sub = matmul_fn(x_flat, entry.weight,
                                x_flat.max().item(), x_flat.min().item(),
                                entry.weight.max().item(), entry.weight.min().item(),
                                **kwargs)
                result[:, entry.out_indices] = sub

        MPDistributionLogger.log_compute(
            self.sc_controller.current_timestep, self.block_idx,
            operator, compute_baseline, compute_actual)

        if bias is not None:
            result = result + bias

        return result.reshape(*orig_shape[:-1], -1)

    # =================================================================
    # QK and AV with per-head precision
    # =================================================================

    def _sc_qk(self, q, k):
        """SC q@k^T matmul — supports per-head mixed precision."""
        q_scaled = q * self.scale
        B, H, N, D = q_scaled.shape

        head_stoc_lens = self.sc_controller.get_group_stoc_lens(
            self.block_idx, "qk")

        # Dynamic MP: compute per-head stoc_lens from Q magnitude
        if head_stoc_lens is None and (self.sc_controller.adaptive_mp_config is not None
                                        or self.sc_controller.mp_config is not None):
            q_metric = q_scaled.float().abs().amax(dim=(0, 2, 3))  # [H]
            MetricProfiler.record(q_metric, self.sc_controller.current_timestep,
                                  self.block_idx, "qk")

            if self.sc_controller.adaptive_mp_config is not None:
                assignment = adaptive_classify_rows(
                    q_metric,
                    self.sc_controller.current_timestep,
                    self.sc_controller.total_timesteps,
                    self.sc_controller.adaptive_mp_config,
                    operator="qk",
)
            else:
                mp_config = self.sc_controller.mp_config
                assignment = classify_rows_by_metric(
                    q_metric, mp_config.stoc_len_levels, mp_config.level_fractions)

            MPDistributionLogger.log(
                self.sc_controller.current_timestep, self.block_idx,
                "qk", assignment, H)

            head_stoc_lens = [0] * H
            for sl, heads in assignment.level_row_indices.items():
                for h_idx in heads:
                    head_stoc_lens[h_idx.item()] = sl

        if head_stoc_lens is None:
            # Uniform precision — existing fast path
            stoc_len = self.sc_controller.get_stoc_len(self.block_idx, "qk")
            sc_prec = SCController.get_sc_prec_for_stoc_len(stoc_len)
            return self._sc_qk_uniform(q_scaled, k, sc_prec, stoc_len)

        # Mixed: group heads by stoc_len, compute within each group
        output = torch.zeros(B, H, N, N, device=q.device, dtype=torch.float32)
        stoc_len_to_heads: dict[int, list[int]] = defaultdict(list)
        for h, sl in enumerate(head_stoc_lens):
            stoc_len_to_heads[sl].append(h)

        max_stoc_len = max(head_stoc_lens)
        compute_baseline = B * H * N * N * max_stoc_len
        compute_actual = 0.0

        for sl, heads in stoc_len_to_heads.items():
            # Pruned heads (stoc_len=0): output already zeroed
            if sl == 0:
                continue
            compute_actual += B * len(heads) * N * N * sl

            sp = SCController.get_sc_prec_for_stoc_len(sl)
            config = self._get_sc_config(D, sp)

            for h in heads:
                # Extract single head: [B, 1, N, D] -> [B, N, D]
                q_h = q_scaled[:, h].float()
                k_h = k[:, h].float()

                if self.sc_controller.sc_enable and self.sc_mode == "bipolar":
                    q_maxs = q_h.amax(dim=(1, 2))
                    q_mins = q_h.amin(dim=(1, 2))
                    k_maxs = k_h.amax(dim=(1, 2))
                    k_mins = k_h.amin(dim=(1, 2))
                    out_h = self._get_batched_bipolar_fn()(
                        q_h, k_h, q_maxs, q_mins, k_maxs, k_mins,
                        sp, config, stoc_len=sl,
                    )
                    output[:, h] = out_h
                else:
                    matmul_fn = self._get_sc_matmul_fn()
                    for b_idx in range(B):
                        qi = q_h[b_idx]
                        ki = k_h[b_idx]
                        output[b_idx, h] = matmul_fn(
                            qi, ki,
                            qi.max().item(), qi.min().item(),
                            ki.max().item(), ki.min().item(),
                            mode=self.sc_mode, sc_prec=sp, config=config,
                            stoc_len=sl,
                        )

        MPDistributionLogger.log_compute(
            self.sc_controller.current_timestep, self.block_idx,
            "qk", compute_baseline, compute_actual)

        return output

    def _sc_qk_uniform(self, q_scaled, k, sc_prec, stoc_len):
        """Uniform precision QK — fully batched kernel or per-head loop."""
        B, H, N, D = q_scaled.shape

        # Noise-model fast path: direct call into the core, no adapter chain.
        if self.sc_controller.noise_model:
            from .noise_matmul import _noisy_matmul_core
            q_flat = q_scaled.reshape(B * H, N, D)
            k_flat = k.reshape(B * H, N, D)
            output = _noisy_matmul_core(
                q_flat, k_flat, L=stoc_len, mode=self.sc_mode,
                per_row_scale=False,
            )
            return output.reshape(B, H, N, N)

        config = self._get_sc_config(D, sc_prec)

        q_flat = q_scaled.reshape(B * H, N, D).float()
        k_flat = k.reshape(B * H, N, D).float()

        q_maxs = q_flat.amax(dim=(1, 2))
        q_mins = q_flat.amin(dim=(1, 2))
        k_maxs = k_flat.amax(dim=(1, 2))
        k_mins = k_flat.amin(dim=(1, 2))

        # Fast path: fully batched kernel
        if self.sc_controller.sc_enable and self.sc_mode == "bipolar":
            output = self._get_batched_bipolar_fn()(
                q_flat, k_flat, q_maxs, q_mins, k_maxs, k_mins,
                sc_prec, config, stoc_len=stoc_len,
            )
            return output.reshape(B, H, N, N)

        # Fallback: per-head loop
        matmul_fn = self._get_sc_matmul_fn()
        q_maxs_cpu = q_maxs.cpu()
        q_mins_cpu = q_mins.cpu()
        k_maxs_cpu = k_maxs.cpu()
        k_mins_cpu = k_mins.cpu()

        BH = B * H
        output = torch.empty(BH, N, N, dtype=torch.float32, device=q_scaled.device)
        for i in range(BH):
            output[i] = matmul_fn(
                q_flat[i], k_flat[i],
                q_maxs_cpu[i].item(), q_mins_cpu[i].item(),
                k_maxs_cpu[i].item(), k_mins_cpu[i].item(),
                mode=self.sc_mode, sc_prec=sc_prec, config=config,
                stoc_len=stoc_len,
            )

        return output.reshape(B, H, N, N)

    def _sc_av(self, attn, v):
        """attn @ v using SC — supports per-head mixed and dynamic per-row MP."""
        B, H, N, D = v.shape

        head_stoc_lens = self.sc_controller.get_group_stoc_lens(
            self.block_idx, "av")

        if (head_stoc_lens is None
                and self.sc_controller.mp_config is None
                and self.sc_controller.adaptive_mp_config is None):
            # Branch 1: Uniform — existing fast path (unchanged)
            stoc_len = self.sc_controller.get_stoc_len(self.block_idx, "av")
            sc_prec = SCController.get_sc_prec_for_stoc_len(stoc_len)
            return self._sc_av_uniform(attn, v, sc_prec, stoc_len)

        # Shared setup for mixed paths
        G_attn = self.av_attn_group_size if self.av_attn_group_size > 0 else N
        G_v = self.av_v_group_size if self.av_v_group_size > 0 else D

        grouped_fn = self._get_av_grouped_fn()

        BH = B * H
        output = torch.zeros(BH, N, D, dtype=torch.float32, device=v.device)
        attn_flat = attn.reshape(BH, N, N).float()
        v_t_flat = v.reshape(BH, N, D).float().transpose(1, 2).contiguous()

        if head_stoc_lens is not None:
            # Branch 2: Static per-head mixed (existing code, unchanged)
            stoc_len_to_heads: dict[int, list[int]] = defaultdict(list)
            for h, sl in enumerate(head_stoc_lens):
                stoc_len_to_heads[sl].append(h)

            for sl, heads in stoc_len_to_heads.items():
                if sl == 0:
                    continue  # pruned heads: output already zeroed
                sp = SCController.get_sc_prec_for_stoc_len(sl)
                config = self._get_sc_config(N, sp)
                for h in heads:
                    for b_idx in range(B):
                        i = b_idx * H + h
                        output[i] = grouped_fn(
                            attn_flat[i], v_t_flat[i],
                            group_a=G_attn, group_b=G_v,
                            mode=self.sc_mode, sc_prec=sp, config=config,
                            stoc_len=sl)
        else:
            # Branch 3: Dynamic per-token-row MP (adaptive or fixed)
            dynamic_levels = (self.sc_controller.adaptive_mp_config.stoc_len_levels
                              if self.sc_controller.adaptive_mp_config is not None
                              else self.sc_controller.mp_config.stoc_len_levels)
            max_stoc_len = max(dynamic_levels)
            compute_baseline = 0
            compute_actual = 0.0

            for i in range(BH):
                row_max = attn_flat[i].amax(dim=-1)  # [N]
                if i == 0:  # profile once per (t, block)
                    MetricProfiler.record(row_max, self.sc_controller.current_timestep,
                                          self.block_idx, "av")

                if self.sc_controller.adaptive_mp_config is not None:
                    assignment = adaptive_classify_rows(
                        row_max,
                        self.sc_controller.current_timestep,
                        self.sc_controller.total_timesteps,
                        self.sc_controller.adaptive_mp_config,
                        operator="av",
)
                else:
                    mp_config = self.sc_controller.mp_config
                    assignment = classify_rows_by_metric(
                        row_max, mp_config.stoc_len_levels, mp_config.level_fractions)

                # Log only once per (timestep, block) — first BH index
                if i == 0:
                    MPDistributionLogger.log(
                        self.sc_controller.current_timestep, self.block_idx,
                        "av", assignment, N)

                for sl, rows in assignment.level_row_indices.items():
                    if len(rows) == 0 or sl == 0:
                        continue  # pruned rows: output already zeroed
                    n_rows = len(rows)
                    compute_baseline += n_rows * D * max_stoc_len
                    compute_actual += n_rows * D * sl

                    sp = SCController.get_sc_prec_for_stoc_len(sl)
                    config = self._get_sc_config(N, sp)
                    attn_sub = attn_flat[i][rows]
                    output[i, rows] = grouped_fn(
                        attn_sub, v_t_flat[i],
                        group_a=min(G_attn, len(rows)), group_b=G_v,
                        mode=self.sc_mode, sc_prec=sp, config=config,
                        stoc_len=sl)

            MPDistributionLogger.log_compute(
                self.sc_controller.current_timestep, self.block_idx,
                "av", compute_baseline, compute_actual)

        return output.reshape(B, H, N, D)

    def _sc_av_uniform(self, attn, v, sc_prec, stoc_len):
        """Uniform precision AV — per-head loop."""
        B, H, N, D = v.shape

        # Noise-model fast path: all BH heads in ONE direct call to the core.
        if self.sc_controller.noise_model:
            from .noise_matmul import _noisy_matmul_core
            BH = B * H
            attn_flat = attn.reshape(BH, N, N)
            v_t_flat = v.reshape(BH, N, D).transpose(1, 2).contiguous()
            output = _noisy_matmul_core(
                attn_flat, v_t_flat, L=stoc_len, mode=self.sc_mode,
                per_row_scale=True,
            )
            return output.reshape(B, H, N, D)

        config = self._get_sc_config(N, sc_prec)
        G_attn = self.av_attn_group_size if self.av_attn_group_size > 0 else N
        G_v = self.av_v_group_size if self.av_v_group_size > 0 else D

        grouped_fn = self._get_av_grouped_fn()

        BH = B * H
        output = torch.empty(BH, N, D, dtype=torch.float32, device=v.device)
        attn_flat = attn.reshape(BH, N, N).float()
        v_t_flat = v.reshape(BH, N, D).float().transpose(1, 2).contiguous()

        for i in range(BH):
            output[i] = grouped_fn(
                attn_flat[i], v_t_flat[i],
                group_a=G_attn, group_b=G_v,
                mode=self.sc_mode, sc_prec=sc_prec, config=config,
                stoc_len=stoc_len)

        return output.reshape(B, H, N, D)

    def to(self, *args, **kwargs):
        super(SCAttention, self).to(*args, **kwargs)
        self.qkv = self.qkv.to(*args, **kwargs)
        self.proj = self.proj.to(*args, **kwargs)
        self.input_quant = self.input_quant.to(*args, **kwargs)
        self.act_quant = self.act_quant.to(*args, **kwargs)
        if self.quantize_bmm_input:
            self.q_quant = self.q_quant.to(*args, **kwargs)
            self.v_quant = self.v_quant.to(*args, **kwargs)
            self.k_quant = self.k_quant.to(*args, **kwargs)
        if self.reorder_index_qkv is not None:
            self.reorder_index_qkv = self.reorder_index_qkv.to(*args, **kwargs)
        if self.reorder_index_proj is not None:
            self.reorder_index_proj = self.reorder_index_proj.to(*args, **kwargs)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        debug = self.sc_controller.debug

        if self.reorder_index_qkv is not None:
            x = torch.index_select(x, 2, self.reorder_index_qkv)
        x = self.input_quant(x)

        # === INPUT PROJECTION ===
        # Force FP for the last block
        force_fp_input_proj = self.block_idx >= self.sc_controller.total_blocks - 1
        if not force_fp_input_proj and self.sc_controller.use_sc_for_input_proj(self.block_idx):
            w = self.qkv.weight  # [3*C, C]
            b = self.qkv.bias    # [3*C] or None
            if self.sc_controller.noise_model:
                # Fused QKV: one surrogate call instead of three. Per-block
                # this saves 2 _sc_linear calls × 28 blocks = 56 calls/timestep.
                qkv_sc = self._sc_linear(
                    x, w, b, operator="input_proj",
                    chunk_d=0, grouped=True, dispatch_key=None,
                )
            else:
                q_sc = self._sc_linear(x, w[:C], b[:C] if b is not None else None,
                                       operator="input_proj", chunk_d=144, grouped=True,
                                       dispatch_key="input_proj_q")
                k_sc = self._sc_linear(x, w[C:2*C], b[C:2*C] if b is not None else None,
                                       operator="input_proj", chunk_d=144, grouped=True,
                                       dispatch_key="input_proj_k")
                v_sc = self._sc_linear(x, w[2*C:], b[2*C:] if b is not None else None,
                                       operator="input_proj", chunk_d=144, grouped=True,
                                       dispatch_key="input_proj_v")
                qkv_sc = torch.cat([q_sc, k_sc, v_sc], dim=-1)
            if debug:
                qkv_fp = self.qkv(x)
                self._log_compare("input_proj", qkv_fp, qkv_sc)
                self.sc_controller.log_debug(self.block_idx, "input_proj", qkv_fp, qkv_sc)
            qkv = qkv_sc
        else:
            qkv = self.qkv(x)

        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.quantize_bmm_input:
            q = self.q_quant(q)
            k = self.k_quant(k)
            v = self.v_quant(v)

        # === QK MATMUL ===
        if self.sc_controller.use_sc_for_qk(self.block_idx):
            if debug:
                q_scaled = q * self.scale
                attn_fp = q_scaled @ k.transpose(-2, -1)
                attn_sc = self._sc_qk(q, k)
                self.sc_controller.log_debug(self.block_idx, "qk", attn_fp, attn_sc)
                attn = attn_sc
            else:
                attn = self._sc_qk(q, k)
        else:
            q_scaled = q * self.scale
            attn = q_scaled @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # === AV MATMUL ===
        if self.sc_controller.use_sc_for_av(self.block_idx):
            if debug:
                x_fp = attn @ v
                x_sc = self._sc_av(attn, v)
                self.sc_controller.log_debug(self.block_idx, "av", x_fp, x_sc)
                x = x_sc
            else:
                x = self._sc_av(attn, v)
        else:
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)

        if self.reorder_index_proj is not None:
            x = torch.index_select(x, 2, self.reorder_index_proj)
        x = self.act_quant(x)

        # === OUTPUT PROJECTION ===
        if self.sc_controller.use_sc_for_proj(self.block_idx):
            x_sc = self._sc_linear(x, self.proj.weight, self.proj.bias,
                                   operator="proj", chunk_d=144, grouped=True)
            if debug:
                x_fp = self.proj(x)
                self._log_compare("proj", x_fp, x_sc)
                self.sc_controller.log_debug(self.block_idx, "proj", x_fp, x_sc)
            x = x_sc
            x = self.proj_drop(x)
        else:
            x = self.proj(x)
            x = self.proj_drop(x)

        return x

    def extra_repr(self) -> str:
        return (
            f"block_idx={self.block_idx}, "
            f"num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, "
            f"sc_prec={self.sc_controller.sc_prec}"
        )
