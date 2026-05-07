"""Minimal copy of UniformQuantizer + QuantLinear from RepQ-ViT.

We can't import the upstream ``quant`` package directly because its
``__init__.py`` transitively imports a timm-version-incompatible
``utils/vision_transformer.py``. Vendor the two classes we actually need
(asymmetric per-tensor / per-channel uniform fake-quant and a Linear that
applies it). Behavior matches
``third_party/QwT-SC/QwT-cls-RepQ-ViT/quant/{quantizer,quant_modules}.py``
verbatim except for the LogSqrt2 / Conv / MatMul classes we don't use.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _lp_loss(pred, tgt, p=2.0, reduction="all"):
    return (pred - tgt).abs().pow(p).mean()


class _RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class UniformQuantizer(nn.Module):
    """Asymmetric uniform fake-quant. Per-tensor by default, channel-wise
    optional. Scales are computed lazily from the first input batch
    (percentile sweep over {0.999, 0.9999, 0.99999} picking lowest L2)."""

    def __init__(self, n_bits: int = 8, channel_wise: bool = False):
        super().__init__()
        assert 2 <= n_bits <= 8, f"bitwidth {n_bits} not supported"
        self.n_bits = n_bits
        self.n_levels = 2 ** n_bits
        self.delta = None
        self.zero_point = None
        self.inited = False
        self.channel_wise = channel_wise

    def forward(self, x: torch.Tensor):
        if not self.inited:
            self.delta, self.zero_point = self.init_quantization_scale(
                x, self.channel_wise)
            self.inited = True
        x_int = _RoundSTE.apply(x / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta
        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool):
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[-1] if x.ndim == 3 else x_clone.shape[0]
            if x.ndim == 4:
                x_max = (x_clone.abs().max(dim=-1)[0].max(dim=-1)[0]
                         .max(dim=-1)[0])
            elif x.ndim == 2:
                x_max = x_clone.abs().max(dim=-1)[0]
            elif x.ndim == 3:
                x_max = x_clone.abs().max(dim=0)[0].max(dim=0)[0]
            else:
                raise NotImplementedError(f"x.ndim={x.ndim}")
            delta = x_max.clone()
            zero_point = x_max.clone()
            for c in range(n_channels):
                if x.ndim == 3:
                    delta[c], zero_point[c] = self.init_quantization_scale(
                        x_clone[:, :, c], channel_wise=False)
                else:
                    delta[c], zero_point[c] = self.init_quantization_scale(
                        x_clone[c], channel_wise=False)
            if x.ndim == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            elif x.ndim == 2:
                delta = delta.view(-1, 1)
                zero_point = zero_point.view(-1, 1)
            elif x.ndim == 3:
                delta = delta.view(1, 1, -1)
                zero_point = zero_point.view(1, 1, -1)
            return delta, zero_point

        x_clone = x.clone().detach()
        best_score = 1e10
        delta = None
        zero_point = None
        for pct in (0.999, 0.9999, 0.99999):
            try:
                new_max = torch.quantile(x_clone.reshape(-1), pct)
                new_min = torch.quantile(x_clone.reshape(-1), 1.0 - pct)
            except Exception:
                new_max = torch.tensor(
                    np.percentile(x_clone.reshape(-1).cpu(), pct * 100),
                    device=x_clone.device, dtype=torch.float32)
                new_min = torch.tensor(
                    np.percentile(x_clone.reshape(-1).cpu(), (1 - pct) * 100),
                    device=x_clone.device, dtype=torch.float32)
            x_q = self._quantize(x_clone, new_max, new_min)
            score = _lp_loss(x_clone, x_q)
            if score < best_score:
                best_score = score
                delta = (new_max - new_min) / (2 ** self.n_bits - 1)
                zero_point = (-new_min / delta).round()
        return delta, zero_point

    def _quantize(self, x, mx, mn):
        delta = (mx - mn) / (2 ** self.n_bits - 1)
        zero_point = (-mn / delta).round()
        x_int = _RoundSTE.apply(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        return (x_quant - zero_point) * delta


class QuantLinear(nn.Linear):
    """nn.Linear with asymmetric uniform fake-quant on activations and weights."""

    def __init__(self, in_features, out_features,
                 input_quant_params=None, weight_quant_params=None):
        super().__init__(in_features, out_features)
        self.input_quantizer = UniformQuantizer(**(input_quant_params or {}))
        self.weight_quantizer = UniformQuantizer(**(weight_quant_params or {}))
        self.use_input_quant = False
        self.use_weight_quant = False

    def set_quant_state(self, input_quant=False, weight_quant=False):
        self.use_input_quant = input_quant
        self.use_weight_quant = weight_quant

    def forward(self, x):
        if self.use_input_quant:
            x = self.input_quantizer(x)
        w = self.weight_quantizer(self.weight) if self.use_weight_quant else self.weight
        return F.linear(x, weight=w, bias=self.bias)
