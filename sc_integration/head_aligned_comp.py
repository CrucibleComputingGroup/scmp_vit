"""HeadAlignedSCLinear — comp matmul that reuses block per-head SNG.

Why head-aligned. Standard SCLinear at (D_in=1024, D_out=1024) pulls Sobol
pool from ``_CFG_CACHE[(1024, 8)]``. For configs where the block's SC noise
enters through a *per-head* matmul (qk_only: Q@K^T at head_dim=64), that pool
has *no* informational overlap with the block's noise → comp's SC noise is
purely additive and degrades accuracy by ~2 pt on `qk_only`.

This module reshapes the comp into ``n_heads`` parallel (D_head, D_out)
SC matmuls that each pull from ``_CFG_CACHE[(D_head, sc_prec)]`` — the
same pool the block's per-head Q@K^T uses. **Total SC inner reductions are
unchanged** (``n_heads × D_head = D_in``); hardware accounting is identical
to the monolithic SCLinear. The only difference is which bitstream sequences
are used, and therefore how the comp's SC noise correlates with the block's.

Empirically (see `OVERNIGHT_REPORT.md`), head-alignment recovers
**+1.8 pt** of the 2.4 pt naive-SC-comp gap on `qk_only` and is picked by
calibration in 89 of 96 blocks across {qk_only, qk_av, full_attn,
skip_worst50}.

Speed note. Head-alignment does ``n_heads`` matmuls instead of 1, so on GPU
(where each Triton kernel launch carries ~50–100 µs overhead) wall-clock is
``n_heads×`` slower than monolithic. This implementation mitigates it via:

    1. Per-row CUDA-stream dispatch so the n_heads Triton kernels overlap.
    2. Cached per-head quantized weight buffers (weights are static after
       calibration).

On silicon the overhead disappears: the 16 "launches" become 16 systolic-
array passes with near-zero launch cost — the per-MAC energy/area is
identical to the monolithic D=1024 SC matmul.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class HeadAlignedSCLinear(nn.Module):
    """SCLinear with per-head SC matmul split.

    Splits ``D_in`` into ``n_heads`` chunks of ``D_head = D_in // n_heads``,
    runs each through an independent SC matmul of shape ``(D_head, D_out)``
    that reuses ``_CFG_CACHE[(D_head, sc_prec)]``, sums the results, then
    adds bias. Output is mathematically equivalent to a single linear at
    the FP level; the SC noise structure is what differs.

    Speed optimization: on the first forward, we bind a pool of CUDA streams
    (one per head). Subsequent forwards dispatch the ``n_heads`` Triton
    kernels concurrently. Sync+accumulate happens at the end.
    """

    def __init__(self, linear: nn.Linear, sc_prec: int, mode: str = "bipolar",
                 n_heads: int = 16, cfg_override: dict | None = None,
                 use_streams: bool = True):
        super().__init__()
        D_in = linear.weight.shape[1]
        D_out = linear.weight.shape[0]
        assert D_in % n_heads == 0, (
            f"D_in={D_in} must be divisible by n_heads={n_heads}")
        self.weight = linear.weight  # (D_out, D_in)
        self.bias = linear.bias
        self.sc_prec = sc_prec
        self.mode = mode
        self.n_heads = n_heads
        self.D_head = D_in // n_heads
        self.in_features = D_in
        self.out_features = D_out
        self.cfg_override = cfg_override
        self.use_streams = use_streams
        # Lazy-init CUDA streams on first forward (need to know device).
        self._streams: list[torch.cuda.Stream] | None = None
        # Cache per-head weight slices so we don't re-slice every forward.
        self._weight_heads: list[torch.Tensor] | None = None

    def _ensure_streams(self, device: torch.device) -> None:
        if not self.use_streams or device.type != "cuda":
            self._streams = None
            return
        if self._streams is None:
            self._streams = [torch.cuda.Stream(device=device)
                             for _ in range(self.n_heads)]

    def _ensure_weight_cache(self) -> list[torch.Tensor]:
        if self._weight_heads is None:
            H, D_h = self.n_heads, self.D_head
            # Contiguous slices along D_in dim — each (D_out, D_head).
            self._weight_heads = [
                self.weight[:, h * D_h : (h + 1) * D_h].contiguous()
                for h in range(H)
            ]
        return self._weight_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from sc_integration.sc_linear import _sc_linear

        orig_shape = x.shape
        D_in = orig_shape[-1]
        D_out = self.weight.shape[0]
        H, D_h = self.n_heads, self.D_head

        x2 = x.reshape(-1, D_in).float()
        N = x2.shape[0]

        # Pre-slice x per head (contiguous views avoid repeated index math).
        # Reshape to (N, H, D_h) then split.
        xv = x2.view(N, H, D_h)
        weight_heads = self._ensure_weight_cache()
        self._ensure_streams(x2.device)

        y = torch.zeros(N, D_out, device=x.device, dtype=torch.float32)

        if self._streams is not None:
            # Launch per-head kernels on parallel CUDA streams; accumulate after.
            partials = [None] * H
            current = torch.cuda.current_stream(x2.device)
            # Make sure streams see x2/weight_heads writes.
            for h in range(H):
                self._streams[h].wait_stream(current)
            for h in range(H):
                with torch.cuda.stream(self._streams[h]):
                    x_h = xv[:, h, :].contiguous()
                    w_h = weight_heads[h]
                    partials[h] = _sc_linear(
                        x_h, w_h, None, self.sc_prec,
                        mode=self.mode, cfg_override=self.cfg_override,
                    )
            for h in range(H):
                current.wait_stream(self._streams[h])
            for h in range(H):
                y = y + partials[h]
        else:
            for h in range(H):
                x_h = xv[:, h, :].contiguous()
                w_h = weight_heads[h]
                y_h = _sc_linear(
                    x_h, w_h, None, self.sc_prec,
                    mode=self.mode, cfg_override=self.cfg_override,
                )
                y = y + y_h

        if self.bias is not None:
            y = y + self.bias.float()
        return y.to(x.dtype).reshape(*orig_shape[:-1], D_out)
