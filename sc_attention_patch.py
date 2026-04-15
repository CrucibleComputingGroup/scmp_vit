"""Monkey-patch DINOv2 attention + MLP to use stochastic computing matmuls.

- Q @ K^T: batched bipolar enable-signal SC via sc_matmul_enable_batched_bipolar.
- Attn @ V: per-head enable-signal SC via sc_matmul_grouped_enable_triton.
- MLP fc1 / fc2: SC linear via sc_matmul_grouped_enable_triton.

Softmax, layernorms, activations, and qkv/proj linears stay FP.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Local SC primitives (vit_sc/sc) and thin integration wrappers
# (vit_sc/sc_integration).
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "sc"))
sys.path.insert(0, str(HERE))

from sc_triton import (  # noqa: E402
    sc_matmul_grouped_enable_triton as sc_matmul_grouped_bipolar,
    sc_matmul_enable_batched_bipolar,
)
from sc_integration.noise_matmul import noisy_sc_matmul_grouped, _noisy_matmul_core  # noqa: E402
from sc_integration.mp_linear import (  # noqa: E402
    MPConfig, RangeMPConfig, DispatchEntry,
    build_range_dispatch, classify_input_rows, sc_prec_for_stoc_len,
)
from mp_config import classify_rows_by_metric  # noqa: E402  # local copy in sc/
from config_helpers import make_sobol_simple_config  # noqa: E402

# Module-level toggle: when True, SC matmuls use the calibrated Gaussian-noise
# surrogate (sc_integration/noise_matmul.py) instead of the real bitstream /
# enable-signal kernels. Orders of magnitude faster — for sensitivity sweeps.
USE_NOISE_MODEL = False


def set_noise_model(enabled: bool):
    global USE_NOISE_MODEL
    USE_NOISE_MODEL = bool(enabled)


_CFG_CACHE = {}


def _get_config(D, sc_prec):
    key = (D, sc_prec)
    if key not in _CFG_CACHE:
        _CFG_CACHE[key] = make_sobol_simple_config(D, D, sc_prec)
    return _CFG_CACHE[key]


def _sym_range(t: torch.Tensor) -> tuple[float, float]:
    m = float(t.detach().abs().max().item())
    m = max(m, 1e-6)
    return m, -m


def _sc_linear(x: torch.Tensor, weight: torch.Tensor, bias, sc_prec: int,
               mode: str = "bipolar", cfg_override: dict = None):
    """y = x @ weight^T + bias via SC.

    mode="bipolar": per-row symmetric quant (|max|), q_max = 2^(sc_prec-1)-1.
        Good for near-zero-mean tensors (post-LN activations, weights).
    mode="unipolar": per-row asymmetric quant (scale = range/q_max, zero-point),
        q_max = 2^sc_prec - 1 → ~2× grid resolution. Good for heavy-tailed /
        asymmetric tensors (post-GELU activations for fc2).

    ``cfg_override``: when not None, use this SC config instead of the cached
    default — lets the comp use an alternative Sobol scrambling per block.
    """
    orig_shape = x.shape
    D_in = orig_shape[-1]
    x2 = x.reshape(-1, D_in).float()
    w = weight.float()  # (D_out, D_in)
    if USE_NOISE_MODEL:
        y = noisy_sc_matmul_grouped(x2, w, group_a=1, group_b=1,
                                    mode=mode, sc_prec=sc_prec)
    else:
        cfg = cfg_override if cfg_override is not None else _get_config(D_in, sc_prec)
        y = sc_matmul_grouped_bipolar(x2, w, group_a=1, group_b=1,
                                      mode=mode, sc_prec=sc_prec, config=cfg)
    if bias is not None:
        y = y + bias.float()
    y = y.to(x.dtype).reshape(*orig_shape[:-1], w.shape[0])
    return y


def _classify_per_head(q: torch.Tensor, cfg: MPConfig) -> list[int]:
    """Per-head ``stoc_len`` assignment matching scmp_llm's ``_sc_qk``:
    metric = per-head ``|q|.amax(dim=(0, 2, 3))`` → fixed-quantile classify.
    Returns a length-H list of stoc_len values."""
    H = q.shape[1]
    metric = q.float().abs().amax(dim=(0, 2, 3))  # (H,)
    assignment = classify_rows_by_metric(
        metric, cfg.stoc_len_levels, cfg.level_fractions,
    )
    head_sl = [0] * H
    for sl, heads in assignment.level_row_indices.items():
        for h in heads:
            head_sl[h.item()] = int(sl)
    return head_sl


def _classify_attn_rows(attn_row_max: torch.Tensor, cfg: MPConfig) -> "RowAssignment":
    """Per-attn-row ``stoc_len`` assignment matching scmp_llm's AV dynamic path:
    metric = ``attn_row.amax(-1)`` → fixed-quantile classify. Called once per head."""
    return classify_rows_by_metric(
        attn_row_max, cfg.stoc_len_levels, cfg.level_fractions,
    )


def _sc_qk_mp(q: torch.Tensor, k: torch.Tensor, sc_prec: int,
              mp_cfg: MPConfig) -> torch.Tensor:
    """Per-head mixed-precision Q@K^T. q, k: (B, H, N, D). Returns (B, H, N, N)
    unscaled (caller applies self.scale).
    """
    B, H, N, D = q.shape
    head_sl = _classify_per_head(q, mp_cfg)

    # Bucket heads by stoc_len, one matmul per bucket.
    stoc_len_to_heads: dict[int, list[int]] = {}
    for h, sl in enumerate(head_sl):
        stoc_len_to_heads.setdefault(sl, []).append(h)

    output = torch.zeros(B, H, N, N, dtype=torch.float32, device=q.device)
    for sl, heads in stoc_len_to_heads.items():
        if sl == 0 or not heads:
            continue
        idx = torch.as_tensor(heads, dtype=torch.long, device=q.device)
        sp = sc_prec_for_stoc_len(sl)
        q_sub = q.index_select(1, idx).contiguous()  # (B, h', N, D)
        k_sub = k.index_select(1, idx).contiguous()

        if USE_NOISE_MODEL:
            out_sub = _noisy_matmul_core(
                q_sub.float(), k_sub.float(), L=sl,
                mode="bipolar", per_row_scale=False,
            )
            output.index_copy_(1, idx, out_sub.float())
        else:
            # Use the batched enable kernel (scmp_llm's QK MP path) — takes
            # floats + per-head min/max, accepts stoc_len natively.
            Bq, Hh, Nq, Dq = q_sub.shape
            qf = q_sub.float().reshape(Bq * Hh, Nq, Dq).contiguous()
            kf = k_sub.float().reshape(Bq * Hh, Nq, Dq).contiguous()
            q_maxs = qf.amax(dim=(1, 2))
            q_mins = qf.amin(dim=(1, 2))
            k_maxs = kf.amax(dim=(1, 2))
            k_mins = kf.amin(dim=(1, 2))
            cfg = _get_config(Dq, sp)
            attn_sub = sc_matmul_enable_batched_bipolar(
                qf, kf, q_maxs, q_mins, k_maxs, k_mins,
                sp, cfg, stoc_len=int(sl),
            ).reshape(Bq, Hh, Nq, Nq)
            output.index_copy_(1, idx, attn_sub.float())

    return output


def _sc_av_mp(attn: torch.Tensor, v: torch.Tensor, sc_prec: int,
              mp_cfg: MPConfig) -> torch.Tensor:
    """Per-attn-row mixed-precision Attn@V. attn: (B, H, N, N), v: (B, H, N, D).
    Per scmp_llm: for each (b, h), classify the N rows by row-max and bucket.
    """
    B, H, N, D = v.shape
    BH = B * H
    attn_flat = attn.float().reshape(BH, N, N)
    b_flat = v.float().transpose(-1, -2).reshape(BH, D, N)  # (BH, D, N)
    out = torch.zeros(BH, N, D, dtype=torch.float32, device=v.device)

    for i in range(BH):
        row_max = attn_flat[i].amax(dim=-1)  # (N,)
        assignment = _classify_attn_rows(row_max, mp_cfg)
        for sl, rows in assignment.level_row_indices.items():
            if sl == 0 or rows.numel() == 0:
                continue
            sp = sc_prec_for_stoc_len(int(sl))
            attn_sub = attn_flat[i].index_select(0, rows).contiguous()  # (n_sel, N)

            if USE_NOISE_MODEL:
                sub = _noisy_matmul_core(
                    attn_sub, b_flat[i], L=int(sl),
                    mode="bipolar", per_row_scale=True,
                )
            else:
                cfg = _get_config(N, sp)
                sub = sc_matmul_grouped_bipolar(
                    attn_sub, b_flat[i],
                    group_a=1, group_b=1,
                    mode="bipolar", sc_prec=sp, config=cfg, stoc_len=int(sl),
                )
            out[i, rows] = sub

    return out.reshape(B, H, N, D)


def make_sc_attention_forward(sc_prec: int, sc_av: bool, sc_qk: bool = True,
                              qk_mp_cfg: MPConfig | None = None,
                              av_mp_cfg: MPConfig | None = None):
    """Return a forward(self, x) with optional SC Q@K^T and SC Attn@V.

    When ``qk_mp_cfg`` / ``av_mp_cfg`` are provided, per-head / per-attn-row
    fixed mixed precision is applied (scmp_llm-compatible). When they are
    ``None`` the original uniform path is taken — unchanged numerics.
    """

    def forward(self, x, attn_bias=None):
        B, N, C = x.shape
        H = self.num_heads
        D = C // H
        qkv = self.qkv(x).reshape(B, N, 3, H, D)
        q, k, v = torch.unbind(qkv, 2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if sc_qk:
            if qk_mp_cfg is not None:
                attn = _sc_qk_mp(q, k, sc_prec, qk_mp_cfg).to(x.dtype) * self.scale
            elif USE_NOISE_MODEL:
                # (B,H,N,D) @ (B,H,N,D)^T via surrogate (per-head amax scale).
                attn = _noisy_matmul_core(
                    q.float(), k.float(), L=2 ** sc_prec,
                    mode="bipolar", per_row_scale=False,
                ).to(x.dtype) * self.scale
            else:
                # Aligned with scmp_llm / MP path: enable-signal batched kernel,
                # takes floats + per-head (min, max), faster and more numerically
                # stable than the packed-XNOR sc_matmul_qk_multihead path.
                qf = q.float().reshape(B * H, N, D).contiguous()
                kf = k.float().reshape(B * H, N, D).contiguous()
                q_maxs = qf.amax(dim=(1, 2))
                q_mins = qf.amin(dim=(1, 2))
                k_maxs = kf.amax(dim=(1, 2))
                k_mins = kf.amin(dim=(1, 2))
                cfg_qk = _get_config(D, sc_prec)
                attn_sc = sc_matmul_enable_batched_bipolar(
                    qf, kf, q_maxs, q_mins, k_maxs, k_mins,
                    sc_prec, cfg_qk, stoc_len=2 ** sc_prec,
                ).reshape(B, H, N, N)
                attn = attn_sc.to(x.dtype) * self.scale
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        if sc_av:
            if av_mp_cfg is not None:
                out = _sc_av_mp(attn, v, sc_prec, av_mp_cfg).to(x.dtype)
            else:
                # Use sc_matmul_grouped (unipolar + signs) with per-row quant to
                # handle the very wide dynamic range of softmax'd attn.
                a_flat = attn.float().reshape(B * H, N, N)
                b_flat = v.float().transpose(-1, -2).reshape(B * H, D, N)
                if USE_NOISE_MODEL:
                    out = _noisy_matmul_core(
                        a_flat, b_flat, L=2 ** sc_prec,
                        mode="bipolar", per_row_scale=True,
                    ).reshape(B, H, N, D).to(x.dtype)
                else:
                    cfg_av = _get_config(N, sc_prec)
                    out = torch.empty(B * H, N, D, dtype=torch.float32, device=x.device)
                    for i in range(B * H):
                        out[i] = sc_matmul_grouped_bipolar(
                            a_flat[i], b_flat[i], group_a=1, group_b=1,
                            mode="bipolar", sc_prec=sc_prec, config=cfg_av,
                        )
                    out = out.to(x.dtype).reshape(B, H, N, D)
        else:
            out = attn @ v

        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    return forward


def _mm_one_bucket(x_sub: torch.Tensor, w_sub: torch.Tensor,
                   stoc_len: int, sc_prec: int, mode: str) -> torch.Tensor:
    """One SC matmul call at a chosen ``stoc_len``. Picks the noise-surrogate
    backend when ``USE_NOISE_MODEL`` is on, otherwise the Triton kernel.
    Both accept ``stoc_len`` as a kwarg.
    """
    if USE_NOISE_MODEL:
        return noisy_sc_matmul_grouped(
            x_sub, w_sub, group_a=1, group_b=1,
            mode=mode, sc_prec=sc_prec, stoc_len=stoc_len,
        )
    cfg = _get_config(x_sub.shape[-1], sc_prec)
    return sc_matmul_grouped_bipolar(
        x_sub, w_sub, group_a=1, group_b=1,
        mode=mode, sc_prec=sc_prec, config=cfg, stoc_len=stoc_len,
    )


def _sc_linear_mp(x: torch.Tensor, weight: torch.Tensor, bias,
                  sc_prec: int, mode: str,
                  range_entries: list[DispatchEntry] | None = None,
                  mp_cfg: MPConfig | None = None) -> torch.Tensor:
    """MP-aware SC linear: ``y = x @ weight^T + bias``.

    * ``mp_cfg``       → per-input-row stoc_len buckets (dynamic each forward).
    * ``range_entries``→ per-weight-output-group stoc_len buckets (static).
    * Both            → per (row, group) effective stoc_len = min(row_sl, weight_sl).
    * Neither         → uniform, equivalent to ``_sc_linear`` (sans cfg_override).
    """
    orig_shape = x.shape
    D_in = orig_shape[-1]
    x_flat = x.reshape(-1, D_in).float()
    w = weight.float()
    M = x_flat.shape[0]
    D_out = w.shape[0]

    if range_entries is None and mp_cfg is None:
        y = _mm_one_bucket(x_flat, w, 2 ** sc_prec, sc_prec, mode)
        if bias is not None:
            y = y + bias.float()
        return y.to(x.dtype).reshape(*orig_shape[:-1], D_out)

    # Per-row stoc_len from dynamic MP (or a uniform sentinel otherwise).
    if mp_cfg is not None:
        assignment = classify_input_rows(x_flat, mp_cfg)
        row_stoc_lens = torch.empty(M, dtype=torch.long, device=x.device)
        for sl, rows in assignment.level_row_indices.items():
            if rows.numel():
                row_stoc_lens[rows] = int(sl)
    else:
        row_stoc_lens = torch.full((M,), 2 ** sc_prec,
                                   dtype=torch.long, device=x.device)

    result = torch.zeros(M, D_out, dtype=torch.float32, device=x.device)

    if range_entries is None:
        # Dynamic MP only: full-width weight, bucket by row stoc_len.
        for sl in torch.unique(row_stoc_lens).tolist():
            if sl <= 0:
                continue
            rows = torch.nonzero(row_stoc_lens == sl, as_tuple=False).squeeze(-1)
            if rows.numel() == 0:
                continue
            x_sub = x_flat.index_select(0, rows).contiguous()
            sp = sc_prec_for_stoc_len(int(sl))
            sub = _mm_one_bucket(x_sub, w, int(sl), sp, mode)
            result.index_copy_(0, rows, sub)
    else:
        # Range dispatch, optionally combined with dynamic MP.
        for entry in range_entries:
            if entry.stoc_len <= 0:
                continue
            weight_sl = torch.tensor(entry.stoc_len, dtype=torch.long,
                                     device=x.device)
            eff = torch.minimum(row_stoc_lens, weight_sl)
            for sl in torch.unique(eff).tolist():
                if sl <= 0:
                    continue
                rows = torch.nonzero(eff == sl, as_tuple=False).squeeze(-1)
                if rows.numel() == 0:
                    continue
                x_sub = x_flat.index_select(0, rows).contiguous()
                sp = sc_prec_for_stoc_len(int(sl))
                sub = _mm_one_bucket(x_sub, entry.weight, int(sl), sp, mode)
                result[rows.unsqueeze(1), entry.out_indices.unsqueeze(0)] = sub

    if bias is not None:
        result = result + bias.float()
    return result.to(x.dtype).reshape(*orig_shape[:-1], D_out)


class SCLinear(nn.Module):
    """Drop-in replacement for nn.Linear that runs matmul via SC.

    mode selects symmetric (bipolar) or asymmetric zero-point (unipolar) quant.

    Optional mixed-precision knobs (ported from scmp_llm):
      * ``mp_cfg`` — per-input-row MPConfig.
      * ``range_mp_cfg`` — per-weight-row-group RangeMPConfig (static
        dispatch built once from the wrapped weight).
      * ``range_mp_group_size`` — rows per group (0 / >=out_features ⇒
        per-tensor).
      * ``operator`` — name passed to RangeMPConfig for per-op thresholds.
    """

    def __init__(self, linear: nn.Linear, sc_prec: int, mode: str = "bipolar",
                 cfg_override: dict = None,
                 mp_cfg: MPConfig | None = None,
                 range_mp_cfg: RangeMPConfig | None = None,
                 range_mp_group_size: int = 0,
                 operator: str | None = None):
        super().__init__()
        self.sc_prec = sc_prec
        self.mode = mode
        self.weight = linear.weight
        self.bias = linear.bias
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        # Stored as a plain dict (not a buffer): used to pick an alternative
        # Sobol scrambling for the comp's SC matmul, baked at calibration time.
        self.cfg_override = cfg_override
        self.mp_cfg = mp_cfg
        self.operator = operator
        if range_mp_cfg is not None:
            self.range_entries = build_range_dispatch(
                self.weight.detach(), range_mp_cfg,
                operator=operator, group_size=range_mp_group_size,
            )
        else:
            self.range_entries = None

    def forward(self, x):
        if self.range_entries is None and self.mp_cfg is None:
            # Uniform path — keep cfg_override support (residual compensator).
            return _sc_linear(x, self.weight, self.bias, self.sc_prec,
                              mode=self.mode, cfg_override=self.cfg_override)
        return _sc_linear_mp(
            x, self.weight, self.bias, self.sc_prec, self.mode,
            range_entries=self.range_entries, mp_cfg=self.mp_cfg,
        )


# Canonical op names for the sc_ops vector API.
SC_OP_NAMES = ("mlp_fc1", "mlp_fc2", "qk", "av", "proj")


def _discover_blocks(model):
    """Return the list of transformer blocks in a DINOv2-style model."""
    if hasattr(model, "backbone") and hasattr(model.backbone, "blocks"):
        return list(model.backbone.blocks)
    if hasattr(model, "blocks"):
        return list(model.blocks)
    raise RuntimeError("Could not locate transformer blocks on model.")


def normalize_sc_ops_per_block(spec, n_blocks):
    """Normalize a per-block-per-op spec into a complete dict.

    Accepts ``spec`` as ``dict[op_name, list[int]]`` where each list has
    length ``n_blocks`` (or length 1 — broadcast — or a scalar 0/1). Missing
    op keys default to all-zero. Returns a dict with every key in
    ``SC_OP_NAMES`` mapped to a length-``n_blocks`` tuple of 0/1 ints.
    """
    if spec is None:
        return {k: tuple([0] * n_blocks) for k in SC_OP_NAMES}
    if not isinstance(spec, dict):
        raise TypeError("sc_ops_per_block must be a dict[op_name, list]")
    out = {}
    for k in SC_OP_NAMES:
        v = spec.get(k, 0)
        if isinstance(v, (int, bool)):
            out[k] = tuple([int(bool(v))] * n_blocks)
            continue
        v = list(v)
        if len(v) == 1:
            out[k] = tuple([int(bool(v[0]))] * n_blocks)
        elif len(v) == n_blocks:
            out[k] = tuple(int(bool(x)) for x in v)
        else:
            raise ValueError(
                f"sc_ops_per_block[{k!r}] length {len(v)} != n_blocks {n_blocks}")
    extra = set(spec) - set(SC_OP_NAMES)
    if extra:
        raise ValueError(f"unknown op names in sc_ops_per_block: {sorted(extra)}")
    return out


def parse_sc_ops(spec):
    """Normalize an ``sc_ops`` spec into a set of canonical op names.

    Accepts any of:
      * a set/list/tuple of names drawn from SC_OP_NAMES
      * a 5-element 0/1 vector in SC_OP_NAMES order
      * a comma-separated string of names (e.g. "mlp_fc1,av")
    Returns a set[str].
    """
    if spec is None:
        return set()
    if isinstance(spec, str):
        items = [s.strip() for s in spec.split(",") if s.strip()]
        return parse_sc_ops(items)
    items = list(spec)
    if len(items) == len(SC_OP_NAMES) and all(isinstance(x, (int, bool)) for x in items):
        return {n for n, v in zip(SC_OP_NAMES, items) if v}
    out = set()
    for x in items:
        if x not in SC_OP_NAMES:
            raise ValueError(f"unknown sc op: {x!r} (valid: {SC_OP_NAMES})")
        out.add(x)
    return out


LINEAR_OP_NAMES = ("mlp_fc1", "mlp_fc2", "qkv_proj", "out_proj")


def _normalize_linear_mp_spec(spec):
    """Normalize a linear_mp_spec dict into {op: (mp_cfg, range_cfg, gsize)}.

    Each key must be one of ``LINEAR_OP_NAMES`` or the alias ``"proj"``
    (which expands to both ``qkv_proj`` and ``out_proj``). Each value is
    either an ``MPConfig``, a ``RangeMPConfig``, or a dict with any of::

        {"fixed": MPConfig, "range": RangeMPConfig, "range_group_size": int}
    """
    out: dict[str, tuple] = {}
    if not spec:
        return out
    for k, v in spec.items():
        if k == "proj":
            targets = ("qkv_proj", "out_proj")
        elif k in LINEAR_OP_NAMES:
            targets = (k,)
        else:
            raise ValueError(
                f"unknown linear_mp_spec key: {k!r}; "
                f"expected one of {LINEAR_OP_NAMES + ('proj',)}")
        if isinstance(v, MPConfig):
            entry = (v, None, 0)
        elif isinstance(v, RangeMPConfig):
            entry = (None, v, 0)
        elif isinstance(v, dict):
            entry = (
                v.get("fixed"),
                v.get("range"),
                int(v.get("range_group_size", 0)),
            )
        else:
            raise TypeError(
                f"linear_mp_spec[{k!r}] must be MPConfig, RangeMPConfig, "
                f"or dict; got {type(v).__name__}")
        for op in targets:
            out[op] = entry
    return out


def _mp_kwargs(linear_mp, op_name: str) -> dict:
    """Build the MP kwargs for SCLinear from a normalized spec dict."""
    if not linear_mp or op_name not in linear_mp:
        return {}
    mp_cfg, range_cfg, gsize = linear_mp[op_name]
    return dict(mp_cfg=mp_cfg, range_mp_cfg=range_cfg,
                range_mp_group_size=gsize, operator=op_name)


def patch_model(model, sc_prec: int = 8,
                sc_qk: bool = True, sc_av: bool = False, sc_mlp: bool = False,
                sc_qkv_proj: bool = False, sc_out_proj: bool = False,
                mlp_skip_last_k: int = 0,
                mlp_skip_first_k: int = 0,
                sc_mlp_mode: str = "bipolar",
                sc_proj_mode: str = "bipolar",
                mlp_fc_mask: str = "both",  # "both" | "fc1" | "fc2"
                mlp_sc_spec=None,           # list[tuple[int, str]] | None
                sc_ops=None,                # set/vec/str over SC_OP_NAMES
                sc_ops_per_block=None,      # dict[op_name, list[0/1]] length n_blocks
                linear_mp_spec=None,        # dict[op_name, MPConfig|RangeMPConfig|dict]
                attn_mp_spec=None):         # dict[{"qk","av"}, MPConfig] per-head / per-row MP
    """Apply SC patches to a DINOv2 ViT model in-place.

    Priority: ``sc_ops_per_block`` > ``sc_ops`` > legacy per-op bools.

    ``sc_ops`` (when given) takes priority over per-op bool flags. It accepts
    any form understood by ``parse_sc_ops`` — a set of names, a 5-vector
    ``[mlp_fc1, mlp_fc2, qk, av, proj]`` of 0/1, or a comma-separated string.
    The selected ops are turned on uniformly across all 24 blocks (no block-
    level granularity); "proj" enables both qkv_proj and out_proj together.

    ``mlp_sc_spec`` (if given) is an explicit list of
    ``(block_idx, "fc1"|"fc2")`` pairs naming exactly which MLP linears go SC.
    When set, it overrides ``mlp_skip_first_k`` / ``mlp_skip_last_k`` /
    ``mlp_fc_mask`` / ``sc_ops`` for MLP selection.

    Returns a dict counting patched modules.
    """
    # sc_ops dispatches to per-op bools (unless mlp_sc_spec is set for MLP).
    if sc_ops is not None:
        ops = parse_sc_ops(sc_ops)
        sc_qk = "qk" in ops
        sc_av = "av" in ops
        sc_qkv_proj = "proj" in ops
        sc_out_proj = "proj" in ops
        if mlp_sc_spec is None:
            want_fc1 = "mlp_fc1" in ops
            want_fc2 = "mlp_fc2" in ops
            sc_mlp = want_fc1 or want_fc2
            if want_fc1 and want_fc2:
                mlp_fc_mask = "both"
            elif want_fc1:
                mlp_fc_mask = "fc1"
            elif want_fc2:
                mlp_fc_mask = "fc2"
    stats = {"attn": 0, "mlp_fc1": 0, "mlp_fc2": 0, "qkv_proj": 0, "out_proj": 0}
    linear_mp = _normalize_linear_mp_spec(linear_mp_spec)
    qk_mp_cfg = (attn_mp_spec or {}).get("qk")
    av_mp_cfg = (attn_mp_spec or {}).get("av")
    if qk_mp_cfg is not None and not isinstance(qk_mp_cfg, MPConfig):
        raise TypeError(f"attn_mp_spec['qk'] must be MPConfig, got {type(qk_mp_cfg).__name__}")
    if av_mp_cfg is not None and not isinstance(av_mp_cfg, MPConfig):
        raise TypeError(f"attn_mp_spec['av'] must be MPConfig, got {type(av_mp_cfg).__name__}")

    # ---- highest-priority path: per-block per-op vectors ----
    if sc_ops_per_block is not None:
        blocks = _discover_blocks(model)
        pbm = normalize_sc_ops_per_block(sc_ops_per_block, len(blocks))
        for i, blk in enumerate(blocks):
            # Locate attention submodule (the one with qkv/proj/num_heads)
            attn_mod = None
            for m in blk.modules():
                if (hasattr(m, "qkv") and hasattr(m, "proj")
                        and hasattr(m, "num_heads")):
                    attn_mod = m
                    break
            q_on, a_on, p_on = pbm["qk"][i], pbm["av"][i], pbm["proj"][i]
            if attn_mod is not None and (q_on or a_on):
                fwd = make_sc_attention_forward(
                    sc_prec, sc_av=bool(a_on), sc_qk=bool(q_on),
                    qk_mp_cfg=qk_mp_cfg if q_on else None,
                    av_mp_cfg=av_mp_cfg if a_on else None)
                attn_mod.forward = fwd.__get__(attn_mod, type(attn_mod))
                stats["attn"] += 1
            if attn_mod is not None and p_on:
                if isinstance(attn_mod.qkv, nn.Linear):
                    attn_mod.qkv = SCLinear(attn_mod.qkv, sc_prec, mode=sc_proj_mode,
                                            **_mp_kwargs(linear_mp, "qkv_proj"))
                    stats["qkv_proj"] += 1
                if isinstance(attn_mod.proj, nn.Linear):
                    attn_mod.proj = SCLinear(attn_mod.proj, sc_prec, mode=sc_proj_mode,
                                             **_mp_kwargs(linear_mp, "out_proj"))
                    stats["out_proj"] += 1
            mlp = getattr(blk, "mlp", None)
            if mlp is not None:
                if pbm["mlp_fc1"][i] and isinstance(getattr(mlp, "fc1", None), nn.Linear):
                    mlp.fc1 = SCLinear(mlp.fc1, sc_prec, mode=sc_mlp_mode,
                                       **_mp_kwargs(linear_mp, "mlp_fc1"))
                    stats["mlp_fc1"] += 1
                if pbm["mlp_fc2"][i] and isinstance(getattr(mlp, "fc2", None), nn.Linear):
                    mlp.fc2 = SCLinear(mlp.fc2, sc_prec, mode=sc_mlp_mode,
                                       **_mp_kwargs(linear_mp, "mlp_fc2"))
                    stats["mlp_fc2"] += 1
        return stats

    if sc_qk or sc_av:
        fwd = make_sc_attention_forward(
            sc_prec, sc_av=sc_av, sc_qk=sc_qk,
            qk_mp_cfg=qk_mp_cfg if sc_qk else None,
            av_mp_cfg=av_mp_cfg if sc_av else None)
        for m in model.modules():
            if hasattr(m, "qkv") and hasattr(m, "proj") and hasattr(m, "num_heads"):
                m.forward = fwd.__get__(m, type(m))
                stats["attn"] += 1

    if sc_qkv_proj or sc_out_proj:
        for m in model.modules():
            if hasattr(m, "qkv") and hasattr(m, "proj") and hasattr(m, "num_heads"):
                if sc_qkv_proj and isinstance(m.qkv, nn.Linear):
                    m.qkv = SCLinear(m.qkv, sc_prec, mode=sc_proj_mode,
                                     **_mp_kwargs(linear_mp, "qkv_proj"))
                    stats["qkv_proj"] += 1
                if sc_out_proj and isinstance(m.proj, nn.Linear):
                    m.proj = SCLinear(m.proj, sc_prec, mode=sc_proj_mode,
                                      **_mp_kwargs(linear_mp, "out_proj"))
                    stats["out_proj"] += 1

    if sc_mlp:
        blocks = None
        if hasattr(model, "backbone") and hasattr(model.backbone, "blocks"):
            blocks = list(model.backbone.blocks)
        elif hasattr(model, "blocks"):
            blocks = list(model.blocks)

        if mlp_sc_spec is not None:
            # Explicit list mode. Build per-block set of linears to patch.
            selected = {}  # block_idx -> set of "fc1"/"fc2"
            for item in mlp_sc_spec:
                bi, fc = item
                assert fc in ("fc1", "fc2"), f"bad linear name: {fc}"
                selected.setdefault(int(bi), set()).add(fc)
            if blocks is None:
                raise RuntimeError("mlp_sc_spec requires discoverable blocks")
            for bi, blk in enumerate(blocks):
                if not hasattr(blk, "mlp"):
                    continue
                mlp = blk.mlp
                want = selected.get(bi, set())
                if "fc1" in want and isinstance(getattr(mlp, "fc1", None), nn.Linear):
                    mlp.fc1 = SCLinear(mlp.fc1, sc_prec, mode=sc_mlp_mode,
                                       **_mp_kwargs(linear_mp, "mlp_fc1"))
                    stats["mlp_fc1"] += 1
                if "fc2" in want and isinstance(getattr(mlp, "fc2", None), nn.Linear):
                    mlp.fc2 = SCLinear(mlp.fc2, sc_prec, mode=sc_mlp_mode,
                                       **_mp_kwargs(linear_mp, "mlp_fc2"))
                    stats["mlp_fc2"] += 1
        else:
            # Legacy first-k / last-k block skipping path.
            skip_mlps = set()
            if blocks is not None and mlp_skip_last_k > 0:
                for blk in blocks[-mlp_skip_last_k:]:
                    if hasattr(blk, "mlp"):
                        skip_mlps.add(id(blk.mlp))
            if blocks is not None and mlp_skip_first_k > 0:
                for blk in blocks[:mlp_skip_first_k]:
                    if hasattr(blk, "mlp"):
                        skip_mlps.add(id(blk.mlp))
            for m in model.modules():
                if m.__class__.__name__ == "Mlp" and hasattr(m, "fc1") and hasattr(m, "fc2"):
                    if id(m) in skip_mlps:
                        continue
                    if mlp_fc_mask in ("both", "fc1") and isinstance(m.fc1, nn.Linear):
                        m.fc1 = SCLinear(m.fc1, sc_prec, mode=sc_mlp_mode,
                                         **_mp_kwargs(linear_mp, "mlp_fc1"))
                        stats["mlp_fc1"] += 1
                    if mlp_fc_mask in ("both", "fc2") and isinstance(m.fc2, nn.Linear):
                        m.fc2 = SCLinear(m.fc2, sc_prec, mode=sc_mlp_mode,
                                         **_mp_kwargs(linear_mp, "mlp_fc2"))
                        stats["mlp_fc2"] += 1

    return stats


# Back-compat alias
def patch_model_attention(model, sc_prec: int = 8):
    return patch_model(model, sc_prec=sc_prec, sc_qk=True)["attn"]
