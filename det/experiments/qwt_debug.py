"""Debug QwT driver: measure per-block residual RMSE on calib set both
during fit AND at eval-time application to verify no training/eval mismatch.
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch

_HERE = Path(__file__).resolve().parent
_DET = _HERE.parent
_REPO = _DET.parent
_QWT_LIB = _REPO / "third_party" / "QwT-SC" / "QwT-vit-sc"
for p in (_DET, _QWT_LIB):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eval_common import load_model_and_loader
from sc_patch import sc_patch_eva
from qwt_sc import calibrate_qwt
from qwt_sc.compensation import CompensationBlock


class CalibLoader:
    def __init__(self, d2_loader, model, device, max_items):
        self._l = d2_loader; self._pp = model; self._d = device; self._m = max_items
    def __iter__(self):
        n = 0
        for bi in self._l:
            t = self._pp.preprocess_image(bi).tensor.to(self._d)
            yield (t,)
            n += t.size(0)
            if n >= self._m: return


@torch.no_grad()
def forward_blocks(blocks, x, chunk=2):
    outs = []
    for s in range(0, x.size(0), chunk):
        xb = x[s:s+chunk]
        for blk in blocks:
            xb = blk(xb)
        outs.append(xb.float())
    return torch.cat(outs, dim=0)


@torch.no_grad()
def forward_one_block(block, x, chunk=2):
    outs = []
    for s in range(0, x.size(0), chunk):
        xb = x[s:s+chunk]
        outs.append(block(xb).float())
    return torch.cat(outs, dim=0)


def main():
    device = torch.device("cuda")
    SC_PREC = 7
    SC_OPS = "qk,av,proj"
    N_CALIB = 8

    print("[load] FP")
    model_fp, loader_fp, _, _ = load_model_and_loader(N_CALIB)
    model_fp.eval()
    print("[load] SC")
    model_sc, _, _, _ = load_model_and_loader(N_CALIB)
    model_sc.eval()
    sc_patch_eva(model_sc, sc_prec=SC_PREC,
                 sc_ops_per_block={"qk": 1, "av": 1, "qkv_proj": 1, "out_proj": 1})

    blocks_fp = list(model_fp.backbone.net.blocks)
    blocks_sc = model_sc.backbone.net.blocks

    # ===== capture X_0 from SC pass =====
    captured = []
    def hook(_m, args):
        captured.append(args[0].detach().float().cpu())
    h = blocks_sc[0].register_forward_pre_hook(hook)
    for bi in loader_fp:
        imgs = model_sc.preprocess_image(bi).tensor.to(device)
        model_sc.backbone.net(imgs)
        if sum(x.size(0) for x in captured) >= N_CALIB: break
    h.remove()
    X0 = torch.cat(captured, dim=0)[:N_CALIB]
    print(f"[capture] X0 shape={tuple(X0.shape)} dtype={X0.dtype}")

    # ===== Run QwT calibration via library =====
    loader = CalibLoader(loader_fp, model_sc, device, N_CALIB)
    report = calibrate_qwt(
        model_fp=model_fp.backbone.net,
        model_sc=model_sc.backbone.net,
        blocks_fp=blocks_fp,
        blocks_sc_container=blocks_sc,
        calib_loader=loader,
        device=device, n_calib=N_CALIB, ridge=1e-2, fwd_chunk=2, avg_sc_draws=1,
    )

    # At this point, blocks_sc[i] is CompensationBlock wrapping SC block i.
    # Propagate X0 through (a) FP blocks, (b) raw SC blocks (unwrapped), (c) comp-wrapped SC blocks.
    # (b) requires unwrapping.

    print("\n[check] propagating X0 through 3 paths on calib images")
    X0_dev = X0.to(device)

    # (a) FP path
    x = X0_dev
    y_fp = forward_blocks(blocks_fp, x)
    print(f"[fp  path] final ||y|| = {y_fp.norm().item():.4f}")

    # (c) COMP-wrapped SC path (what eval-time model uses)
    x = X0_dev
    y_comp = forward_blocks(list(blocks_sc), x)
    print(f"[comp path] final ||y|| = {y_comp.norm().item():.4f}")

    # (b) RAW SC path (unwrap each CompensationBlock to its inner .block)
    raw_sc = [blk.block if isinstance(blk, CompensationBlock) else blk for blk in blocks_sc]
    x = X0_dev
    y_sc_raw = forward_blocks(raw_sc, x)
    print(f"[raw-sc path] final ||y|| = {y_sc_raw.norm().item():.4f}")

    # Differences
    rmse_raw = (y_fp - y_sc_raw).pow(2).mean().sqrt().item()
    rmse_comp = (y_fp - y_comp).pow(2).mean().sqrt().item()
    print(f"\n[final-block rmse to FP]")
    print(f"  SC raw : {rmse_raw:.4e}")
    print(f"  SC comp: {rmse_comp:.4e}")
    print(f"  ratio  : {rmse_comp/rmse_raw:.4f}")

    # Per-block compounding
    print("\n[per-block rmse at each point]")
    x_fp = x_sc = x_comp = X0_dev
    for i in range(40):
        x_fp = forward_one_block(blocks_fp[i], x_fp)
        x_sc = forward_one_block(raw_sc[i], x_sc)
        x_comp = forward_one_block(blocks_sc[i], x_comp)
        r_raw = (x_fp - x_sc).pow(2).mean().sqrt().item()
        r_cmp = (x_fp - x_comp).pow(2).mean().sqrt().item()
        if i in (0, 1, 5, 10, 20, 30, 35, 38, 39):
            print(f"  blk {i:2d}: raw rmse {r_raw:.4e}  comp rmse {r_cmp:.4e}  "
                  f"ratio {r_cmp/r_raw:.3f}")


if __name__ == "__main__":
    main()
