"""Smoke-probe for --stoc_len fast-path correctness on EVA-ViTDet.

Goal: confirm that passing stoc_len directly (no mp_cfg wrapper) routes every
SC module through the uniform fast path:
  - SCLinear:  mp_cfg is None AND adaptive_mp_cfg is None AND range_entries is None
  - SCMatMul:  mp_cfg is None AND adaptive_mp_cfg is None
  - .stoc_len attribute matches what we passed.

Then run a small inference batch and report wall time.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
for _p in (_HERE, _REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from eval_common import load_model_and_loader
from sc_patch import sc_patch_eva, normalize_sc_ops_per_block
from sc_integration.sc_linear import SCLinear  # noqa: F401
import json
from sc_patch.sc_matmul import SCMatMul

p = argparse.ArgumentParser()
p.add_argument("--n-eval",   type=int, default=10)
p.add_argument("--sc_prec",  type=int, default=7)
p.add_argument("--stoc_len", type=int, default=96)
p.add_argument("--size",     type=int, default=1024)
p.add_argument("--sched",    default="sensitivity/skip/skip_worst_30_len96.json")
args = p.parse_args()

print(f"[probe] sc_prec={args.sc_prec}  stoc_len={args.stoc_len}  size={args.size}  n={args.n_eval}")
model, loader, evaluator, _ = load_model_and_loader(
    args.n_eval, size=args.size,
    use_soft_nms=True, interp_type="beit")
n_blocks = len(list(model.backbone.net.blocks))
sched = normalize_sc_ops_per_block(json.load(open(args.sched)), n_blocks)
sc_patch_eva(model, sc_prec=args.sc_prec,
             sc_ops_per_block=sched,
             mlp_mode="bipolar", proj_mode="bipolar",
             linear_mp_spec=None, attn_mp_spec=None,
             mlp_chunk_d=0, stoc_len=args.stoc_len)

# Inspect first SCLinear and SCMatMul to confirm fast path.
def first_of(model, cls):
    for m in model.modules():
        if isinstance(m, cls):
            return m
    return None

scl = first_of(model, SCLinear)
scm = first_of(model, SCMatMul)

def report(name, m):
    if m is None:
        print(f"  [{name}] none found")
        return
    mp = getattr(m, "mp_cfg", "?")
    amp = getattr(m, "adaptive_mp_cfg", "?")
    re = getattr(m, "range_entries", "?")
    sl = getattr(m, "stoc_len", "?")
    sp = getattr(m, "sc_prec", "?")
    print(f"  [{name}] sc_prec={sp}  stoc_len={sl}  mp_cfg={mp}  adaptive_mp_cfg={amp}  range_entries={re}")
    fast = (mp is None) and (amp is None) and (re is None or re is False)
    print(f"  [{name}] fast_path = {fast}")

print("[probe] post-patch attribute dump:")
report("SCLinear[0]", scl)
report("SCMatMul[0]", scm)

print(f"[probe] running inference on {args.n_eval} imgs...")
t0 = time.time()
with torch.no_grad():
    for i, batch in enumerate(loader):
        if i >= args.n_eval:
            break
        _ = model(batch)
        torch.cuda.synchronize()
dt = time.time() - t0
print(f"[probe] done {dt:.2f}s  ({dt/args.n_eval:.3f}s/img)")
