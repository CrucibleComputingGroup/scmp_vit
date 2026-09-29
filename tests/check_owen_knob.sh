#!/usr/bin/env bash
# Is SC_OWEN_MODE actually honored? Dump a deterministic sc_matmul checksum
# under each mode; identical checksums across modes => the knob is dead.
set -u
cd /nfs/turbo/coe-nbleier/yjrcs/HPCA/vit_sc || exit 1

for m in bitrev off random; do
  for sl in 256 16; do
    SC_OWEN_MODE=$m SL=$sl python - <<'PY' 2>&1 | grep -viE "futurewarning|import pynvml"
import os, sys, torch
sys.path.insert(0, "kernels")
from scmp_kernels import sc_matmul
torch.manual_seed(0)
a = torch.randn(64, 512, device="cuda")
b = torch.randn(128, 512, device="cuda")
sl = int(os.environ["SL"])
out = sc_matmul(a, b, granularity="per_row", mode="bipolar", sc_prec=8, stoc_len=sl)
ref = a @ b.T
print(f'  mode={os.environ["SC_OWEN_MODE"]:<7} stoc_len={sl:<4} '
      f'sum={out.sum().item():+.6f}  rel_err={((out-ref).norm()/ref.norm()).item():.6f}')
PY
  done
done
