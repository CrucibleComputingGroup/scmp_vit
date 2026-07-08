"""Generate per-precision skip-worst-20 sc_ops_per_block JSONs.

Reads cls/sensitivity/sensitivity_per_operator_real_sc_p{6,7,8}.json (144 entries
= 6 ops x 24 blocks ranked by L2), drops the 20 worst (op, block) pairs, and
writes a 5-op schedule JSON consumable by eval.py --sc_ops_per_block_json and
qwt_sc_overnight.py's fine_schedule path.

Merges qkv_proj + out_proj -> proj (5-op SC_OP_NAMES). If either sub-op at
block B is in worst-20, proj[B] is dropped.
"""
import argparse
import json
from pathlib import Path

SC_OP_NAMES = ("mlp_fc1", "mlp_fc2", "qk", "av", "proj")
N_BLOCKS = 24


def build(sens_json_path: Path, k: int = 20, n_blocks: int = N_BLOCKS):
    data = json.loads(sens_json_path.read_text())
    rows = sorted(data["grid"], key=lambda r: r["l2"], reverse=True)
    worst = rows[:k]
    spec = {op: [1] * n_blocks for op in SC_OP_NAMES}
    drops = []
    for r in worst:
        op_raw = r["op"]
        bi = int(r["block"])
        key = "proj" if op_raw in ("qkv_proj", "out_proj") else op_raw
        spec[key][bi] = 0
        drops.append({"op_raw": op_raw, "op_merged": key, "block": bi,
                      "l2": r["l2"]})
    n_active = sum(sum(v) for v in spec.values())
    return spec, drops, n_active


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sens_dir", default="cls/sensitivity")
    ap.add_argument("--out_dir", default="cls/sensitivity/skip_worst20")
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args()

    sens_dir = Path(args.sens_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in (6, 7, 8):
        src = sens_dir / f"sensitivity_per_operator_real_sc_p{p}.json"
        spec, drops, n_active = build(src, k=args.k)
        out = out_dir / f"skip_worst{args.k}_p{p}.json"
        out.write_text(json.dumps(spec, indent=2))
        print(f"[p{p}] {src} -> {out}  active={n_active}/120")
        for d in drops:
            print(f"    drop {d['op_raw']:>10s} block={d['block']:>2d} "
                  f"l2={d['l2']:.4f}")


if __name__ == "__main__":
    main()
