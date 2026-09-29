"""Summarize an SC_MP_TRACE summary JSON: per-op call/block/MAC coverage.

    SC_MP_TRACE=out.json python cls/eval.py ...
    python tests/summarize_trace.py out.json

Use it to prove which ops actually issued SC matmuls — the patch-count dict
printed by eval.py lumps qk and av together under "attn", and a silently
FP-fallen-back op leaves no other trace.
"""
import collections
import json
import sys


def main(path):
    with open(path) as f:
        payload = json.load(f)
    groups = payload.get("groups", [])
    per = collections.defaultdict(lambda: {"calls": 0, "macs": 0,
                                           "blocks": set(), "sl": set()})
    for r in groups:
        e = per[r.get("op")]
        e["calls"] += r.get("calls", 0)
        e["macs"] += r.get("macs", 0)
        e["blocks"].add(r.get("block"))
        e["sl"].add(r.get("stoc_len"))

    hdr = f'{"op":<12}{"calls":>8}{"blocks":>8}{"stoc_len":>12}{"MACs":>20}'
    print(hdr)
    print("-" * len(hdr))
    total = 0
    for op in sorted(per, key=lambda k: (k is None, k)):
        e = per[op]
        total += e["macs"]
        sl = ",".join(str(s) for s in sorted(x for x in e["sl"] if x is not None))
        print(f'{str(op):<12}{e["calls"]:>8}{len(e["blocks"]):>8}'
              f'{sl:>12}{e["macs"]:>20,}')
    print("-" * len(hdr))
    print(f'{"TOTAL":<12}{"":>8}{"":>8}{"":>12}{total:>20,}')

    # Realized MAC-weighted average stream length — the honest iso-compute
    # check. A calibration table only PREDICTS this; the deployed dispatch is
    # what actually spends cycles, so an MP cell may only be compared against a
    # uniform cell whose stoc_len matches THIS number, not the nominal budget.
    num = sum(r.get("macs", 0) * (r.get("stoc_len") or 0) for r in groups)
    if total:
        print(f"\nrealized MAC-weighted avg stoc_len = {num / total:.2f}"
              f"   (effective levels {2 * num / total:.1f} if halve=1)")
        per_op_avg = {}
        for op in sorted(per):
            n = sum(r.get("macs", 0) * (r.get("stoc_len") or 0)
                    for r in groups if r.get("op") == op)
            d = per[op]["macs"]
            if d:
                per_op_avg[op] = n / d
        print("per-op realized avg stoc_len:")
        for op, v in per_op_avg.items():
            print(f"    {str(op):<12}{v:>8.2f}   "
                  f"({100 * per[op]['macs'] / total:.1f}% of MACs)")

    cov = payload.get("coverage")
    if cov:
        print(f"\ncoverage note: {cov}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "trace.json")
