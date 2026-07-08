#!/usr/bin/env python3
"""Rank (op, block) entries from a per-operator sensitivity sweep, worst first.

Usage:
    python rank_worst.py <json_path> [--top N]

Sort key: dtop1 ascending (most negative = worst), KL descending as tiebreaker.
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", type=Path)
    ap.add_argument("--top", type=int, default=50)
    args = ap.parse_args()

    d = json.loads(args.json_path.read_text())
    g = list(d["grid"])
    g.sort(key=lambda r: (r["dtop1"], -r["kl"]))

    header = f'{"rank":>4}  {"op":<10} {"blk":>3}  {"top1":>5}  {"dtop1":>7}  {"flip":>5}  {"l2":>7}  {"kl":>9}'
    print(header)
    print("-" * len(header))
    for i, r in enumerate(g[: args.top], 1):
        print(
            f'{i:>4}  {r["op"]:<10} {r["block"]:>3}  '
            f'{r["top1"]:.2f}  {r["dtop1"]:>+7.3f}  {r["flip"]:.2f}  '
            f'{r["l2"]:>7.2f}  {r["kl"]:>9.4f}'
        )


if __name__ == "__main__":
    main()
