#!/usr/bin/env python3
"""Refuse to report across cells that did not score the same images.

    python cls/check_eval_sets.py results/final10k          # exit 1 on mismatch

Every paired test here compares per-image hit vectors POSITIONALLY. Two cells
evaluated on different images -- or the same images in a different order -- give
a delta that is silently meaningless rather than an error, and the failure looks
exactly like a real effect. The eval list is easy to perturb by accident:

  * ``--exclude_calib_images`` edits the keep-list BEFORE the shuffle, so one
    cell using it and another not shifts every index that follows;
  * ``--skip_images`` opens a different window on purpose (config search);
  * a different ``--seed`` reshuffles everything;
  * a different ``--max_images`` truncates to a different length.

``cls/eval.py`` records ``eval_set_sha`` (sha256 of the exact index list, in
loader order) in every result JSON. This script groups a directory by that
fingerprint and fails if there is more than one group, naming the odd cells.

Cells written before the fingerprint existed report ``legacy`` and are listed
separately: they are not proven wrong, only unverifiable.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    groups: dict = defaultdict(list)
    for p in sorted(args.dir.glob("*.json")):
        if p.name.endswith("_trace.json") or p.name == "verdict.json":
            continue
        try:
            j = json.loads(p.read_text())
        except Exception:
            continue
        if "top1" not in j:
            continue
        sha = j.get("eval_set_sha", "legacy")
        groups[(sha, j.get("n"), j.get("eval_seed"), j.get("eval_skip"))].append(p.stem)

    if not groups:
        print(f"[evalset] no result cells in {args.dir}")
        sys.exit(0)

    real = {k: v for k, v in groups.items() if k[0] != "legacy"}
    legacy = {k: v for k, v in groups.items() if k[0] == "legacy"}

    for (sha, n, seed, skip), cells in sorted(real.items(), key=lambda kv: -len(kv[1])):
        print(f"[evalset] {sha}  n={n} seed={seed} skip={skip}  "
              f"{len(cells)} cells")
        if not args.quiet:
            print(f"[evalset]   {', '.join(sorted(cells))}")
    for _, cells in legacy.items():
        print(f"[evalset] legacy (no fingerprint recorded): {len(cells)} cells")
        if not args.quiet:
            print(f"[evalset]   {', '.join(sorted(cells))}")

    if len(real) > 1:
        print(f"\n[evalset] FAIL: {len(real)} distinct eval sets in one directory.")
        print("[evalset] Cells from different groups must NOT be differenced.")
        print("[evalset] A config-search window (skip>0) belongs in its own dir.")
        sys.exit(1)

    print("\n[evalset] OK: every fingerprinted cell scored the same images.")
    sys.exit(0)


if __name__ == "__main__":
    main()
