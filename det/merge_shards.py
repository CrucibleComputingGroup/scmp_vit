"""Merge sharded run_coco_val.py checkpoints into a single COCO AP report.

Usage::

    python det/merge_shards.py det/results/int7_skip20 \\
        --d2_datasets <COCO_ROOT>

Looks for ``shard_*/checkpoint.pt`` under the given dir, validates that all
shards share the same SC config (sc_prec / skip / num_shards / mode), confirms
that their image_id sets are disjoint and union = full val, then runs a
fresh COCOEvaluator over the merged predictions and writes
``<parent>/metrics.json``.

If shards together cover only a strict subset of val (e.g. some shards still
running), use ``--allow_partial`` to evaluate AP on that subset instead.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
for _p in (_HERE, _REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load_shards(parent: Path):
    """Return list of (shard_dir, state_dict) sorted by shard_id."""
    shards = []
    for d in sorted(parent.glob("shard_*")):
        if not d.is_dir():  # skip shard_*.log etc.
            continue
        ck = d / "checkpoint.pt"
        if not ck.exists():
            print(f"[skip] {d.name}: no checkpoint.pt")
            continue
        state = torch.load(ck, map_location="cpu", weights_only=False)
        shards.append((d, state))
    return shards


def _check_consistent(shards):
    """Verify all shards share the same SC config (sans shard_id/start/n_eval)."""
    invariants = ["sc_prec", "skip_json_sha256", "sc_mlp_mode",
                  "sc_proj_mode", "num_shards"]
    sigs = []
    for d, st in shards:
        cfg = st.get("config", {})
        sig = tuple(cfg.get(k) for k in invariants)
        sigs.append((d.name, sig, cfg))
    base = sigs[0][1]
    bad = [n for n, s, _ in sigs if s != base]
    if bad:
        print("[error] inconsistent SC config across shards:")
        for n, s, c in sigs:
            print(f"    {n}: {dict(zip(invariants, s))}")
        sys.exit(2)
    num_shards_seen = base[invariants.index("num_shards")]
    return num_shards_seen, sigs[0][2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("parent", type=str,
                   help="Dir containing shard_0/, shard_1/, ... subdirs.")
    p.add_argument("--d2_datasets", type=str, default="",
                   help="DETECTRON2_DATASETS root.")
    p.add_argument("--allow_partial", action="store_true",
                   help="Allow merging even if shards don't cover the full "
                        "val set. AP is then computed on the union only.")
    p.add_argument("--out_name", type=str, default="metrics.json",
                   help="Output filename in <parent>/ (default metrics.json).")
    args = p.parse_args()

    parent = Path(args.parent).resolve()
    if not parent.is_dir():
        sys.exit(f"not a dir: {parent}")

    if args.d2_datasets:
        os.environ["DETECTRON2_DATASETS"] = args.d2_datasets
    elif "DETECTRON2_DATASETS" not in os.environ:
        from eval_common import _DEFAULT_D2_DATASETS
        os.environ["DETECTRON2_DATASETS"] = _DEFAULT_D2_DATASETS

    shards = _load_shards(parent)
    if not shards:
        sys.exit(f"no shard_*/checkpoint.pt found under {parent}")

    num_shards_expected, sample_cfg = _check_consistent(shards)
    print(f"[1] {len(shards)} shards loaded "
          f"(num_shards={num_shards_expected})")
    if num_shards_expected and len(shards) != num_shards_expected:
        msg = (f"only {len(shards)}/{num_shards_expected} shards present")
        if not args.allow_partial:
            sys.exit(f"[error] {msg} — pass --allow_partial to merge anyway")
        print(f"[warn] {msg}")

    # --- Merge predictions, dedup by image_id, sanity-check disjointness. ---
    merged = []
    seen = {}  # image_id -> shard_name
    dups = []
    elapsed = 0.0
    for d, st in shards:
        elapsed += float(st.get("elapsed_seconds", 0.0))
        for pred in st.get("predictions", []):
            iid = int(pred["image_id"])
            if iid in seen:
                dups.append((iid, seen[iid], d.name))
                continue
            seen[iid] = d.name
            merged.append(pred)
    if dups:
        print(f"[warn] {len(dups)} duplicate image_ids across shards "
              f"(first kept). Examples:")
        for iid, a, b in dups[:5]:
            print(f"    image_id={iid}: {a} & {b}")

    print(f"[2] merged predictions: {len(merged)} images "
          f"(cumulative compute: {elapsed/60:.1f} min)")

    # --- Coverage check vs full val. ---
    from detectron2.data import DatasetCatalog
    full_items = DatasetCatalog.get("coco_2017_val")
    full_ids = {int(it["image_id"]) for it in full_items}
    covered = set(seen.keys())
    missing = full_ids - covered
    extra = covered - full_ids
    print(f"    coverage: {len(covered)}/{len(full_ids)} val images")
    if extra:
        print(f"[warn] {len(extra)} predicted image_ids not in val (ignored).")
    if missing:
        if not args.allow_partial:
            sys.exit(f"[error] {len(missing)} val images missing. "
                     f"Pass --allow_partial to compute AP on the covered subset.")
        print(f"[warn] {len(missing)} val images missing — AP on covered subset.")

    # --- Build a fresh evaluator and inject predictions. ---
    print("[3] Running COCO evaluation on merged predictions")
    from detectron2.evaluation import COCOEvaluator

    out_dir = parent
    evaluator = COCOEvaluator("coco_2017_val", tasks=("bbox", "segm"),
                              distributed=False, output_dir=str(out_dir),
                              max_dets_per_image=None)
    if missing:
        # Restrict the COCO API to the covered subset so AP is comparable.
        coco = evaluator._coco_api
        coco.imgs = {k: v for k, v in coco.imgs.items() if k in covered}
        coco.anns = {k: v for k, v in coco.anns.items()
                     if v["image_id"] in covered}
        coco.imgToAnns = {k: v for k, v in coco.imgToAnns.items()
                          if k in covered}
        coco.catToImgs = {c: [i for i in imgs if i in covered]
                          for c, imgs in coco.catToImgs.items()}

    evaluator._predictions = merged
    results = evaluator.evaluate()

    # --- Write merged metrics.json. ---
    out = {
        "shard_parent": str(parent),
        "num_shards_present": len(shards),
        "num_shards_expected": num_shards_expected,
        "n_predictions": len(merged),
        "n_val_total": len(full_ids),
        "n_covered": len(covered),
        "n_missing": len(missing),
        "duplicates": len(dups),
        "elapsed_seconds_sum": elapsed,
        "config": {k: sample_cfg.get(k) for k in
                   ("sc_prec", "skip_json", "skip_json_sha256",
                    "sc_mlp_mode", "sc_proj_mode", "num_shards")},
        **results,
    }
    out_path = parent / args.out_name
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    for task, m in results.items():
        print(f"    [{task}] AP={m.get('AP', 0):.2f}  "
              f"AP50={m.get('AP50', 0):.2f}  AP75={m.get('AP75', 0):.2f}")
    print(f"[OK] merged metrics -> {out_path}")


if __name__ == "__main__":
    main()
