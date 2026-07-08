"""Visualize COCO detection results on images.

Usage:
    python visualize.py --results_dir results/sc_p8 --n_images 5
    python visualize.py --results_dir results/fp --n_images 10 --score_thr 0.5
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# COCO 80-class names (index 1..90, with gaps)
_COCO_NAMES = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep", 21: "cow",
    22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe", 27: "backpack",
    28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase", 34: "frisbee",
    35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite",
    39: "baseball bat", 40: "baseball glove", 41: "skateboard", 42: "surfboard",
    43: "tennis racket", 44: "bottle", 46: "wine glass", 47: "cup", 48: "fork",
    49: "knife", 50: "spoon", 51: "bowl", 52: "banana", 53: "apple",
    54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog",
    59: "pizza", 60: "donut", 61: "cake", 62: "chair", 63: "couch",
    64: "potted plant", 65: "bed", 67: "dining table", 70: "toilet", 72: "tv",
    73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard", 77: "cell phone",
    78: "microwave", 79: "oven", 80: "toaster", 81: "sink", 82: "refrigerator",
    84: "book", 85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear",
    89: "hair drier", 90: "toothbrush",
}


def _color_for_cat(cat_id: int):
    """Deterministic color per category."""
    rng = np.random.RandomState(cat_id * 7 + 13)
    return tuple(int(c) for c in rng.randint(60, 255, 3))


def draw_detections(img, dets, score_thr: float = 0.3):
    """Draw bounding boxes + labels on image."""
    for det in dets:
        score = det["score"]
        if score < score_thr:
            continue
        x, y, w, h = det["bbox"]
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        cat_id = det["category_id"]
        label = _COCO_NAMES.get(cat_id, str(cat_id))
        color = _color_for_cat(cat_id)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        txt = f"{label} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(img, txt, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, required=True,
                   help="Directory containing coco_instances_results.json")
    p.add_argument("--coco_root", type=str,
                   default="/scratch/nbleier_owned_root/nbleier_owned1/shared_data/coco/val2017",
                   help="Path to COCO val2017 images.")
    p.add_argument("--n_images", type=int, default=5,
                   help="Max number of images to visualize.")
    p.add_argument("--score_thr", type=float, default=0.3,
                   help="Score threshold for drawing boxes.")
    p.add_argument("--out_dir", type=str, default="",
                   help="Output directory for visualizations (default: results_dir/vis).")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    coco_json = results_dir / "coco_instances_results.json"
    if not coco_json.exists():
        raise FileNotFoundError(f"{coco_json} not found. Run eval first.")

    with open(coco_json) as f:
        dets = json.load(f)

    # Group detections by image_id
    by_image = defaultdict(list)
    for d in dets:
        by_image[d["image_id"]].append(d)

    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "vis"
    out_dir.mkdir(parents=True, exist_ok=True)
    coco_root = Path(args.coco_root)

    image_ids = sorted(by_image.keys())[:args.n_images]
    print(f"Visualizing {len(image_ids)} images (score_thr={args.score_thr})")

    for img_id in image_ids:
        img_name = f"{img_id:012d}.jpg"
        img_path = coco_root / img_name
        if not img_path.exists():
            print(f"  [skip] {img_path} not found")
            continue

        img = cv2.imread(str(img_path))
        img = draw_detections(img, by_image[img_id], args.score_thr)

        n_boxes = sum(1 for d in by_image[img_id] if d["score"] >= args.score_thr)
        out_path = out_dir / img_name
        cv2.imwrite(str(out_path), img)
        print(f"  {img_name}: {n_boxes} boxes -> {out_path}")

    print(f"[OK] Saved to {out_dir}/")


if __name__ == "__main__":
    main()
