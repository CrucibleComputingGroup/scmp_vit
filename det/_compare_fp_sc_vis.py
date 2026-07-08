"""Build side-by-side FP vs SC@len192 detection comparisons for the first 10 images."""
from pathlib import Path

import cv2
import numpy as np

FP_DIR = Path(
    "/home/yjrcs/SC_V/vit_sc/det/results/fp5000_sz1024_revis"
)
SC_DIR = Path(
    "/home/yjrcs/SC_V/vit_sc/det/results/sc_sw30_len192_sz1024_n5000_soft_beit/vis"
)
OUT_DIR = Path("/home/yjrcs/SC_V/vit_sc/det/results/compare_fp_vs_sc192")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# First 10 shared image IDs (already confirmed identical between FP and SC).
IDS = [1268, 1296, 1353, 1425, 1490, 1503, 1532, 1584, 1675, 1761]

LABEL_H = 40  # banner height in px


def label_band(width: int, text: str, color=(40, 40, 40)) -> np.ndarray:
    band = np.full((LABEL_H, width, 3), color, dtype=np.uint8)
    cv2.putText(band, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return band


def stack_pair(fp_img: np.ndarray, sc_img: np.ndarray) -> np.ndarray:
    # Match heights by resizing SC to FP's height (they should already match).
    h_fp = fp_img.shape[0]
    if sc_img.shape[0] != h_fp:
        scale = h_fp / sc_img.shape[0]
        sc_img = cv2.resize(sc_img, (int(sc_img.shape[1] * scale), h_fp))
    fp_banner = label_band(fp_img.shape[1], "FP (baseline)")
    sc_banner = label_band(sc_img.shape[1], "SC  len=192")
    fp_col = np.vstack([fp_banner, fp_img])
    sc_col = np.vstack([sc_banner, sc_img])
    gap = np.full((fp_col.shape[0], 6, 3), 255, dtype=np.uint8)
    return np.hstack([fp_col, gap, sc_col])


for img_id in IDS:
    name = f"{img_id:012d}.jpg"
    fp = cv2.imread(str(FP_DIR / name))
    sc = cv2.imread(str(SC_DIR / name))
    if fp is None or sc is None:
        print(f"[skip] {name}: fp={fp is not None} sc={sc is not None}")
        continue
    out = stack_pair(fp, sc)
    out_path = OUT_DIR / name
    cv2.imwrite(str(out_path), out)
    print(f"  {name}: FP {fp.shape[1]}x{fp.shape[0]} | SC {sc.shape[1]}x{sc.shape[0]} -> {out_path}")

print(f"[OK] Saved {len(IDS)} comparisons to {OUT_DIR}/")
