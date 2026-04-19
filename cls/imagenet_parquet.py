"""ImageNet-1k validation dataset loaded from HuggingFace parquet shards."""
import io
import glob
from pathlib import Path

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from PIL import Image


class ImageNetParquetVal(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        files = sorted(glob.glob(str(Path(root) / "validation-*.parquet")))
        assert files, f"No parquet shards under {root}"
        self.shards = []
        offsets = [0]
        for f in files:
            t = pq.read_table(f, columns=["image", "label"])
            self.shards.append(t)
            offsets.append(offsets[-1] + t.num_rows)
        self.offsets = offsets
        self.total = offsets[-1]

    def __len__(self):
        return self.total

    def _locate(self, idx):
        lo, hi = 0, len(self.offsets) - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if self.offsets[mid] <= idx:
                lo = mid
            else:
                hi = mid
        return lo, idx - self.offsets[lo]

    def __getitem__(self, idx):
        s, i = self._locate(idx)
        row = self.shards[s].slice(i, 1).to_pydict()
        img_bytes = row["image"][0]["bytes"]
        label = int(row["label"][0])
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label
