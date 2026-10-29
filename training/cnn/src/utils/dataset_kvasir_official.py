"""Kvasir-Capsule dataset using the official two-fold split CSVs.

Key differences from dataset.py:
- 11 classes (not 14) — polyp, ampulla_of_vater, blood_hematin dropped by official benchmark
- Use split_0.csv and split_1.csv from simula/kvasir-capsule

The released CSVs are treated as official frame-list folds. A filename-prefix
audit finds 7 shared video prefixes across split_0/split_1, so callers must
not interpret these folds as video-disjoint evidence.
"""
import os
import csv
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


DATA_ROOT = Path(os.environ.get("CAPSULE_ROOT", ".") + "/data/kvasir_capsule/labelled_images")
SPLITS_DIR = Path(os.environ.get("CAPSULE_ROOT", ".") + "/data/official_splits")

# Map official CSV label → folder name in labelled_images/
LABEL_TO_FOLDER = {
    "Angiectasia": "angiectasia",
    "Blood": "blood_fresh",   # blood_hematin NOT in official
    "Erosion": "erosion",
    "Erythematous": "erythema",
    "Foreign Bodies": "foreign_body",
    "Ileo-cecal valve": "ileocecal_valve",
    "Lymphangiectasia": "lymphangiectasia",
    "Normal": "normal_clean_mucosa",
    "Pylorus": "pylorus",
    "Reduced Mucosal View": "reduced_mucosal_view",
    "Ulcer": "ulcer",
}

OFFICIAL_CLASSES = sorted(LABEL_TO_FOLDER.keys())  # 11 classes
NUM_CLASSES = len(OFFICIAL_CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(OFFICIAL_CLASSES)}

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def set_data_paths(data_root: str | Path | None = None, splits_dir: str | Path | None = None) -> None:
    """Set dataset roots explicitly for rerunnable release scripts."""
    global DATA_ROOT, SPLITS_DIR
    if data_root is not None:
        DATA_ROOT = Path(data_root)
    if splits_dir is not None:
        SPLITS_DIR = Path(splits_dir)


class KvasirCapsuleOfficial(Dataset):
    """Loads Kvasir-Capsule using official two-fold split CSV."""

    def __init__(self, split: str = "split_0", transform=EVAL_TRANSFORM):
        if split not in ("split_0", "split_1"):
            raise ValueError(f"split must be split_0 or split_1, got {split}")
        self.transform = transform
        self.items = []
        with open(SPLITS_DIR / f"{split}.csv") as f:
            for row in csv.DictReader(f):
                fn = row["filename"]
                lbl = row["label"]
                if lbl not in LABEL_TO_FOLDER:
                    continue
                folder = LABEL_TO_FOLDER[lbl]
                path = DATA_ROOT / folder / fn
                self.items.append((str(path), CLASS_TO_IDX[lbl], lbl))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label, _ = self.items[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label, idx

    def class_distribution(self):
        d = {c: 0 for c in OFFICIAL_CLASSES}
        for _, _, lbl in self.items:
            d[lbl] += 1
        return d

    def unique_videos(self):
        """Return set of video IDs (filename prefix before underscore)."""
        vids = set()
        for path, _, _ in self.items:
            fn = Path(path).name
            vids.add(fn.split("_")[0])
        return vids


if __name__ == "__main__":
    for s in ("split_0", "split_1"):
        ds = KvasirCapsuleOfficial(s)
        d = ds.class_distribution()
        vids = ds.unique_videos()
        print(f"\n{s}: {len(ds)} images, {len(vids)} unique videos")
        for c, n in sorted(d.items(), key=lambda x: -x[1]):
            print(f"  {c:<22} {n:>6}")
