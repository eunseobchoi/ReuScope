"""Kvasir-Capsule dataset using the official two-fold split CSVs.

Official label and split conventions:
- 11 classes (not 14) — polyp, ampulla_of_vater, blood_hematin dropped by official benchmark
- Use split_0.csv and split_1.csv from simula/kvasir-capsule

Supply the split CSVs and labelled_images directory externally. Frame-list
folds must not be assumed to be video-disjoint.
"""
import os
import csv
from pathlib import Path, PureWindowsPath
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


DATA_ROOT = Path(os.environ["KVASIR_ROOT"]) if os.environ.get("KVASIR_ROOT") else None
SPLITS_DIR = Path(os.environ["KVASIR_SPLITS_DIR"]) if os.environ.get("KVASIR_SPLITS_DIR") else None

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
        if DATA_ROOT is None or SPLITS_DIR is None:
            raise ValueError("Set KVASIR_ROOT and KVASIR_SPLITS_DIR for Kvasir evaluation")
        if not DATA_ROOT.is_dir() or not SPLITS_DIR.is_dir():
            raise ValueError("KVASIR_ROOT and KVASIR_SPLITS_DIR must be existing directories")
        self.transform = transform
        self.items = []
        with open(SPLITS_DIR / f"{split}.csv", newline="") as f:
            reader = csv.DictReader(f)
            if any((reader.fieldnames or []).count(name) != 1 for name in ("filename", "label")):
                raise ValueError(f"{split}.csv requires unique filename and label columns")
            for row in reader:
                fn = row["filename"]
                lbl = row["label"]
                if None in row or not isinstance(lbl, str) or not lbl.strip():
                    raise ValueError(f"Malformed {split}.csv row at line {reader.line_num}")
                if lbl not in LABEL_TO_FOLDER:
                    continue
                if (not isinstance(fn, str) or not fn.strip() or fn in (".", "..")
                        or "/" in fn or "\\" in fn or "\x00" in fn or PureWindowsPath(fn).drive):
                    raise ValueError(f"Expected a filename without directories at {split}.csv line {reader.line_num}")
                folder = LABEL_TO_FOLDER[lbl]
                path = DATA_ROOT / folder / fn
                self.items.append((str(path), CLASS_TO_IDX[lbl], lbl))
        if not self.items:
            raise ValueError(f"No supported images in {split}.csv")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label, _ = self.items[idx]
        with Image.open(path) as source:
            img = source.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label, idx
