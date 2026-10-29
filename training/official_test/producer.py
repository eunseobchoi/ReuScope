"""Train CV2024 counterfactual arms and evaluate on official AIIMS test.

This is a narrow extension of phase5_counterfactual_v5.py.  It keeps the
same DINOv2-L/14 LoRA training protocol and final-epoch reporting, but adds
an official-test evaluation dataset loaded from the Figshare class-separated
test release.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random as py_random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from torchvision import transforms


ROOT = Path(os.environ.get("CAPSULE_ROOT", ".")).resolve()
OUT = ROOT / "results"
CV2024_ROOT = Path(os.environ.get("CV2024_ROOT", ROOT / "data/cv2024/Dataset"))
if (CV2024_ROOT / "Dataset").is_dir():
    CV2024_ROOT = CV2024_ROOT / "Dataset"

sys.path.insert(0, str(ROOT / "src"))
import dataset_official as _ds  # noqa: E402

_ds.DATA_ROOT = Path(
    os.environ.get("KVASIR_ROOT", ROOT / "data/kvasir_capsule/labelled_images")
)
if (_ds.DATA_ROOT / "labelled_images").is_dir():
    _ds.DATA_ROOT = _ds.DATA_ROOT / "labelled_images"
_ds.SPLITS_DIR = Path(os.environ.get("KVASIR_SPLITS_DIR", ROOT / "data/official_splits"))
from dataset_official import LABEL_TO_FOLDER  # noqa: E402


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CV2024_CLASSES = [
    "Angioectasia",
    "Bleeding",
    "Erosion",
    "Erythema",
    "Foreign Body",
    "Lymphangiectasia",
    "Normal",
    "Polyp",
    "Ulcer",
    "Worms",
]
CV2024_SOURCES = ["KVASIR", "SEE-AI", "KID", "AIIMS"]

TF_EVAL = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
TF_TRAIN = TF_EVAL


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def file_md5(path: str | Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def set_determinism(seed: int) -> None:
    py_random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:
        print(f"  [warn] use_deterministic_algorithms failed: {e}", flush=True)


def worker_init_fn(worker_id: int) -> None:
    seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(seed)
    py_random.seed(seed)


class CV2024DS(torch.utils.data.Dataset):
    def __init__(self, csv_path: str | Path, classes=CV2024_CLASSES, tf=TF_TRAIN):
        df = (
            pd.read_csv(csv_path)
            if str(csv_path).endswith(".csv")
            else pd.read_excel(csv_path)
        )
        label_cols = [c for c in df.columns if c in classes]
        if not label_cols:
            raise ValueError(f"No class columns in {csv_path}")
        df = df[df[label_cols].sum(axis=1) > 0].reset_index(drop=True)
        self.df = df
        self.classes = classes
        self.label_cols = label_cols
        self.tf = tf
        self.sources = df["Dataset"].values if "Dataset" in df.columns else None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        rel = row["image_path"].replace("\\", "/")
        path = CV2024_ROOT / rel
        img = Image.open(path).convert("RGB")
        for c in self.label_cols:
            if row[c] == 1:
                label = self.classes.index(c)
                break
        else:
            label = 0
        return self.tf(img), label, idx


class OfficialTestDS(torch.utils.data.Dataset):
    """Official CV2024 class-separated test release from Figshare."""

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root: str | Path, classes=CV2024_CLASSES, tf=TF_EVAL):
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(root)
        candidates = [root] + [p for p in root.iterdir() if p.is_dir()]
        base = None
        for cand in candidates:
            if all((cand / c).is_dir() for c in classes):
                base = cand
                break
        if base is None:
            raise ValueError(f"Cannot find class directories under {root}")

        self.base = base
        self.classes = classes
        self.tf = tf
        self.items: list[tuple[str, int]] = []
        for label, cls in enumerate(classes):
            for path in sorted((base / cls).iterdir()):
                if path.suffix.lower() in self.IMG_EXTS:
                    self.items.append((str(path), label))
        if not self.items:
            raise ValueError(f"No images found under {base}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        return self.tf(Image.open(path).convert("RGB")), label, idx


class KvasirEvalDS(torch.utils.data.Dataset):
    KVASIR_TO_CV = {
        "Angiectasia": "Angioectasia",
        "Blood": "Bleeding",
        "Erosion": "Erosion",
        "Erythematous": "Erythema",
        "Foreign Bodies": "Foreign Body",
        "Lymphangiectasia": "Lymphangiectasia",
        "Normal": "Normal",
        "Pylorus": None,
        "Ileo-cecal valve": None,
        "Reduced Mucosal View": None,
        "Ulcer": "Ulcer",
    }

    def __init__(self):
        self.items = []
        with open(_ds.SPLITS_DIR / "split_1.csv") as f:
            for row in csv.DictReader(f):
                kv_lbl = row["label"]
                cv_lbl = self.KVASIR_TO_CV.get(kv_lbl)
                if cv_lbl is None or cv_lbl not in CV2024_CLASSES:
                    continue
                if kv_lbl not in LABEL_TO_FOLDER:
                    continue
                folder = LABEL_TO_FOLDER[kv_lbl]
                path = _ds.DATA_ROOT / folder / row["filename"]
                self.items.append((str(path), CV2024_CLASSES.index(cv_lbl)))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, lbl = self.items[idx]
        return TF_EVAL(Image.open(path).convert("RGB")), lbl, idx


class LoRALinear(nn.Module):
    def __init__(self, base, r=8, alpha=16.0):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad = False
        self.lora_a = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scale


class DINOv2LoRA(nn.Module):
    def __init__(self, n_cls=10, r=8):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.lora_wrappers = nn.ModuleList()
        for blk in self.backbone.blocks[-4:]:
            w = LoRALinear(blk.attn.qkv, r=r)
            blk.attn.qkv = w
            self.lora_wrappers.append(w)
        self.head = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_cls),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


def cbw(labels: torch.Tensor, n: int, beta=0.999) -> torch.Tensor:
    c = torch.bincount(labels, minlength=n).float()
    e = 1.0 - torch.pow(beta, c)
    e[c == 0] = 1.0
    w = (1.0 - beta) / e
    w = w / w.sum() * n
    w[c == 0] = 0
    return w


@torch.no_grad()
def ev(m, loader, classes=CV2024_CLASSES, dataset=None):
    m.eval()
    preds, labels, indices, scores = [], [], [], []
    for img, label, idx in loader:
        img = img.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            logits = m(img)
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        preds.extend(probs.argmax(1).tolist())
        scores.append(probs)
        labels.extend(label.tolist())
        indices.extend(idx.tolist() if hasattr(idx, "tolist") else idx)

    p = np.array(preds)
    y = np.array(labels)
    idx_arr = np.array(indices)
    s = np.concatenate(scores, axis=0)
    majority = int(np.bincount(y).argmax()) if len(y) else 0
    per_class = {}
    auc_values = []
    for i, cls in enumerate(classes):
        mask = y == i
        auc = None
        if mask.sum() > 0 and mask.sum() < len(y):
            auc = float(roc_auc_score(mask.astype(int), s[:, i]))
            auc_values.append(auc)
        per_class[cls] = {
            "recall": None if mask.sum() == 0 else float((p[mask] == i).mean()),
            "auc": auc,
            "n": int(mask.sum()),
        }
    mean_auc = float(np.mean(auc_values)) if auc_values else None
    bal_acc = float(balanced_accuracy_score(y, p))
    out = {
        "acc": float((p == y).mean()),
        "bal_acc": bal_acc,
        "mean_auc": mean_auc,
        "combined": None if mean_auc is None else float((mean_auc + bal_acc) / 2.0),
        "f1_macro": float(f1_score(y, p, average="macro", zero_division=0)),
        "null_acc": float((np.full_like(y, majority) == y).mean()),
        "n_test": int(len(y)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y, p, labels=list(range(len(classes)))).tolist(),
    }
    if dataset is not None and getattr(dataset, "sources", None) is not None:
        per_source = {}
        src_values = dataset.sources[idx_arr]
        for src in CV2024_SOURCES:
            src_mask = np.array([s == src for s in src_values])
            if src_mask.sum() == 0:
                per_source[src] = {"n": 0, "bal_acc": None, "acc": None, "per_class": {}}
                continue
            ps = p[src_mask]
            ys = y[src_mask]
            ss = s[src_mask]
            ba = (
                float((ps == ys).mean())
                if len(set(ys)) < 2
                else float(balanced_accuracy_score(ys, ps))
            )
            pc_s = {}
            auc_s = []
            for i, cls in enumerate(classes):
                cm = ys == i
                auc = None
                if cm.sum() > 0 and cm.sum() < len(ys):
                    auc = float(roc_auc_score(cm.astype(int), ss[:, i]))
                    auc_s.append(auc)
                if cm.sum() > 0:
                    pc_s[cls] = {
                        "recall": float((ps[cm] == i).mean()),
                        "auc": auc,
                        "n": int(cm.sum()),
                    }
            mean_auc_s = float(np.mean(auc_s)) if auc_s else None
            per_source[src] = {
                "n": int(src_mask.sum()),
                "bal_acc": ba,
                "mean_auc": mean_auc_s,
                "combined": None if mean_auc_s is None else float((mean_auc_s + ba) / 2.0),
                "acc": float((ps == ys).mean()),
                "per_class": pc_s,
            }
        out["per_source"] = per_source
    return out


def evaluate_optional(m, loader, dataset, enabled: bool):
    if not enabled:
        return None
    return ev(m, loader, dataset=dataset)


def train_one_seed(train_csv, seed, args):
    set_determinism(seed)
    train_ds = CV2024DS(train_csv, tf=TF_TRAIN)
    labels = torch.tensor(
        [
            train_ds.classes.index([c for c in train_ds.label_cols if train_ds.df.iloc[i][c] == 1][0])
            for i in range(len(train_ds))
        ]
    )
    weights = cbw(labels, len(CV2024_CLASSES)).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=g,
    )

    orig_val = CV2024DS(args.orig_val_csv, tf=TF_EVAL)
    dedup_val = CV2024DS(args.dedup_val_csv, tf=TF_EVAL)
    kvasir_test = KvasirEvalDS()
    official_test = OfficialTestDS(args.official_test_dir, tf=TF_EVAL)
    orig_loader = DataLoader(orig_val, batch_size=args.batch * 2, shuffle=False, num_workers=args.workers)
    dedup_loader = DataLoader(dedup_val, batch_size=args.batch * 2, shuffle=False, num_workers=args.workers)
    kvasir_loader = DataLoader(kvasir_test, batch_size=args.batch * 2, shuffle=False, num_workers=args.workers)
    official_loader = DataLoader(
        official_test, batch_size=args.batch * 2, shuffle=False, num_workers=args.workers
    )

    print(
        f"  Train={len(train_ds)} OrigVal={len(orig_val)} DedupVal={len(dedup_val)} "
        f"KvasirTest={len(kvasir_test)} OfficialTest={len(official_test)}",
        flush=True,
    )

    model = DINOv2LoRA(n_cls=len(CV2024_CLASSES)).to(DEVICE)
    wrapped = nn.DataParallel(model) if torch.cuda.device_count() > 1 else model

    lora = [p for wr in model.lora_wrappers for p in [wr.lora_a, wr.lora_b]]
    head_params = list(model.head.parameters())
    opt = torch.optim.AdamW([{"params": head_params, "lr": 3e-4}], weight_decay=1e-4)
    lora_added = False
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    history = []

    for ep in range(args.epochs):
        if ep >= 3 and not lora_added:
            opt.add_param_group({"params": lora, "lr": 1e-4, "weight_decay": 1e-4})
            lora_added = True
            print(f"  [seed={seed} ep={ep+1}] LoRA params added", flush=True)

        wrapped.train()
        model.backbone.eval()
        losses = []
        t0 = time.perf_counter()
        for step, (img, label, _) in enumerate(train_loader):
            img = img.to(DEVICE, non_blocking=True)
            label = label.to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                logits = wrapped(img)
                loss = loss_fn(logits, label)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss seed={seed} ep={ep+1} step={step}")
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())

        do_full_eval = args.eval_each_epoch or ep == args.epochs - 1
        official_metrics = ev(wrapped, official_loader, dataset=official_test)
        record = {
            "epoch": ep + 1,
            "loss": float(np.mean(losses)),
            "official_test": official_metrics,
            "orig_val": evaluate_optional(wrapped, orig_loader, orig_val, do_full_eval),
            "dedup_val": evaluate_optional(wrapped, dedup_loader, dedup_val, do_full_eval),
            "kvasir_s1": evaluate_optional(wrapped, kvasir_loader, kvasir_test, do_full_eval),
            "train_s": float(time.perf_counter() - t0),
        }
        history.append(record)
        print(
            f"  [seed={seed} ep={ep+1}] official bal={official_metrics['bal_acc']:.4f} "
            f"auc={official_metrics['mean_auc']:.4f} "
            f"combined={official_metrics['combined']:.4f} "
            f"acc={official_metrics['acc']:.4f} loss={record['loss']:.4f} "
            f"[{record['train_s']:.0f}s]",
            flush=True,
        )
    return history


def summarize_runs(runs):
    vals = np.array([r["last"]["official_test"]["bal_acc"] for r in runs], dtype=float)
    acc = np.array([r["last"]["official_test"]["acc"] for r in runs], dtype=float)
    auc = np.array([r["last"]["official_test"]["mean_auc"] for r in runs], dtype=float)
    combined = np.array([r["last"]["official_test"]["combined"] for r in runs], dtype=float)
    return {
        "n": int(len(vals)),
        "official_test_bal_acc_mean": float(vals.mean()) if len(vals) else None,
        "official_test_bal_acc_sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        "official_test_mean_auc_mean": float(auc.mean()) if len(auc) else None,
        "official_test_mean_auc_sd": float(auc.std(ddof=1)) if len(auc) > 1 else 0.0,
        "official_test_combined_mean": float(combined.mean()) if len(combined) else None,
        "official_test_combined_sd": float(combined.std(ddof=1)) if len(combined) > 1 else 0.0,
        "official_test_acc_mean": float(acc.mean()) if len(acc) else None,
        "official_test_acc_sd": float(acc.std(ddof=1)) if len(acc) > 1 else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--orig_val_csv", default=str(CV2024_ROOT / "validation/validation_data.xlsx"))
    ap.add_argument("--dedup_val_csv", required=True)
    ap.add_argument("--official_test_dir", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--output", required=True)
    ap.add_argument("--eval_each_epoch", action="store_true")
    ap.add_argument("--no_batch128_assert", action="store_true")
    args = ap.parse_args()

    if not args.no_batch128_assert and args.batch != 128:
        raise SystemExit(f"Expected canonical batch=128, got {args.batch}")
    seed_count_match = re.search(r"_n(\d+)\.json$", Path(args.output).name)
    if seed_count_match:
        expected = int(seed_count_match.group(1))
        if len(args.seeds) != expected:
            raise SystemExit(
                f"Output filename declares n={expected}, but got seeds {args.seeds}"
            )

    final_path = OUT / args.output
    partial_path = final_path.with_name(final_path.name + ".partial")
    tmp_path = final_path.with_name(final_path.name + ".tmp")
    final_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Args: {vars(args)}", flush=True)
    meta = {
        "script_path": os.path.abspath(__file__),
        "script_sha256": file_sha256(__file__),
        "script_version": "official-test-v1",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "train_csv_md5": file_md5(args.train_csv),
        "orig_val_md5": file_md5(args.orig_val_csv) if os.path.exists(args.orig_val_csv) else None,
        "dedup_val_csv_md5": file_md5(args.dedup_val_csv) if os.path.exists(args.dedup_val_csv) else None,
        "official_test_dir": args.official_test_dir,
        "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "determinism_enabled": True,
    }
    print(f"Provenance: script_sha256={meta['script_sha256'][:16]}...", flush=True)

    results = {"args": vars(args), "meta": meta, "runs": []}
    for seed in args.seeds:
        print(f"\n=== seed={seed} ===", flush=True)
        history = train_one_seed(args.train_csv, seed, args)
        results["runs"].append({"seed": seed, "history": history, "last": history[-1]})
        results["summary"] = summarize_runs(results["runs"])
        with open(partial_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

    meta["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    results["summary"] = summarize_runs(results["runs"])
    with open(tmp_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    os.replace(tmp_path, final_path)
    if partial_path.exists():
        partial_path.unlink()
    print(f"\nSaved {final_path}", flush=True)
    print(json.dumps(results["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
