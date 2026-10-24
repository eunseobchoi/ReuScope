"""Frozen-backbone training used by ../crossmodel.py.

The public CNN recipe uses ResNet-50 and ConvNeXt-Tiny with nonlinear heads,
resize/center-crop preprocessing and final-epoch evaluation. DINOv2 modes
remain available through this module's CLI. Use the wrapper
for pinned local CNN weights, image inventories and prediction exports.
"""
import argparse
import hashlib
import json
import os
import random as py_random
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("CAPSULE_ROOT", Path(__file__).resolve().parents[1])).resolve()
ARTIFACT_ROOT = Path(os.environ.get("CAPSULE_ARTIFACT_ROOT", ROOT))
OUT = ARTIFACT_ROOT / "results"

def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True,
                    choices=[
                        "dinov2_vits14",
                        "dinov2_vitb14",
                        "dinov2_vitl14",
                        "dinov2_vitg14",
                        "resnet50",
                        "convnext_tiny",
                    ])
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--orig_val_csv", required=True)
    ap.add_argument("--dedup_val_csv", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--output", required=True)
    return ap


if __name__ == "__main__":
    _cli_parser = build_parser()
    _cli_args = _cli_parser.parse_args()


sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.path_resolution import require_directory

CV2024_ROOT = require_directory("CV2024_ROOT")
if (CV2024_ROOT / "Dataset").is_dir():
    CV2024_ROOT = CV2024_ROOT / "Dataset"


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import torchvision
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision import transforms

import dataset_official as _ds
from dataset_official import KvasirCapsuleOfficial

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CV2024_CLASSES = [
    "Angioectasia", "Bleeding", "Erosion", "Erythema",
    "Foreign Body", "Lymphangiectasia", "Normal", "Polyp", "Ulcer", "Worms",
]


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_input_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    return (ROOT / p).resolve()


def scrub_path(value):
    if not isinstance(value, str):
        return value
    out = value.replace("\\", "/")
    replacements = [
        ("<CAPSULE_ARTIFACT_ROOT>", ARTIFACT_ROOT),
        ("<CAPSULE_ROOT>", ROOT),
        ("<CV2024_ROOT>", CV2024_ROOT),
        ("<KVASIR_ROOT>", _ds.DATA_ROOT),
        ("<KVASIR_SPLITS_DIR>", _ds.SPLITS_DIR),
    ]
    for token, path in sorted(replacements, key=lambda kv: len(str(kv[1])), reverse=True):
        if path is None:
            continue
        path_s = str(path.resolve()).replace("\\", "/")
        out = out.replace(path_s, token)
    return out


def scrub_obj(obj):
    if isinstance(obj, dict):
        return {k: scrub_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub_obj(v) for v in obj]
    if isinstance(obj, tuple):
        return [scrub_obj(v) for v in obj]
    if isinstance(obj, str):
        return scrub_path(obj)
    return obj


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
        print(f"  [warn] deterministic algorithms unavailable: {e}", flush=True)


def worker_init_fn(worker_id: int) -> None:
    seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(seed)
    py_random.seed(seed)

# === Preprocessing ===
TF_DINOV2 = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
TF_RESNET = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_dataset_class(tf):
    """Return a Dataset class using given transform."""
    class CV2024DS(torch.utils.data.Dataset):
        def __init__(self, csv_path):
            p = resolve_input_path(csv_path)
            df = pd.read_csv(p)
            self.items = []
            for _, row in df.iterrows():
                img_rel = row.get("image_path", row.get("filename"))
                if pd.isna(img_rel) or not str(img_rel).strip():
                    raise ValueError(f"Missing image path in {p}")
                rel = str(img_rel).replace("\\", "/")
                # Read one-hot labels in CV2024 class order.
                onehot = np.asarray([row.get(c, 0) for c in CV2024_CLASSES], dtype=float)
                if not np.isfinite(onehot).all() or not np.isin(onehot, [0, 1]).all() or onehot.sum() != 1:
                    raise ValueError(f"Invalid finite binary one-hot label in {p}")
                lbl = int(onehot.argmax())
                path = CV2024_ROOT / rel if not rel.startswith("/") else Path(rel)
                self.items.append((str(path), lbl))
            if not self.items:
                raise ValueError(f"Empty image list: {p}")

        def __len__(self): return len(self.items)

        def __getitem__(self, idx):
            path, lbl = self.items[idx]
            return tf(Image.open(path).convert("RGB")), lbl, idx
    return CV2024DS


class KvasirCapsuleAdapter(torch.utils.data.Dataset):
    """Wraps Kvasir-Capsule split_1, remaps labels to CV2024 10-class."""
    # Map 11-class Kvasir official labels → 10-class CV2024 labels
    KV_TO_CV = {
        "Normal": "Normal",
        "Blood": "Bleeding",
        "Angiectasia": "Angioectasia",
        "Erosion": "Erosion",
        "Erythematous": "Erythema",
        "Foreign Bodies": "Foreign Body",
        "Lymphangiectasia": "Lymphangiectasia",
        "Ulcer": "Ulcer",
        # Ileo-cecal valve, Pylorus, Reduced Mucosal View → no CV2024 equivalent; skip
    }

    def __init__(self, split_idx, tf):
        self.inner = KvasirCapsuleOfficial(split=f"split_{split_idx}")
        self.tf = tf
        self.items = []
        # inner.items: list of (path, class_idx, label_name)
        for path, _cls, kv_name in self.inner.items:
            if kv_name not in self.KV_TO_CV: continue
            cv_lbl = CV2024_CLASSES.index(self.KV_TO_CV[kv_name])
            self.items.append((str(path), cv_lbl))

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        path, lbl = self.items[idx]
        return self.tf(Image.open(path).convert("RGB")), lbl, idx


# === Backbones ===
class LoRALinear(nn.Module):
    def __init__(self, base, r=8, alpha=16.0):
        super().__init__()
        self.base = base
        for p in base.parameters(): p.requires_grad = False
        self.lora_a = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r
    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scale


class DINOv2LoRA(nn.Module):
    def __init__(self, backbone_name, n_cls=10, r=8):
        super().__init__()
        repo = str(require_directory("DINOV2_REPO"))
        self.backbone = torch.hub.load(repo, backbone_name, source="local")
        feat_dim = {"dinov2_vits14": 384, "dinov2_vitb14": 768,
                    "dinov2_vitl14": 1024, "dinov2_vitg14": 1536}[backbone_name]
        for p in self.backbone.parameters(): p.requires_grad = False
        self.lora_wrappers = nn.ModuleList()
        for blk in self.backbone.blocks[-4:]:
            w = LoRALinear(blk.attn.qkv, r=r); blk.attn.qkv = w
            self.lora_wrappers.append(w)
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim), nn.Linear(feat_dim, 256),
            nn.GELU(), nn.Dropout(0.1), nn.Linear(256, n_cls))

    def forward(self, x): return self.head(self.backbone(x))


class ResNet50Frozen(nn.Module):
    """ImageNet-supervised ResNet-50, frozen features and nonlinear head."""
    def __init__(self, n_cls=10):
        super().__init__()
        net = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT)
        # Replace final FC with head
        for p in net.parameters(): p.requires_grad = False
        net.fc = nn.Identity()
        self.backbone = net  # outputs 2048
        self.head = nn.Sequential(
            nn.LayerNorm(2048), nn.Linear(2048, 256),
            nn.GELU(), nn.Dropout(0.1), nn.Linear(256, n_cls))

    def forward(self, x): return self.head(self.backbone(x))

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self


class ConvNeXtTinyFrozen(nn.Module):
    """Frozen ConvNeXt features; fresh LayerNorm and nonlinear head."""
    def __init__(self, n_cls=10):
        super().__init__()
        net = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.DEFAULT)
        for p in net.parameters():
            p.requires_grad = False
        net.classifier = nn.Identity()
        self.backbone = net
        self.head = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768, 256),
            nn.GELU(), nn.Dropout(0.1), nn.Linear(256, n_cls))

    def forward(self, x):
        feat = self.backbone(x)
        if feat.ndim > 2:
            feat = torch.flatten(feat, 1)
        return self.head(feat)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self


def build_model(backbone):
    if backbone.startswith("dinov2"):
        return DINOv2LoRA(backbone_name=backbone), TF_DINOV2
    elif backbone == "resnet50":
        return ResNet50Frozen(), TF_RESNET
    elif backbone == "convnext_tiny":
        return ConvNeXtTinyFrozen(), TF_RESNET
    else:
        raise ValueError(f"Unknown backbone: {backbone}")


# === Training ===
def cbw(labels, n, beta=0.999):
    c = torch.bincount(labels, minlength=n).float()
    e = 1.0 - torch.pow(beta, c); e[c == 0] = 1.0
    w = (1.0 - beta) / e; w = w / w.sum() * n; w[c == 0] = 0
    return w


def eval_one(model, loader, n_cls):
    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(DEVICE, non_blocking=True)
            logits = model(x)
            if not torch.isfinite(logits).all():
                raise RuntimeError("Nonfinite evaluation logits")
            preds.append(logits.argmax(-1).cpu())
            tgts.append(y)
    if not preds:
        raise ValueError("Empty evaluation pool")
    preds = torch.cat(preds).numpy()
    tgts = torch.cat(tgts).numpy()
    return {
        "acc": float((preds == tgts).mean()),
        "bal_acc": float(balanced_accuracy_score(tgts, preds)),
        "f1_macro": float(f1_score(tgts, preds, labels=np.unique(tgts), average="macro", zero_division=0)),
        "f1_macro_class_set": np.unique(tgts).tolist(),
        "null_acc": float((tgts == np.bincount(tgts).argmax()).mean()),
        "n_test": len(tgts),
    }


def step_optimizer(scaler, optimizer, scheduler):
    """GradScaler lowers its scale when it skips a nonfinite-gradient update."""
    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    updated = scaler.get_scale() >= scale_before
    if updated:
        scheduler.step()
    return updated


def train_one_seed(backbone, train_csv, seed, args, orig_val_csv, dedup_val_csv):
    set_determinism(seed)
    model, tf = build_model(backbone)
    model = model.to(DEVICE)
    DS = build_dataset_class(tf)
    train = DS(train_csv)
    if len(train) < args.batch:
        raise ValueError("Training pool smaller than drop-last batch")
    orig_val = DS(orig_val_csv)
    dedup_val = DS(dedup_val_csv)
    kv_test = KvasirCapsuleAdapter(split_idx=1, tf=tf)
    print(f"  Train={len(train)} OrigVal={len(orig_val)} DedupVal={len(dedup_val)} KvasirTest={len(kv_test)}", flush=True)

    g = torch.Generator()
    g.manual_seed(seed)
    l_tr = DataLoader(
        train,
        batch_size=args.batch,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=g,
    )
    l_ov = DataLoader(orig_val, batch_size=args.batch * 2, shuffle=False, num_workers=6)
    l_dv = DataLoader(dedup_val, batch_size=args.batch * 2, shuffle=False, num_workers=6)
    l_kt = DataLoader(kv_test, batch_size=args.batch * 2, shuffle=False, num_workers=6)

    # Class-balanced weights
    train_labels = torch.tensor([t[1] for t in train.items], dtype=torch.long)
    weights = cbw(train_labels, len(CV2024_CLASSES), beta=0.999).to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(l_tr))
    scaler = torch.cuda.amp.GradScaler()

    history = []
    t0 = time.perf_counter()
    for ep in range(args.epochs):
        model.train()
        model.backbone.eval()
        optimizer_steps = skipped_steps = 0
        for x, y, _ in l_tr:
            x = x.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.float16 if backbone == "resnet50" else torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits, y, weight=weights, label_smoothing=0.05)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at seed={seed} epoch={ep + 1}")
            scaler.scale(loss).backward()
            updated = step_optimizer(scaler, opt, sched)
            optimizer_steps += int(updated)
            skipped_steps += int(not updated)

        met_ov = eval_one(model, l_ov, len(CV2024_CLASSES))
        met_dv = eval_one(model, l_dv, len(CV2024_CLASSES))
        met_kt = eval_one(model, l_kt, len(CV2024_CLASSES))
        dt = time.perf_counter() - t0; t0 = time.perf_counter()
        print(f"  [bb={backbone} seed={seed} ep={ep+1}] orig_val bal={met_ov['bal_acc']:.4f} "
              f"| dedup_val bal={met_dv['bal_acc']:.4f} "
              f"| kvasir_s1 bal={met_kt['bal_acc']:.4f} [{dt:.0f}s]", flush=True)
        history.append({"epoch": ep + 1, "loss": float(loss.item()),
                        "orig_val": met_ov, "dedup_val": met_dv, "kvasir_s1": met_kt,
                        "train_s": dt, "optimizer_steps": optimizer_steps,
                        "amp_skipped_steps": skipped_steps, "amp_scale": scaler.get_scale()})
    return {"seed": seed, "backbone": backbone, "history": history, "last": history[-1]}


def main(ap=None, args=None):
    if ap is None:
        ap = build_parser()
    if args is None:
        args = ap.parse_args()
    if args.epochs < 1 or args.batch < 1 or len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 for s in args.seeds):
        ap.error("Positive epochs/batch and unique uint32 seeds required")
    require_directory("TORCH_HOME")
    if args.backbone.startswith("dinov2"):
        require_directory("DINOV2_REPO")
    require_directory("KVASIR_ROOT")
    require_directory("KVASIR_SPLITS_DIR")
    final_path = OUT / args.output
    partial_path = final_path.with_name(final_path.name + ".partial")
    if final_path.exists() or partial_path.exists():
        ap.error("Use a fresh output path; prior results will not be replaced")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Args: {vars(args)}")
    input_paths = {
        "train_csv": resolve_input_path(args.train_csv),
        "orig_val_csv": resolve_input_path(args.orig_val_csv),
        "dedup_val_csv": resolve_input_path(args.dedup_val_csv),
    }
    script_path = Path(__file__).resolve()
    meta = {
        "description": "Frozen-backbone robustness diagnostic; reports final-epoch balanced accuracy, not an official CV2024 leaderboard metric.",
        "script_path": scrub_path(str(script_path)),
        "script_sha256": file_sha256(script_path),
        "script_version": "cross_model",
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "determinism_enabled": True,
        "determinism_warn_only": True,
        "bit_exact_reproduction_claimed": False,
        "dinov2_repo": os.environ.get("DINOV2_REPO"),
        "backbone_mode": "eval throughout training; nonlinear head in train mode",
        "input_files": {
            key: {
                "path": scrub_path(str(path)),
                "sha256": file_sha256(path),
            }
            for key, path in input_paths.items()
        },
    }

    runs = []
    results = {"args": scrub_obj(vars(args)), "meta": meta, "runs": runs}
    for seed in args.seeds:
        print(f"\n=== {args.backbone} seed={seed} ===")
        r = train_one_seed(args.backbone, args.train_csv, seed, args,
                           args.orig_val_csv, args.dedup_val_csv)
        runs.append(r)
        # Save after each seed
        with open(partial_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved {partial_path}")
    meta["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with open(partial_path, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(partial_path, final_path)
    print(f"Saved {final_path}")


if __name__ == "__main__":
    main(_cli_parser, _cli_args)
