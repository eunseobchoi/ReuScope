"""DINOv2 fixed-list training for the primary comparisons.

Keeps per-source evaluation, final-epoch predictions and trainable checkpoints.
See ../README.md for input lists, seeds and the fresh-training recipe.
"""
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

ROOT = Path(os.environ.get("CAPSULE_ROOT", Path(__file__).resolve().parents[1])).resolve()
OUT = ROOT / "results"

def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--orig_val_csv", required=True)
    ap.add_argument("--dedup_val_csv", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument(
        "--backbone",
        choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"],
        default="dinov2_vitl14",
        help="DINOv2 backbone; default: vitl14.",
    )
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--prediction_output",
        help=(
            "Write final-epoch per-frame probabilities for --dedup_val_csv. "
            "Single-seed invocations only."
        ),
    )
    ap.add_argument(
        "--trainable_checkpoint",
        help=(
            "Write the final learned head/LoRA tensors without duplicating "
            "the frozen DINOv2 backbone. Single-seed invocations only."
        ),
    )
    ap.add_argument("--no_batch128_assert", action="store_true")
    ap.add_argument(
        "--skip_kvasir_eval",
        action="store_true",
        help="Skip the external Kvasir split_1 evaluator for CV2024-only evaluation.",
    )
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
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms

import dataset_official as _ds
from utils.path_resolution import resolve_cv2024_or_kvasir_path  # noqa: E402

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CV2024_CLASSES = [
    "Angioectasia", "Bleeding", "Erosion", "Erythema", "Foreign Body",
    "Lymphangiectasia", "Normal", "Polyp", "Ulcer", "Worms"
]
CV2024_SOURCES = ["KVASIR", "SEE-AI", "KID", "AIIMS"]

TF_EVAL = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
# Identical train/eval transforms avoid augmentation noise in paired-seed comparisons.
TF_TRAIN = TF_EVAL


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def set_determinism(seed):
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


def worker_init_fn(worker_id):
    seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(seed)
    py_random.seed(seed)


class CV2024DS(torch.utils.data.Dataset):
    """CSV-based dataset with optional Dataset-source tracking."""
    def __init__(self, csv_path, classes=CV2024_CLASSES, tf=TF_TRAIN):
        df = pd.read_csv(csv_path)
        label_cols = [c for c in df.columns if c in classes]
        if not label_cols:
            raise ValueError(f"No class columns in {csv_path}")
        labels = df[label_cols].to_numpy(dtype=float)
        if (not len(df) or not np.isfinite(labels).all()
                or not np.isin(labels, [0, 1]).all()
                or not (labels.sum(axis=1) == 1).all()):
            raise ValueError(f"Expected nonempty finite binary one-hot labels in {csv_path}")
        if "image_path" not in df or not df["image_path"].map(
                lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"Expected nonblank image_path strings in {csv_path}")
        df = df.reset_index(drop=True)
        self.df = df
        self.classes = classes
        self.label_cols = label_cols
        self.tf = tf
        # Track Dataset column if present (for per-source eval)
        if "Dataset" in df.columns:
            if not df["Dataset"].map(
                    lambda value: isinstance(value, str) and bool(value.strip())).all():
                raise ValueError(f"Expected nonblank Dataset source names in {csv_path}")
            self.sources = df["Dataset"].values
        else:
            self.sources = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = resolve_cv2024_or_kvasir_path(
            row["image_path"],
            cv2024_root=CV2024_ROOT,
            kvasir_data_root=_ds.DATA_ROOT,
        )
        with Image.open(path) as source:
            img = source.convert("RGB")
        for c in self.label_cols:
            if row[c] == 1:
                label = self.classes.index(c)
                break
        else:
            label = 0
        return self.tf(img), label, idx


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
        for path, _, kv_lbl in _ds.KvasirCapsuleOfficial(split="split_1", transform=None).items:
            cv_lbl = self.KVASIR_TO_CV.get(kv_lbl)
            if cv_lbl is None or cv_lbl not in CV2024_CLASSES:
                continue
            self.items.append((str(path), CV2024_CLASSES.index(cv_lbl)))
        if not self.items:
            raise ValueError("No supported CV2024 classes in Kvasir split_1.csv")

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        path, lbl = self.items[idx]
        with Image.open(path) as source:
            img = source.convert("RGB")
        return TF_EVAL(img), lbl, idx


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
    def __init__(self, n_cls=10, r=8, backbone_name="dinov2_vitl14"):
        super().__init__()
        feature_dims = {
            "dinov2_vits14": 384,
            "dinov2_vitb14": 768,
            "dinov2_vitl14": 1024,
        }
        if backbone_name not in feature_dims:
            raise ValueError(f"Unsupported DINOv2 backbone: {backbone_name}")
        repo = str(require_directory("DINOV2_REPO"))
        self.backbone = torch.hub.load(
            repo, backbone_name, source="local"
        )
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.lora_wrappers = nn.ModuleList()
        for blk in self.backbone.blocks[-4:]:
            w = LoRALinear(blk.attn.qkv, r=r)
            blk.attn.qkv = w
            self.lora_wrappers.append(w)
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dims[backbone_name]),
            nn.Linear(feature_dims[backbone_name], 256),
            nn.GELU(), nn.Dropout(0.1), nn.Linear(256, n_cls))

    def forward(self, x):
        return self.head(self.backbone(x))


def cbw(labels, n, beta=0.999):
    c = torch.bincount(labels, minlength=n).float()
    e = 1.0 - torch.pow(beta, c)
    e[c == 0] = 1.0
    w = (1.0 - beta) / e
    w = w / w.sum() * n
    w[c == 0] = 0
    return w


def write_prediction_csv(path, P, L, S, IDX, classes, dataset):
    """Persist aligned per-frame probabilities for post-hoc paired analyses."""
    if dataset is None or not hasattr(dataset, "df"):
        raise ValueError("prediction export requires a dataframe-backed dataset")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    probability_columns = [f"p_{label}" for label in classes]
    fieldnames = [
        "dataset_index",
        "image_path",
        *(["stratum"] if dataset.sources is not None else []),
        "true_label_index",
        "true_label",
        "predicted_label_index",
        "predicted_label",
        "correct",
        "p_true",
        "confidence_margin",
        "nll",
        *probability_columns,
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for prediction, label, probabilities, index in zip(P, L, S, IDX):
            index = int(index)
            label = int(label)
            prediction = int(prediction)
            other_max = float(np.max(np.delete(probabilities, label)))
            p_true = float(probabilities[label])
            row = {
                "dataset_index": index,
                "image_path": str(dataset.df.iloc[index]["image_path"]),
                "true_label_index": label,
                "true_label": classes[label],
                "predicted_label_index": prediction,
                "predicted_label": classes[prediction],
                "correct": int(prediction == label),
                "p_true": p_true,
                "confidence_margin": p_true - other_max,
                "nll": float(-np.log(np.clip(p_true, 1e-12, 1.0))),
            }
            if dataset.sources is not None:
                row["stratum"] = str(dataset.sources[index])
            row.update(
                {
                    column: float(probability)
                    for column, probability in zip(
                        probability_columns, probabilities
                    )
                }
            )
            writer.writerow(row)
    return {
        "path": str(path),
        "rows": len(P),
        "sha256": file_sha256(path),
        "probability_columns": probability_columns,
    }


def save_trainable_checkpoint(path, model, seed, args):
    """Save only the learned head and LoRA tensors, not the frozen backbone."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if name.startswith("head.")
        or name.endswith(".lora_a")
        or name.endswith(".lora_b")
    }
    if not state:
        raise RuntimeError("no trainable head/LoRA tensors selected")
    torch.save(
        {
            "format": "cv2024_dinov2_lora_trainable_v1",
            "seed": int(seed),
            "backbone": args.backbone,
            "classes": list(CV2024_CLASSES),
            "state_dict": state,
        },
        path,
    )
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "tensor_count": len(state),
    }


@torch.no_grad()
def ev(
    m,
    loader,
    classes=CV2024_CLASSES,
    dataset=None,
    prediction_output=None,
):
    """Return per-source metrics when dataset.sources is available."""
    m.eval()
    P, L, S, IDX = [], [], [], []
    for img, l, idx in loader:
        img = img.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            lg = m(img)
        if not torch.isfinite(lg).all():
            raise RuntimeError("Nonfinite evaluation logits")
        prob = torch.softmax(lg.float(), dim=1).cpu().numpy()
        P.extend(prob.argmax(1).tolist())
        S.extend(prob.tolist())
        L.extend(l.tolist())
        IDX.extend(idx.tolist() if hasattr(idx, 'tolist') else idx)
    P, L = np.array(P), np.array(L)
    S = np.array(S, dtype=float)
    IDX = np.array(IDX)
    if not len(L):
        raise ValueError("Empty evaluation pool")
    mj = int(np.bincount(L).argmax()) if len(L) else 0
    per_class = {}
    for i, c in enumerate(classes):
        mask = (L == i)
        if mask.sum() > 0:
            per_class[c] = {
                "recall": float((P[mask] == i).mean()),
                "n": int(mask.sum()),
            }
        else:
            per_class[c] = {"recall": None, "n": 0}
    mean_auc, auc_mode, auc_classes = None, None, []
    true_class_prob = S[np.arange(len(L)), L] if len(L) else np.array([])
    mean_true_class_prob = (
        float(true_class_prob.mean()) if len(true_class_prob) else None
    )
    nll = (
        float(-np.log(np.clip(true_class_prob, 1e-12, 1.0)).mean())
        if len(true_class_prob)
        else None
    )
    present = [i for i in range(len(classes)) if np.any(L == i)]
    if len(present) >= 2:
        auc_vals = []
        for i in present:
            y_bin = (L == i).astype(int)
            if y_bin.min() == y_bin.max():
                continue
            auc_vals.append(float(roc_auc_score(y_bin, S[:, i])))
        if auc_vals:
            mean_auc = float(np.mean(auc_vals))
            auc_mode = "all_classes" if len(present) == len(classes) else "present_classes"
            auc_classes = [classes[i] for i in present]

    out = {
        "acc": float((P == L).mean()),
        "bal_acc": float(balanced_accuracy_score(L, P)),
        "f1_macro": float(f1_score(L, P, labels=present, average="macro", zero_division=0)),
        "f1_macro_class_set": [classes[i] for i in present],
        "mean_auc_ovr": mean_auc,
        "combined_ovr": None if mean_auc is None else float((balanced_accuracy_score(L, P) + mean_auc) / 2.0),
        "mean_auc_ovr_mode": auc_mode,
        "mean_auc_ovr_classes": auc_classes,
        "mean_true_class_prob": mean_true_class_prob,
        "nll": nll,
        "null_acc": float((np.full_like(L, mj) == L).mean()),
        "n_test": len(L),
        "per_class": per_class,
    }
    # Standard validation CSVs use source names. Diagnostic CSVs may instead
    # use the same column for explicit evaluation strata; retain the four
    # standard CV2024 source labels, including zero-count rows, while
    # reporting arbitrary stratum names when they are present.
    if dataset is not None and dataset.sources is not None:
        per_source = {}
        observed_sources = sorted(set(dataset.sources[IDX].tolist()))
        source_groups = (
            CV2024_SOURCES
            if set(observed_sources).issubset(set(CV2024_SOURCES))
            else observed_sources
        )
        for src in source_groups:
            src_mask = np.array([s == src for s in dataset.sources[IDX]])
            if src_mask.sum() == 0:
                per_source[src] = {"n": 0, "bal_acc": None, "acc": None, "per_class": {}}
                continue
            P_s = P[src_mask]
            L_s = L[src_mask]
            S_s = S[src_mask]
            if len(set(L_s)) < 2:
                # For one supported class, its recall equals accuracy.
                ba = float((P_s == L_s).mean())
            else:
                ba = float(balanced_accuracy_score(L_s, P_s))
            true_prob_s = S_s[np.arange(len(L_s)), L_s]
            pc_s = {}
            for i, c in enumerate(classes):
                cm = (L_s == i)
                if cm.sum() > 0:
                    pc_s[c] = {"recall": float((P_s[cm] == i).mean()), "n": int(cm.sum())}
            per_source[src] = {
                "n": int(src_mask.sum()),
                "bal_acc": ba,
                "acc": float((P_s == L_s).mean()),
                "mean_true_class_prob": float(true_prob_s.mean()),
                "nll": float(
                    -np.log(np.clip(true_prob_s, 1e-12, 1.0)).mean()
                ),
                "per_class": pc_s,
            }
        out["per_source"] = per_source
    if prediction_output is not None:
        out["per_frame_predictions"] = write_prediction_csv(
            prediction_output,
            P,
            L,
            S,
            IDX,
            classes,
            dataset,
        )
    return out


def train_one_seed(train_csv, seed, args, orig_val_csv, dedup_val_csv):
    set_determinism(seed)
    tr = CV2024DS(train_csv, tf=TF_TRAIN)
    n = len(tr)
    if n < args.batch:
        raise ValueError("Training pool smaller than drop-last batch: no optimizer steps")
    labels = torch.tensor([tr.classes.index(
        [c for c in tr.label_cols if tr.df.iloc[i][c] == 1][0])
        for i in range(n)])
    w = cbw(labels, len(CV2024_CLASSES)).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)

    g = torch.Generator()
    g.manual_seed(seed)
    ltr = DataLoader(tr, batch_size=args.batch, shuffle=True,
                     num_workers=6, pin_memory=True,
                     persistent_workers=True, drop_last=True,
                     worker_init_fn=worker_init_fn, generator=g)
    # Track sources for evaluation datasets
    orig_val = CV2024DS(orig_val_csv, tf=TF_EVAL)
    dedup_val = CV2024DS(dedup_val_csv, tf=TF_EVAL)
    l_ov = DataLoader(orig_val, batch_size=args.batch*2, shuffle=False, num_workers=6)
    l_dv = DataLoader(dedup_val, batch_size=args.batch*2, shuffle=False, num_workers=6)
    kv_test = None if args.skip_kvasir_eval else KvasirEvalDS()
    l_kt = (
        None
        if kv_test is None
        else DataLoader(
            kv_test,
            batch_size=args.batch * 2,
            shuffle=False,
            num_workers=6,
        )
    )
    print(
        f"  Train={n} OrigVal={len(orig_val)} DedupVal={len(dedup_val)} "
        f"KvasirTest={0 if kv_test is None else len(kv_test)}",
        flush=True,
    )

    model = DINOv2LoRA(
        n_cls=len(CV2024_CLASSES),
        backbone_name=args.backbone,
    ).to(DEVICE)
    if torch.cuda.device_count() > 1:
        m = nn.DataParallel(model)
    else:
        m = model

    lora = [p for wr in model.lora_wrappers for p in [wr.lora_a, wr.lora_b]]
    head_params = list(model.head.parameters())

    opt = torch.optim.AdamW(
        [{"params": head_params, "lr": 3e-4}],
        weight_decay=1e-4,
    )
    lora_added = False

    scaler = torch.amp.GradScaler("cuda")
    history = []

    for ep in range(args.epochs):
        if ep >= 3 and not lora_added:
            opt.add_param_group({"params": lora, "lr": 1e-4, "weight_decay": 1e-4})
            lora_added = True
            print(f"  [seed={seed} ep={ep+1}] LoRA params added to optimizer", flush=True)

        m.train()
        model.backbone.eval()

        losses = []
        optimizer_steps = skipped_steps = 0
        t0 = time.perf_counter()
        for step, (img, l, _) in enumerate(ltr):
            img = img.to(DEVICE, non_blocking=True)
            l = l.to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                lg = m(img)
                loss = loss_fn(lg, l)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at seed={seed} ep={ep+1} step={step}"
                )
            opt.zero_grad()
            scaler.scale(loss).backward()
            scale_before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            skipped = scaler.get_scale() < scale_before
            skipped_steps += int(skipped)
            optimizer_steps += int(not skipped)
            losses.append(loss.item())

        met_ov = ev(m, l_ov, dataset=orig_val)
        met_dv = ev(
            m,
            l_dv,
            dataset=dedup_val,
            prediction_output=(
                args.prediction_output
                if ep == args.epochs - 1
                else None
            ),
        )
        met_kt = None if l_kt is None else ev(m, l_kt)
        dt = time.perf_counter() - t0
        kt_text = "SKIP" if met_kt is None else f"{met_kt['bal_acc']:.4f}"
        print(f"  [seed={seed} ep={ep+1}] orig_val bal={met_ov['bal_acc']:.4f} "
              f"| dedup_val bal={met_dv['bal_acc']:.4f} "
              f"| kvasir_s1 bal={kt_text} "
              f"[{dt:.0f}s]",
              flush=True)
        history.append({
            "epoch": ep + 1,
            "loss": float(np.mean(losses)),
            "orig_val": met_ov,
            "dedup_val": met_dv,
            "kvasir_s1": met_kt,
            "train_s": float(dt),
            "optimizer_steps": optimizer_steps,
            "amp_skipped_steps": skipped_steps,
            "amp_scale": scaler.get_scale(),
        })
    checkpoint = None
    if args.trainable_checkpoint:
        checkpoint = save_trainable_checkpoint(
            args.trainable_checkpoint,
            model,
            seed,
            args,
        )
    return history, checkpoint


def main(ap=None, args=None):
    if ap is None:
        ap = build_parser()
    if args is None:
        args = ap.parse_args()
    if args.epochs < 1 or args.batch < 1:
        ap.error("epochs and batch must be positive")
    if (not args.seeds or len(set(args.seeds)) != len(args.seeds)
            or any(s < 0 or s >= 2**32 for s in args.seeds)):
        ap.error("seeds must be unique integers in [0, 2**32)")
    require_directory("DINOV2_REPO")
    require_directory("TORCH_HOME")
    if not args.skip_kvasir_eval:
        require_directory("KVASIR_ROOT")
        require_directory("KVASIR_SPLITS_DIR")
    if (args.prediction_output or args.trainable_checkpoint) and len(args.seeds) != 1:
        raise SystemExit(
            "Prediction/checkpoint export requires exactly one seed per process"
        )
    final_path = OUT / args.output
    partial_path = final_path.with_name(final_path.name + ".partial")
    tmp_path = final_path.with_name(final_path.name + ".tmp")
    outputs = [Path(p).resolve() for p in (final_path, partial_path, tmp_path, args.prediction_output, args.trainable_checkpoint) if p]
    if len(outputs) != len(set(outputs)):
        ap.error("Metrics, partial, temporary, prediction and checkpoint paths must be distinct")
    for candidate in outputs:
        if candidate and Path(candidate).exists():
            raise SystemExit(f"Refusing to replace prior output: {candidate}; use a fresh run directory")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    seed_count_match = re.search(r"_n(\d+)\.json$", Path(args.output).name)
    if seed_count_match:
        expected_seeds = int(seed_count_match.group(1))
        if len(args.seeds) != expected_seeds:
            raise SystemExit(
                f"Output filename {args.output} declares n={expected_seeds}, "
                f"but {len(args.seeds)} seeds were supplied: {args.seeds}. "
                "Use a matching output filename or provide the declared seed count."
            )

    if not args.no_batch128_assert and args.batch != 128:
        ap.error(f"Expected batch=128 (study configuration), got {args.batch}")
    print(f"Args: {vars(args)}", flush=True)

    script_path = os.path.abspath(__file__)
    meta = {
        "script_path": script_path,
        "script_sha256": file_sha256(script_path),
        "script_md5": file_md5(script_path),
        "script_version": "primary",
        "backbone": args.backbone,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "train_csv_md5": file_md5(args.train_csv),
        "orig_val_path": args.orig_val_csv,
        "orig_val_md5": (
            file_md5(args.orig_val_csv) if os.path.exists(args.orig_val_csv) else None
        ),
        "dedup_val_csv": args.dedup_val_csv,
        "dedup_val_csv_md5": (
            file_md5(args.dedup_val_csv) if os.path.exists(args.dedup_val_csv) else None
        ),
        "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "determinism_enabled": True,
        "determinism_warn_only": True,
        "bit_exact_reproduction_claimed": False,
        "dinov2_repo": os.environ.get("DINOV2_REPO"),
        "visible_cuda_devices": torch.cuda.device_count(),
        "batch": args.batch,
    }
    print(f"Provenance: script_sha256={meta['script_sha256'][:16]}...", flush=True)

    results = {"args": vars(args), "meta": meta, "runs": []}
    for seed in args.seeds:
        print(f"\n=== seed={seed} ===", flush=True)
        history, checkpoint = train_one_seed(
            args.train_csv,
            seed,
            args,
            args.orig_val_csv,
            args.dedup_val_csv,
        )
        results["runs"].append({
            "seed": seed,
            "history": history,
            "last": history[-1],
            "trainable_checkpoint": checkpoint,
        })
        with open(partial_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

    meta["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with open(tmp_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    os.replace(tmp_path, final_path)
    if partial_path.exists():
        partial_path.unlink()
    print(f"\nSaved {final_path}", flush=True)


if __name__ == "__main__":
    main(_cli_parser, _cli_args)
