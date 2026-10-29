# Frozen-backbone CNN runs

Producer of the ResNet-50 and ConvNeXt-Tiny rows in Table 1(b) and Table 23:
two backbones × Full / Source-only / Random × seeds 0–9 = 60 fits.

| File | Role |
|---|---|
| `src/trainer.py` | Archived trainer; executable code unchanged, docstrings revised |
| `src/dataset_official.py`, `src/utils/dataset_kvasir_official.py` | Archived loader dependencies, byte-identical |
| `run.py` | Plans inputs and runs one backbone/pool/seed cell through the trainer |
| `predict.py` | Scores a CSV with a saved `classifier.pt` |
| `SOURCE.json` | Archived and published file hashes, recorded environment, original list and weight hashes |

`run.py` keeps the archived observer functions (`inventory`, `observe`,
`snapshot`, `restore`, `pinned_builder`) unchanged; its changes are
package-relative paths and a command line limited to the reported runs.

## Requirements

- Images: the CV2024 training/validation release; set `CV2024_ROOT` to its
  `Dataset` folder.
- Lists: `$LISTS` from `python training/prepare.py` (see `training/README.md`).
  CLI arm names map to `baseline` = Full (37,607 rows), `le6` = Source-only
  (10,596 rows; not the 10,591-row Removal pool), `random` = Random (10,596 rows).
- Weights: TorchVision `resnet50-11ad3fa6.pth` (`IMAGENET1K_V2`) and
  `convnext_tiny-983f1562.pth` (`IMAGENET1K_V1`), downloaded separately; `plan`
  and `predict.py` check their full SHA-256 values from `SOURCE.json`.
- Environment recorded for the runs: Python 3.10.18, PyTorch 2.6.0+cu124,
  TorchVision 0.21.0+cu124, CUDA 12.4, cuDNN 9.1, NumPy 2.2.6, pandas 2.3.3,
  scikit-learn 1.7.2, Pillow 12.1.1, one A100-SXM4-80GB per seed
  (`training/requirements.txt`). `--help` needs only Python.

## Train

Run from this directory. Outputs are never overwritten; use fresh paths.

```bash
export CV2024_ROOT=/path/to/Dataset
python run.py plan --output plan.json \
  --baseline "$LISTS/primary/Full.csv" \
  --le6 "$LISTS/primary/Source-only.csv" \
  --random "$LISTS/primary/Random.csv" \
  --orig-val "$LISTS/views/Public.csv" \
  --dedup-val "$LISTS/views/Non-KV.csv" \
  --resnet50-weights "$WEIGHTS/resnet50-11ad3fa6.pth" \
  --convnext-tiny-weights "$WEIGHTS/convnext_tiny-983f1562.pth" \
  --skip-kvasir-eval --rig gpu0
for seed in 0 1 2 3 4 5 6 7 8 9; do
  for backbone in resnet50 convnext_tiny; do
    for arm in baseline le6 random; do
      CUDA_VISIBLE_DEVICES=0 python run.py run --plan plan.json --rig gpu0 \
        --backbone "$backbone" --arm "$arm" --seed "$seed" \
        --output "runs/${backbone}_${arm}_${seed}"
    done
  done
done
```

The plan hashes every list, image and weight file; `run` refuses stale inputs,
changed sources or a different runtime, and requires exactly one visible GPU.
In the reported runs, all six cells of a seed used the same physical GPU; use
one `--rig` label per GPU and a separate plan for another GPU.
`--skip-kvasir-eval` is part of the reported recipe (no Kvasir images needed).
`--smoke-epochs 1|2` shortens the schedule for plumbing checks only.

## Recipe

- Input: RGB, Resize(256, bilinear, antialias), CenterCrop(224), ImageNet
  mean/std; no augmentation.
- Model: frozen backbone held in evaluation mode (BatchNorm buffers fixed,
  stochastic depth inactive); head LayerNorm → Linear(256) → GELU →
  Dropout(0.1) → Linear(10).
- Training: ten epochs, batch 128, six workers, seeded shuffle, drop-last;
  AdamW lr 3e-4, weight decay 1e-4; effective-number class weights (β = 0.999);
  label smoothing 0.05; cosine schedule over all minibatches, advanced after
  each successful mixed-precision update. FP16 autocast (ResNet-50), BF16
  (ConvNeXt-Tiny).
- Evaluation: FP32, batch 256, Public and Non-KV every epoch; the epoch-10
  model is reported. BA is mean recall over represented classes.
- Determinism: seeded, warn-only deterministic algorithms; bitwise-identical
  results are not guaranteed, even on identical hardware.

## Outputs and inference

Each run directory holds `binding.json`, `complete.json` (history, AMP step
trace, runtime, peak CUDA memory), `orig_val.pt` / `dedup_val.pt` (final
logits, predictions and probabilities for Public / Non-KV) and `classifier.pt`
(trainable head and all backbone buffers; frozen weights come from the
pinned files). Plans and run files contain local paths; sanitize them before
sharing. `classifier.pt` supports inference, not training resumption.

```bash
python predict.py --checkpoint runs/resnet50_baseline_0/classifier.pt \
  --backbone-weights "$WEIGHTS/resnet50-11ad3fa6.pth" \
  --input-csv "$LISTS/views/Public.csv" --output predictions.pt
```

`predict.py` validates the weight hash and every checkpoint tensor and writes
row indices, labels, logits and probabilities in input order. The paper's
saved CNN probabilities are in `data/predictions` (`../../REPRODUCE.md`);
checkpoints of the reported runs are not distributed.
