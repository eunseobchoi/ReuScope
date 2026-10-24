# Training

This directory contains DINOv2 training and fixed-list CNN comparison code.
Datasets, split CSVs, weights and run outputs are not included. Supply your own
training and evaluation lists; the scripts do not construct splits.

Use Python 3.12 and the root requirements. The training configuration uses
PyTorch 2.6.0 and TorchVision 0.21.0 with CUDA 12.4; obtain matching builds from
the [official installation instructions](https://pytorch.org/get-started/previous-versions/).
Run commands from `training/` with a suitable GPU.

## External inputs

- Set `CV2024_ROOT` to your CV2024 `Dataset` directory. CSV `image_path` values
  may be absolute or relative to this directory, with either slash convention.
  Absolute paths must be valid on the machine running the scripts.
  Recognized Kvasir prefixes in DINOv2 lists require `KVASIR_ROOT` as well.
- Each CSV needs nonempty `image_path` values and binary one-hot class columns:
  `Angioectasia`, `Bleeding`, `Erosion`, `Erythema`, `Foreign Body`,
  `Lymphangiectasia`, `Normal`, `Polyp`, `Ulcer`, `Worms`. Present-class subsets
  are supported. An optional `Dataset` column supplies nonblank source names for
  DINOv2 evaluation; omit it when source metadata is unavailable.
  Preserve intended row order and duplicate rows. Training requires at least one
  full batch because the loaders use `drop_last=True`.
- DINOv2 requires `DINOV2_REPO` pointing to a separately obtained local DINOv2
  repository and `TORCH_HOME` pointing to an existing Torch cache directory.
  Populate its `hub/checkpoints/` with the pretrained weights for the selected
  backbone. The local repository's pretrained loader may download missing weights;
  prepopulate the cache for offline use. The default backbone is `dinov2_vitl14`.
- CNN plans require explicit local ResNet-50 and ConvNeXt-Tiny weight files.
  [crossmodel_weight_hashes.json](crossmodel_weight_hashes.json) pins their public
  filenames and full SHA256 hashes. Obtain these weights separately.
- To enable external Kvasir evaluation, set `KVASIR_ROOT` to `labelled_images/`
  and `KVASIR_SPLITS_DIR` to a directory containing externally obtained official
  `split_1.csv` (`filename,label` columns). Omit the skip flag. Frame-list folds
  do not establish video-disjoint evaluation.

## DINOv2

```bash
export CV2024_ROOT=/path/to/CV2024/Dataset
export DINOV2_REPO=/path/to/dinov2
export TORCH_HOME=/path/to/torch-cache
python3 src/primary.py \
  --train_csv /path/to/lists/train.csv \
  --orig_val_csv /path/to/lists/original_validation.csv \
  --dedup_val_csv /path/to/lists/evaluation.csv \
  --seeds 0 --epochs 10 --batch 128 --skip_kvasir_eval \
  --output /path/to/new-primary-run/metrics.json \
  --prediction_output /path/to/new-primary-run/predictions.csv \
  --trainable_checkpoint /path/to/new-primary-run/trainable.pt
```

For local-versus-distant confidence comparisons, run `src/confidence.py`
with the same flags and the corresponding training and evaluation lists, using
fresh output paths. Both DINOv2 trainers use the same classifier and LoRA
optimization recipe; the confidence trainer does not introduce a separate
confidence loss. It uses union-label macro-F1; primary and CNN evaluation use
supported true labels. Keep each recipe's training and evaluation pools explicit.

The study configuration uses batch size 128 and ten epochs. A different batch
size requires the explicit `--no_batch128_assert` override and changes the
training recipe. Prediction exports score the rows in `--dedup_val_csv` in
their supplied order; that filename does not perform any filtering.

Prediction and checkpoint exports require exactly one seed per invocation.
Metrics, partial, temporary, prediction and checkpoint paths must be distinct
and unused. Relative metrics paths go under `training/results/`; relative
prediction/checkpoint paths use the working directory. `CAPSULE_ROOT` overrides
that results root. Avoid concurrent runs sharing output paths.

## CNN comparisons

```bash
export CV2024_ROOT=/path/to/CV2024/Dataset
python3 crossmodel.py plan --output /path/to/cnn_plan.json \
  --rig local_gpu0 --seeds 0 --batch 128 \
  --baseline /path/to/lists/baseline.csv \
  --source-only /path/to/lists/source_only.csv --random /path/to/lists/random.csv \
  --orig-val /path/to/lists/original_validation.csv \
  --dedup-val /path/to/lists/evaluation.csv --skip-kvasir-eval \
  --resnet50-weights /path/to/resnet50-11ad3fa6.pth \
  --convnext-tiny-weights /path/to/convnext_tiny-983f1562.pth
CUDA_VISIBLE_DEVICES=0 python3 crossmodel.py run \
  --plan /path/to/cnn_plan.json --rig local_gpu0 \
  --backbone resnet50 --variant fixed --arm baseline --seed 0 \
  --output /path/to/new-cnn-run
```

The plan parent directory must exist. Planning reads and hashes supplied lists,
images and weights without decoding images. It binds the local code, weight
configuration and runtime versions. Runs check this binding and input contents,
require exactly one explicit visible GPU, and create a new output directory.
Create a new plan after changing source files or package versions.
A full run uses 10 epochs. Repeat for both backbones and all three arms for each
seed, keeping the same physical GPU within each paired comparison.

The arms are `baseline` (full training), `source_only` (exclude KVASIR training
rows) and `random` (a random subset matched to the source-only pool size).
Construct these lists before planning. `--rig` is a user-chosen identifier for
the shared physical GPU; `--variant fixed` selects the frozen-backbone recipe.
For a short execution check, `run --smoke-epochs 1` or `2` shortens training and
the scheduler horizon. Such runs are excluded from scientific summaries.

`python3 crossmodel.py summarize --plan /path/to/cnn_plan.json --runs <run directories>
--output /path/to/summary.json` reports arm-minus-baseline contrasts and missing
comparisons. Plans and summaries refuse existing output files.

The lower-level `src/cross_model.py` CLI requires all three CSV arguments,
`CV2024_ROOT`, `KVASIR_ROOT`, `KVASIR_SPLITS_DIR` and `TORCH_HOME`; its DINOv2
choices also require `DINOV2_REPO`. It always evaluates Kvasir and uses pretrained
loaders that may download missing weights. Use `crossmodel.py` for local pinned
CNN weights, optional Kvasir evaluation and prediction exports.
`CAPSULE_ARTIFACT_ROOT` overrides the lower-level trainer's results root.

## CPU interface checks

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python3 -S check_cpu.py
```

This stdlib-only check exercises help, required list arguments, confidence input
and output validation, and path handling with synthetic fixtures. It does not
train, decode images, download weights or verify GPU execution.
