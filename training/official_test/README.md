# Official AIIMS test runs

Producer of Table 1(c) and Table 16: DINOv2-L/14 head/LoRA training on Full,
Random and Source-only (10,596 rows) × seeds 0–9 = 30 fits, each evaluated on
the 4,385-image official test release.

| File | Role |
|---|---|
| `producer.py` | Archived producer, byte-identical (SHA-256 `f834a5bcce8fdf5e8462af690df3d1d0b30d6ca04cb01fdd737749b449a1cfa1`); all 30 result records name this hash |
| `run.py` | Entry point: sets the package root, refuses existing outputs, loads Torch Hub `dinov2_vitl14` from a local checkout |
| `src/dataset_official.py` | Kvasir-Capsule loader dependency; default data roots changed to relative paths (the producer sets them from the environment); docstring claims about video-disjoint folds removed |
| `SOURCE.json` | Per-run arguments, GPU model, library versions and input-file MD5s from the 30 records |

The producer saves metrics only. The original runs therefore left no model
checkpoints or per-image test predictions; the release provides their
confusion matrices and class AUCs (`data/results`) and this code.

## Requirements

- Images: CV2024 training/validation release (`CV2024_ROOT` = its `Dataset`
  folder) and the official test release (`--official_test_dir`, the folder of
  ten class-named subfolders; 4,385 images).
- Kvasir-Capsule labelled images and official `split_1.csv` (`KVASIR_ROOT`,
  `KVASIR_SPLITS_DIR`): the producer evaluates this split once, after the final
  epoch. Reported results do not use that evaluation, but the producer requires it.
- DINOv2: a local checkout of [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)
  (`DINOV2_REPO`) with `dinov2_vitl14` weights in the Torch Hub cache
  (`TORCH_HOME`). The original runs loaded the unpinned Torch Hub default; its
  revision and weight digest were not recorded.
- Environment recorded per run: PyTorch 2.6.0+cu124, CUDA 12.4, cuDNN 9.1;
  GPU model per run in `SOURCE.json` (L40S 28, A100-SXM4-80GB 1, RTX A6000 1).
  Other package versions were not recorded; `training/requirements.txt` lists
  the closest recorded environment. `openpyxl` is needed only for XLSX lists.

## GPU count

`producer.py` wraps the model in `nn.DataParallel` when more than one GPU is
visible. The original launcher requested **two A100 GPUs for Full seed 0 and
one GPU for every other run**; the result records name only device 0, so the
number of visible devices was not recorded. With two devices, each batch is
split across them: the objective is unchanged, but dropout masks and
floating-point reduction order differ from a one-GPU run, and no setting is
guaranteed to repeat a run bitwise. The commands below follow the launcher
request; running every fit on one GPU follows the same recipe.

## Train and evaluate

Run from this directory. `LISTS` is the output of `training/prepare.py`,
`OFFICIAL_TEST` the official test folder and `OUT` a new absolute directory.

```bash
export CV2024_ROOT=/path/to/Dataset KVASIR_ROOT=/path/to/labelled_images \
       KVASIR_SPLITS_DIR=/path/to/official_splits DINOV2_REPO=/path/to/dinov2 \
       TORCH_HOME=/path/to/torch_cache
LISTS=/path/to/lists OFFICIAL_TEST=/path/to/official_test OUT=/path/to/new_output
for arm in Full Random Source-only; do
  for seed in 0 1 2 3 4 5 6 7 8 9; do
    gpus=0; [ "$arm" = Full ] && [ "$seed" = 0 ] && gpus=0,1
    CUDA_VISIBLE_DEVICES=$gpus python run.py \
      --train_csv "$LISTS/primary/$arm.csv" \
      --orig_val_csv "$LISTS/views/Public.csv" \
      --dedup_val_csv "$LISTS/views/Non-KV.csv" \
      --official_test_dir "$OFFICIAL_TEST" \
      --epochs 10 --batch 128 --workers 6 --seeds "$seed" \
      --output "$OUT/${arm}_s${seed}.json"
  done
done
```

The original runs read the organizer workbooks for Full and Public validation;
the prepared CSVs contain the same rows in the same order. In `SOURCE.json`,
the input MD5s identify the original files, and the arm labels `baseline`,
`random10596_s0` and `le6` denote Full, Random and Source-only. Leave `--eval_each_epoch` off, as in all 30 runs:
enabling it adds evaluations before later epochs and changes the run.

## Recipe

- Input: RGB, Resize((224, 224)), ImageNet mean/std; no augmentation.
- Model: frozen DINOv2-L/14 in evaluation mode; rank-8 LoRA (α = 16) on qkv of
  the last four blocks; head LayerNorm(1024) → Linear(256) → GELU →
  Dropout(0.1) → Linear(10).
- Training: ten epochs, batch 128, six workers, seeded shuffle, drop-last;
  head-only epochs 1–3, LoRA parameters join AdamW at epoch 4 (head lr 3e-4,
  LoRA lr 1e-4, weight decay 1e-4, no schedule); class weights β = 0.999;
  label smoothing 0.05; FP16 autocast with gradient scaling.
- Evaluation: official test after every epoch; Public, Non-KV and Kvasir
  after epoch 10 only. FP16 autocast, FP32 softmax. Epoch 10 is reported.
- Metrics: BA = mean class recall; AUC = mean one-vs-rest AUC over defined
  classes; combined = (BA + AUC) / 2.

## Output

One JSON per invocation with `args`, `meta`, `runs` and `summary`.
`runs[0].last.official_test` holds BA, AUC, combined score, accuracy, class
recalls, class AUCs and the confusion matrix. `summary.*_sd` is a sample SD;
the paper reports SE = SD/√10 across seeds. Output metadata contain local
paths. `replay.py` aggregates the released confusion matrices and class AUCs
of the original 30 runs (`../../REPRODUCE.md`).
