# DINOv2-L/14 primary and local-removal runs

Producers of the DINOv2 results in Table 1(a,b,e), Table 2, Figure 3 and
Tables 18–22 and 27–28: `primary.py` (11 pools × seeds 0–9 = 110 fits) and
`local.py` (local removal and three distant draws × 10 seeds = 40 fits), sharing
`dinov2_training.py`. Model, loss, optimizer, seed handling, evaluation order and
the epoch-10 endpoint follow the archived producers of these runs; a CPU
comparison on reduced inputs matched all trainable tensors, RNG states, losses
and epoch metrics over four epochs. Changes concern local paths, validated CSV
inputs, output safety and checkpoint restoration, so the files are not
byte-identical to the archived producers; `SOURCE.json` lists their hashes and
the changes.

## Inputs

Set `CV2024_ROOT` to the extracted `Dataset` directory, `DINOV2_REPO` to a local
[DINOv2 source checkout](https://github.com/facebookresearch/dinov2), and
`TORCH_HOME` to its pretrained-weight cache. Training calls the local hub's
pretrained loader; a missing weight file may trigger that loader's download.
Provide weights separately. No model code, images or weights are bundled here.
The exact DINOv2 Git revision and per-fit frozen-weight digest of the paper runs
were not recorded; a newly selected checkout cannot be called their original revision.
Keep the same source and frozen weights for training and restoration.

The primary campaign recorded PyTorch 2.6.0+cu124, CUDA 12.4, cuDNN 90100 and
A100 80 GB. Contemporaneous primary probes recorded TorchVision 0.21.0+cu124 and
Pillow 12.1.1. Local records establish PyTorch 2.6.0+cu124, CUDA 12.4 and cuDNN
90100, with one A100 80 GB, RTX A6000 or L40S per seed (all four arms of a seed
on the same GPU model); a complete per-fit dependency snapshot is unavailable. NumPy, pandas,
scikit-learn, Pillow and TorchVision are also required. CLI `--help` needs only
Python; model execution needs the training dependencies.

Use ordered CSVs with this header:

```text
image_path,Dataset,Angioectasia,Bleeding,Erosion,Erythema,Foreign Body,Lymphangiectasia,Normal,Polyp,Ulcer,Worms
```

Each row selects exactly one class with binary one-hot values. Class indices use
the header's class order above, independently of supplied column ordering.
Duplicate training rows are retained. Paths may be relative to `CV2024_ROOT`;
both slash conventions are supported. Valid subsets of class columns are accepted.
For source metrics `Dataset` contains `KVASIR`, `SEE-AI`, `KID` or `AIIMS`.
For the **local paired evaluation**, it instead contains `NCC99_EXPOSED` or
`MATCHED_UNFLAGGED`: 588 rows each, 1,176 total. Do not replace these strata with
source names. A missing `Dataset` column disables stratified metrics.

The local runs read public validation from the organizer workbook;
`views/Public.csv` holds the same rows in the same order. **Retain both
validation passes every epoch**, even when only paired-evaluation predictions
are needed: DataLoader iteration consumes RNG state. No split is created here.
Image-path disjointness does not establish video or patient independence.

## Run recipes

`LISTS` is the output of `training/prepare.py`; `OUT` is a new absolute
directory. From the repository root, with one visible GPU as in the paper runs:

```bash
export CV2024_ROOT=/path/to/Dataset DINOV2_REPO=/path/to/dinov2 TORCH_HOME=/path/to/torch_cache
export CUDA_VISIBLE_DEVICES=0
for POOL in Full Random Source-only Removal Class-match-R Comp A B C D-dup D-uniq; do
  for SEED in 0 1 2 3 4 5 6 7 8 9; do
    RUN="$OUT/primary/$POOL/seed$SEED"; mkdir -p "$RUN"
    python training/dinov2/primary.py \
      --train_csv "$LISTS/primary/$POOL.csv" \
      --orig_val_csv "$LISTS/views/Public.csv" --dedup_val_csv "$LISTS/views/Public.csv" \
      --seeds "$SEED" --epochs 10 --batch 128 --backbone dinov2_vitl14 --skip_kvasir_eval \
      --output "$RUN/metrics.json" --prediction_output "$RUN/predictions.csv" \
      --trainable_checkpoint "$RUN/trainable.pt" > "$RUN/log.txt" 2>&1
  done
done
for ARM in local draw101 draw202 draw303; do
  for SEED in 14 15 16 17 4 5 6 7 8 9; do
    RUN="$OUT/local/$ARM/seed$SEED"; mkdir -p "$RUN"
    python training/dinov2/local.py \
      --train_csv "$LISTS/local/$ARM.csv" \
      --orig_val_csv "$LISTS/views/Public.csv" --dedup_val_csv "$LISTS/local/Evaluation.csv" \
      --seeds "$SEED" --epochs 10 --batch 128 --backbone dinov2_vitl14 --skip_kvasir_eval \
      --output "$RUN/metrics.json" --prediction_output "$RUN/predictions.csv" \
      --trainable_checkpoint "$RUN/trainable.pt" > "$RUN/log.txt" 2>&1
  done
done
```

Run primary seeds 0–9 for each of the eleven pools. Local runs use seeds
4–9 and 14–17; the released local arrays store them in the order
**14,15,16,17,4,5,6,7,8,9**. Each local training list has 35,631 rows. Use one seed
per process when exporting predictions or a checkpoint. Relative metrics paths resolve
under `$REUSCOPE_ROOT/results` (working directory if unset); prediction and
checkpoint paths resolve from the working directory. Existing outputs are rejected.

`--dedup_val_csv` is an input name only: it does not deduplicate or filter rows.
Primary exports score all 16,132 public-validation rows; obtain the 4,551-row
non-KVASIR and 4,326-row filtered views from these same predictions. Both paper
campaigns set `--skip_kvasir_eval`. The optional Kvasir evaluator additionally
needs `KVASIR_ROOT` and `KVASIR_SPLITS_DIR` and changes the run's RNG trajectory.

## Fixed numerical recipe

- RGB PIL images; direct `Resize((224,224))`, `ToTensor`, ImageNet normalization
  (mean .485/.456/.406; SD .229/.224/.225). Same train/eval transform; no random
  augmentation. Resize interpolation/antialias use the installed TorchVision defaults.
- DINOv2-L/14, 1,024-dimensional output; frozen backbone. Rank-8 LoRA on fused
  attention qkv in the last four blocks, alpha 16 (scale 2), A initialized with
  `randn * .01`, B zero. LoRA aliases are retained in the state dictionary.
- Head: LayerNorm(1024), Linear(1024,256), GELU, Dropout(.1), Linear(256,10).
  `model.train()` is followed by `backbone.eval()` each training epoch; the head
  trains while the backbone's stochastic layers remain in evaluation mode.
- Class-balanced weights: `(1-beta)/(1-beta**count)`, beta .999; normalize weights
  to sum to ten **before** setting absent-class weights to zero. Cross-entropy
  label smoothing .05; mean reduction.
- Persistent AdamW: head LR .0003, weight decay .0001; add LoRA group at zero-based
  epoch 3 (epoch 4), LR .0001, weight decay .0001. AdamW defaults include betas
  (.9,.999), eps 1e-8, amsgrad false. No scheduler, clipping or accumulation.
  LoRA gradients remain enabled during warm-up exactly as in the producers;
  those parameters are not in the optimizer until epoch 4.
- Ten epochs, batch 128, shuffled seeded generator, six workers, pinned memory,
  persistent training workers, drop-last. Evaluation batch 256, six workers,
  no shuffle. CUDA float16 autocast and GradScaler. DataParallel if multiple
  devices are visible; the primary paper records used one visible device.
- Seed Python, NumPy, Torch and CUDA; worker seed is
  `(torch.initial_seed()+worker_id) % 2**32`. cuDNN deterministic true, benchmark
  false; deterministic algorithms warn-only; CUBLAS workspace default `:4096:8`.
  These settings do not guarantee bitwise-identical results, even on identical hardware.
- Final epoch, not best epoch, supplies reported predictions/checkpoints.
  Evaluation uses `eval()` and `no_grad()`, FP32 softmax and float64 NumPy scores.
  Primary macro-F1 uses represented true classes; local macro-F1 uses the union
  of true/predicted classes. BA is mean recall over represented true classes;
  AUC is mean valid one-versus-rest AUC. Combined score is (BA+AUC)/2.

## Outputs and checkpoint inference

Metrics JSON contains `args`, runtime `meta`, and `runs`, each with `seed`, full
`history`, final `last`, and checkpoint metadata. The checkpoint contains learned
head/LoRA tensors, class order, backbone name, seed and recipe. It excludes frozen
weights and optimizer/scaler state: it supports final inference, not training resume.
Strict loading checks keys, shapes, dtypes, finite values and duplicate LoRA aliases.

```bash
python training/dinov2/predict.py \
  --checkpoint "$RUN/trainable.pt" \
  --backbone-weights "$TORCH_HOME/hub/checkpoints/dinov2_vitl14_pretrain.pth" \
  --input-csv "$LISTS/views/Public.csv" --recipe primary \
  --output "$RUN/restored_predictions.csv" --batch 256
```

Restoration constructs the local backbone with `pretrained=False` and loads the
explicit frozen weights strictly (the Torch Hub file `dinov2_vitl14_pretrain.pth`). Pass `--recipe primary` or `--recipe local`
for checkpoints that do not record their recipe; the CLI does not guess. Use the training evaluation batch size for
comparison. CPU inference differs from CUDA AMP and is not a paper-result replay.

Prediction fields, in order:

```text
dataset_index,image_path,stratum,true_label_index,true_label,predicted_label_index,predicted_label,correct,p_true,confidence_margin,nll,p_Angioectasia,p_Bleeding,p_Erosion,p_Erythema,p_Foreign Body,p_Lymphangiectasia,p_Normal,p_Polyp,p_Ulcer,p_Worms
```

`dataset_index` is the zero-based input row; `stratum` copies `Dataset` verbatim
(or is empty). `p_true` is true-class probability; `confidence_margin` is true-class
probability minus the largest other-class probability; `nll=-log(clip(p_true,1e-12,1))`.
Preserve row alignment and all ten probabilities when mapping to public arrays.
The saved predictions of the reported runs are released separately; see
`../../REPRODUCE.md` for the scoring commands. Checkpoints of the reported runs
are not distributed.
