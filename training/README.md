# Training and inference

Code that trained the primary, local-removal, CNN and official-test models.
Each subdirectory documents its recipe, inputs, recorded environment and outputs.

| Directory | Runs | Paper results |
|---|---|---|
| [`dinov2/`](dinov2/) | Primary: 11 pools × seeds 0–9 (110 fits). Local removal: 4 arms × 10 seeds (40 fits) | Table 1(a,b,e), Table 2, Figure 3, Tables 18–22, 27–28 |
| [`cnn/`](cnn/) | ResNet-50 and ConvNeXt-Tiny × Full / Source-only / Random × seeds 0–9 (60 fits) | Table 1(b) CNN rows, Table 23 |
| [`official_test/`](official_test/) | DINOv2-L/14 × Full / Random / Source-only × seeds 0–9 (30 fits) | Table 1(c), Table 16 |

`official_test/producer.py` is the archived producer byte for byte. The CNN
trainer keeps the archived executable code; the DINOv2 recipes and the CNN
runner follow their archived producers with path, input-validation and
output-safety changes. `SOURCE.json` files and the subdirectory READMEs state
what changed.

## Inputs

All runs read ordered CSV lists generated from `data/results`:

```bash
python training/prepare.py lists
(cd lists && sha256sum -c ../training/lists.sha256)   # macOS: shasum -a 256 -c
```

`prepare.py` writes 19 files: `primary/` (Full, Random, Source-only, Removal,
Class-match-R, Comp, A, B, C, D-dup, D-uniq), `views/` (Public, Non-KV,
Filtered) and `local/` (local, draw101, draw202, draw303, Evaluation). Rows
keep the original order and repeated occurrences; paths are relative to the
CV2024 `Dataset` folder. The lists equal the input files of the reported runs
row for row; `lists.sha256` pins the exported bytes. In `local/Evaluation.csv`
the `Dataset` column holds the exposure stratum (`NCC99_EXPOSED` /
`MATCHED_UNFLAGGED`, 588 rows each) that the local trainer reads; all its
images are KVASIR validation frames.

Images and pretrained weights are obtained from their providers (see the main
README). Model training needs a CUDA GPU: primary and CNN runs recorded one
visible A100 80 GB per fit, local runs were configured with one A100 80 GB,
RTX A6000 48 GB or L40S 48 GB, and the official-test launcher requested one GPU
per run (two A100s for Full seed 0). `requirements.txt` lists the recorded training environment; it is
separate from the CPU replay requirements in the repository root.

## Reproducibility scope

Seeds, data order and deterministic cuDNN settings are fixed, but mixed
precision and GPU kernels do not guarantee bitwise-identical results, even on
identical hardware. The DINOv2 Torch Hub revision and frozen-weight digest of the
original runs were not recorded. The saved predictions and metrics of the
reported runs are released (`../REPRODUCE.md`); their checkpoints are not.
Code for the auxiliary CV2024 training analyses (Tables 24–26) and the
Kvasir-Capsule experiments (Tables 29–31) is not part of this release.
