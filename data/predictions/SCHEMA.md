# ReuScope predictions

Final-epoch per-image probabilities of the 110 primary DINOv2-L/14 fits and
the 60 frozen-backbone CNN fits in *ReuScope: Auditing Data Reuse and
Evaluation Sensitivity in Capsule Vision 2024*. From the repository root, with
the CPU requirements:

```bash
python recompute.py --output recomputed.json
```

`recomputed.json` must be a new file. No images, weights, GPU or PyTorch are needed. The command scores
every seed and view once with `analysis.metrics`, compares all 450 records
field by field (counts, recalls, BA, accuracy, AUC, combined score; absolute
tolerance 1e-10) with `data/results/numerical/seed_metrics.json.gz`,
then rebuilds the 45 mean cells and the paired BA contrasts of Table 1(b)
with `analysis.paired_intervals`.

## Layout

Paths in `manifest.json` are relative to `data/`. `manifest.json` lists each file's
family, backbone, training pool (`arm`), seed, view, row count and dtype.
Paths are `predictions/{family}/{backbone}/{arm}/seed{seed}_{view}.npz`.

Each NPZ (compressed, no pickled objects) contains:

| Array | Meaning |
|---|---|
| `probabilities` | N × 10 class probabilities: float64 (primary), float32 (CNN) |
| `row_id` | Validation row IDs, in order, matching `inputs/validation.csv.gz` |
| `labels` | Integer class labels 0–9 |
| `source` | Source of each row (`Dataset` column of the validation table) |
| `classes` | The ten class names below, in column order |
| `seed`, `arm` | Training seed and pool name |

Classes: Angioectasia, Bleeding, Erosion, Erythema, Foreign Body,
Lymphangiectasia, Normal, Polyp, Ulcer, Worms.

- **Primary**: 11 pools (Full, Random, Source-only, Removal, Class-match-R,
  Comp, A, B, C, D-dup, D-uniq) × seeds 0–9 = 110 files of 16,132 rows. The
  Non-KV (4,551) and Filtered (4,326) views select rows of these same arrays
  through `inputs/evaluation_memberships.csv.gz`: 330 seed/view records.
- **CNN**: ResNet-50 and ConvNeXt-Tiny × Full, Random, Source-only × seeds
  0–9 = 60 fits. Each fit has two arrays written by its own evaluation passes,
  `public` (16,132 rows) and `non_kvasir` (4,551 rows): 120 files.

## Values

These are the saved outputs of the reported runs, not new inference. Primary
probabilities were exported by the training runs as decimal CSV text; each
value is the exact float64 parse of that text. CNN probabilities are the exact
float32 tensors saved by the runs. All 30,155,000 values, labels and row orders
were compared with those saved outputs before packaging; no normalization,
clipping or rounding was applied. Logits and machine-specific paths are omitted.

## Coverage

| Paper result | Covered here |
|---|---|
| Table 1(a,b), Figure 3, Tables 18–20, 22, 23 | Per-seed metrics recomputed from these arrays |
| Table 21 (per-source) | Computable from `probabilities`, `labels` and `source` |
| Table 1(c), Table 16 (official test) | Not here: the producer saved no per-image outputs; `data/results` has confusion matrices and class AUCs |
| Table 1(d), Tables 13–15 (participant entries) | Not here: `rescore.py` reads the upstream workbooks |
| Table 1(e), Table 2, Tables 27–28 (local removal) | Not here: `data/results/numerical/local_probabilities.npz` |

The seed intervals are conditional on the fixed lists and pipeline; the
paper's selection and inference limitations apply.

## License and attribution

Predictions: CC BY 4.0, Eunseob Choi, Kyeonghun Kim, Hyuk-Jae Lee and
Nam-Joon Kim, ReuScope (2026). Row metadata derive from Handa et al.,
Training and Validation Dataset of Capsule Vision 2024 Challenge, v3
(doi:10.6084/m9.figshare.26403469.v3); provider terms and
`data/results/LICENSE.txt` apply.
