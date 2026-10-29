<div align="center">

# ReuScope

### Auditing Data Reuse and Evaluation Sensitivity<br>in Capsule Vision 2024

Eunseob Choi · Kyeonghun Kim · Hyuk-Jae Lee · Nam-Joon Kim

Gwangju Institute of Science and Technology · OUTTA · Seoul National University

**NeurIPS 2026 · Evaluations and Datasets**

[Paper](https://openreview.net/forum?id=69GCzBgr1S) · [Data](data/) · [Quickstart](#replay-the-results) · [Reproduce](REPRODUCE.md) · [Citation](#citation)

</div>

ReuScope connects measurements of data reuse to controlled comparisons of training
and evaluation populations. This repository provides the audit code, training and
inference code for the primary, local-removal, CNN and official-test models, and
the numerical evidence behind the paper's results in [`data/`](data/).

## Replay the results

Use Python 3.12. From the repository root:

```bash
pip install -r requirements.txt
python replay.py --output replay.json
```

Replay reads the retained records in [`data/results`](data/results). It runs on CPU without images, model
weights or network access and writes to a new output file. It checks retained numerical targets at full precision and SSIM intervals at
the paper's three-decimal display precision. A separate
synthetic usage example is available with `python example.py`.

| Evidence | What replay computes |
|---|---|
| Ordered primary pools, evaluation views and local-removal lists | Pool/view counts and local evaluation alignment; removal lists are supplied for inspection |
| Byte-identity, video-prefix, hash and pair measurements | Cross-split counts, selected/any-counterpart counts and SSIM summaries with joint-prefix intervals |
| 110 primary and 60 CNN runs: saved seed-level metrics | Main means and reported paired training contrasts (Table 1a–b) |
| 30 official-test runs: confusion matrices and class AUCs | BA and accuracy from counts; mean AUC and combined score from stored class AUCs; seed-paired contrasts (Table 1c, Appendix B.2) |
| Derived scores for 24 public entries across three views | Primary and 19-entry sensitivity cohort changes (Table 1d) |
| 47,040 local prediction rows with ten-class probabilities | Absolute endpoints, difference-in-differences, paired/seed/prefix intervals (Tables 1e, 2, 27–28) |

The documentation in [`data/`](data/) specifies each file's schema, ordering and scope. In this
quick replay, primary, CNN and public-entry results are re-aggregated from saved
metrics, local results are recomputed from per-image probabilities, and
image-audit results are recounted from stored measurements. Replay does not rerun
models or image measurements and does not render the paper's tables or figures.

## Reproduce further

[REPRODUCE.md](REPRODUCE.md) maps the paper's released results to the code,
inputs and outputs that produced them, and lists what is not included.

- **Per-image predictions.** `recompute.py` rescores the 230 saved prediction
  arrays of the 110 primary and 60 CNN fits in [`data/predictions`](data/predictions) (127 MB)
  and checks all 450 seed/view metric records and the Table 1(b) contrasts.
- **Participant entries.** `rescore.py` rescores the public submission workbooks,
  fetched from the [challenge repository](https://github.com/palaxydf/Capsule-Vision-2024-Challenge)
  at a fixed revision, for Table 1(d) and Tables 13–15.
- **Kvasir-Capsule lists.** `kvasir_splits.py` rechecks the Appendix F.1 properties
  of the official Kvasir-Capsule lists from the upstream files at a fixed revision.
- **Training and inference.** [`training/`](training/) contains the code of the
  primary, local-removal, CNN and official-test runs, an exporter for their exact
  input lists, and the recorded GPU environment; compute is in
  [REPRODUCE.md](REPRODUCE.md#compute).

## Core code

| File | Purpose |
|---|---|
| [reuse.py](reuse.py) | Byte-confirmed identity groups, video sharing, independent hash minima and 96×96 counterpart measurements |
| [controls.py](controls.py) | Primary comparison pools and the three evaluation views |
| [analysis.py](analysis.py) | Fixed-class scoring, paired contrasts, local difference-in-differences and official-test aggregation |
| [replay.py](replay.py) | Numerical replay of the released evidence |
| [recompute.py](recompute.py) | Primary/CNN metrics from per-image predictions |
| [rescore.py](rescore.py) | Participant scores from upstream workbooks ([submissions.json](submissions.json)) |
| [training/](training/) | Model training, inference and input-list export |
| [kvasir_splits.py](kvasir_splits.py) | Appendix F.1 checks of the official Kvasir-Capsule lists |
| [example.py](example.py) | Small synthetic examples of the core operations |

<details>
<summary><strong>Input formats and API</strong></summary>

Image tables are pandas DataFrames with `image_path`, `Dataset`, and the ten
binary one-hot class columns listed in `controls.CLASSES`. Preserve workbook
order and repeated rows. Source labels come from the organizer's `Dataset` field.
Paths are relative to the supplied image root or absolute on your machine.

The code implements the CV2024 case. `reuse.py` and `controls.py` accept only
the four CV2024 sources (KVASIR, SEE-AI, KID, AIIMS) and its ten classes;
`make_pools` also requires 66 KVASIR Ulcer rows in Comp and 66 AIIMS Ulcer rows
in Full, and for Removal the supplied reference screen must flag every KVASIR
training row. Another release needs these constants replaced; arbitrary
schema-valid subsets need not satisfy these case-specific conditions.

```python
from reuse import measure_images, reference_minima
from controls import make_pools

train_hashes = measure_images(train, image_root)
val_hashes = measure_images(validation, image_root)
reference_hashes = measure_images(reference_paths, kvasir_root, reference=True)
screen = reference_minima(train_hashes, reference_hashes)
pools = make_pools(train, validation, screen, train_hashes, val_hashes)
```

Here `train` and `validation` are the organizer tables; `reference_paths` contains
an `image_path` column for Kvasir-Capsule's labeled reference images. Independent
pHash and dHash minima can have different witnesses. Comp and Class-match-R
retain their distinct seed-0 sampling rules; `ulcer_swap` implements the paper's
66-row replacement. Returned keys include `full`, `source_only`, `removal`,
`random`, `composition`, `class_match`, `ulcer_swap`, `public`, `non_kvasir`, and
`filtered`. Hash inputs are required for the corresponding filtered pools.

Prediction records are dictionaries containing unique `row_id`, integer `labels`
(0–9), and an N × 10 `scores` array in `analysis.CLASSES` order. Optional `source`
contains dataset names. View definitions map names to subsets of the same row IDs.
Assign a unique string ID to each row occurrence, including repeated image paths.
Carry these IDs into predictions and views, and align all records in the same ID
order before cohort, seed or arm comparisons. The functions validate alignment;
they do not join or reorder prediction records.
Organizer scores are evaluated directly, without adding softmax.

```python
from analysis import score_views, compare, local_contrast

scores = score_views(record, {"non_kvasir": non_kvasir_ids})
contrast = compare(removal_by_seed, full_by_seed, {"non_kvasir": non_kvasir_ids})
local = local_contrast(records_by_seed_and_arm)
```

`compare` takes `{seed: record}` mappings; each record also contains its seed.
`metrics`, `score_views`, `fixed_prediction_rescore` and `compare` default to
balanced accuracy (BA) over **all ten CV2024 classes**, for every pooled view.
Missing any required class makes pooled BA and the combined BA/AUC score `None`,
as well as their re-scoring changes; `compare` returns `None` for that view.
Pass `required_classes=(0, 1)` to fix a different BA target, or explicitly pass
`required_classes=None` to opt into represented-class BA, whose target can change
across views. Per-source BA always uses represented classes; equal-source BA
averages those source scores. AUC independently averages only defined one-vs-rest
classes (both positive and negative rows present), so it can remain defined when
fixed-target BA is undefined. A single-class view has undefined AUC; if any source
has undefined AUC, equal-source AUC and the combined score are also `None`.

`local_contrast` takes `{(seed, arm): record}` with arms `local`, `draw101`,
`draw202`, `draw303`, strict `probabilities`, scalar `seed`/`arm`, exposure
`group` (`E`/`U`) and `video_prefix`. It averages the three distant arms within
each seed before computing E−U and the paired interval. Source and exposure group
are separate fields. Its BA default remains the fixed six classes Angioectasia,
Bleeding, Erythema, Foreign Body, Lymphangiectasia and Normal; missing required
support makes its BA endpoint `None`. Function docstrings specify the details.


</details>

Code for the auxiliary CV2024 and Kvasir-Capsule training analyses, several
appendix image diagnostics, the local/distant removal sampling and the
construction of controls B–D is not included; those lists are released as exact
lists. [REPRODUCE.md](REPRODUCE.md#not-included) lists each item.

## Data and licenses

Obtain original images from the [CV2024 training/validation release](https://figshare.com/articles/dataset/Training_and_Validation_Dataset_of_Capsule_Vision_2024_Challenge/26403469/3),
[official-test release](https://figshare.com/articles/dataset/Testing_Dataset_of_Capsule_Vision_2024_Challenge/27200664/4)
and [Kvasir-Capsule](https://doi.org/10.17605/OSF.IO/DV2AG).
CV2024 combines frames from Kvasir-Capsule, the [SEE-AI project](https://doi.org/10.1002/deo2.258)
(CC BY 4.0), [KID](https://doi.org/10.1055/s-0043-105488) (academic research upon registration)
and a private AIIMS collection ([Goel et al., 2022](https://doi.org/10.1007/s00500-021-06546-y)); cite these
sources when using the data.
[`data/results`](data/results) retains their attribution for adapted public-file metadata. Follow
provider terms, including the stated research-use conditions and
[Kvasir terms](https://github.com/simula/kvasir-capsule#terms-of-use).
No medical images, patient identifiers or model weights are redistributed.
Participant workbooks are read from the upstream repository and not redistributed.

Author-written code is [MIT](LICENSE). Author-created measurements, predictions
and derived results in [`data/`](data/) are [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
([license notice](data/results/LICENSE.txt)). These licenses do not relicense
upstream images or weights. [`croissant.json`](croissant.json) describes every file in
`data/` with its SHA-256 (Croissant 1.1 with responsible-AI fields).

Training uses [DINOv2](https://github.com/facebookresearch/dinov2) code and
weights (Apache 2.0) and TorchVision ImageNet weights, obtained separately under
their terms. PyTorch, TorchVision, NumPy, SciPy, pandas, scikit-learn and
scikit-image use BSD-3-Clause;
[ImageHash](https://github.com/JohannesBuchner/imagehash) uses BSD-2-Clause;
[Pillow](https://github.com/python-pillow/Pillow/blob/main/LICENSE) uses MIT-CMU with
component notices. Third-party licenses remain separate.

## Citation

```bibtex
@inproceedings{choi2026reuscope,
  title     = {ReuScope: Auditing Data Reuse and Evaluation Sensitivity in Capsule Vision 2024},
  author    = {Choi, Eunseob and Kim, Kyeonghun and Lee, Hyuk-Jae and Kim, Nam-Joon},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  note      = {Evaluations and Datasets Track},
  url       = {https://openreview.net/forum?id=69GCzBgr1S}
}
```
