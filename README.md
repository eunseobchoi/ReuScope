<div align="center">

# ReuScope

### Auditing Data Reuse and Evaluation Sensitivity<br>in Capsule Vision 2024

Eunseob Choi<sup>1</sup> · Kyeonghun Kim<sup>2</sup> · Hyuk-Jae Lee<sup>3</sup> · Nam-Joon Kim<sup>3,✉</sup>

<p><sup>1</sup> Gwangju Institute of Science and Technology<br><sup>2</sup> OUTTA · <sup>3</sup> Seoul National University<br><sup>✉</sup> Corresponding author</p>

[`Python 3.12`](#cpu-quickstart) · [MIT code](LICENSE) · [Paper](https://openreview.net/forum?id=69GCzBgr1S) · [Resource terms](THIRD_PARTY_NOTICES.md)

[Overview](#overview) · [Abstract](#abstract) · [Findings](#findings) · [Quickstart](#cpu-quickstart) · [Code & data](#code-and-data) · [Citation](#citation) · [Acknowledgements](#acknowledgements)

</div>

## Overview

**Can a model comparison reverse when the evaluation population changes?** ReuScope connects measurements of data reuse to controlled training and evaluation comparisons. It changes the evaluation population while holding predictions fixed, and changes training data while holding evaluation fixed, using Capsule Vision 2024 (CV2024) as a worked case.

The study examines **37,607 public training rows and 16,132 validation rows from four acquisition sources**. Ten-seed DINOv2 comparisons, frozen-backbone CNN comparisons, matched training controls and a local-versus-distant removal experiment show why file identity, shared videos, source composition and performance sensitivity need to be measured separately.

## Abstract

We develop ReuScope, a protocol for auditing data reuse and evaluation sensitivity, using Capsule Vision 2024 as a case study. It combines measurements of data relationships with two comparisons: changing the evaluation population while holding predictions fixed, and changing training data while holding evaluation fixed. Byte-identical files cross the public training–validation boundary, and all 41 contributing KVASIR video prefixes occur in both partitions. Across ten paired seeds, removing KVASIR and five other-source training rows flagged by the Kvasir-Capsule hash screen lowers public-validation balanced accuracy from 0.825 to 0.612. Evaluating the same fitted models on non-KVASIR validation reverses the comparison: removal improves balanced accuracy by 1.38 percentage points. Matched controls show that sample size and class counts alone do not explain the observed score differences. A local-versus-distant removal experiment yields an exploratory difference-in-differences estimate of −0.355 percentage points in true-class probability, while the accuracy effect remains unresolved. Missing patient and examination identifiers prevent assessing independence at those levels. The resulting procedure distinguishes observed reuse from conditional performance sensitivity and specifies the population, class support and uncertainty unit for each conclusion.

## Findings

- **Different checks reveal different relationships.** Byte-identical files cross the public training–validation boundary in SEE-AI and AIIMS. KVASIR has no byte-identical crossings, yet all 41 contributing video prefixes appear in both partitions.
- **A training comparison reverses across evaluation populations.** Removing KVASIR and five additional screen-positive training rows lowers public-validation balanced accuracy from **0.825 to 0.612**. The same fitted model pairs show a **+1.38 percentage-point** removal benefit on non-KVASIR validation.
- **Matching sample count is insufficient.** Source-only and Random each use 10,596 training rows, but reach public-validation balanced accuracy of **0.611 and 0.779**, respectively. Class-count controls and within-class source replacement further show the importance of training composition.
- **Confidence and accuracy answer different questions.** Local-versus-distant removal gives an exploratory **−0.355 percentage-point** difference-in-differences in true-class probability; the accuracy interval spans zero.

ReuScope turns these observations into a four-step protocol: define the boundary and evidence unit, measure each data relationship, separate training from evaluation changes, and report conclusions with their scope and uncertainty. The diagnostic views supplement the official evaluation; each result is tied to a specified population and comparison.

## CPU quickstart

Use **Python 3.12** and run from the repository root. These commands run synthetic metric and interface checks. No GPU, raw images or pretrained weights are needed.

```bash
git clone https://github.com/eunseobchoi/ReuScope.git
cd ReuScope
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -B evaluate.py --self-check
python -B -S training/check_cpu.py
```

To train models, follow the [training guide](training/README.md). Training requires PyTorch, TorchVision, a suitable GPU, separately obtained images and weights, and explicit ordered training/evaluation lists.

## Evaluate predictions

Evaluate your own predictions from the repository root:

```bash
python evaluate.py /path/to/predictions.csv
python evaluate.py /path/to/predictions.npz --exclude-source KVASIR
python evaluate.py /path/to/local_predictions.csv --f1-class-set union
python evaluate.py /path/to/cnn_run/orig_val.pt
```

| Format | Required fields | Optional fields checked when supplied |
|---|---|---|
| CSV | `true_label_index` and ten `p_CLASS` columns | `dataset_index`, `predictions`, `stratum` |
| NPZ | `true_label_index`, `probabilities` (N × 10) | `classes`, `dataset_index`, `predictions`, `logits`, `stratum` |
| PT | Tensor export from `training/crossmodel.py` | Requires PyTorch; read with its restricted loader |

Class order: **Angioectasia, Bleeding, Erosion, Erythema, Foreign Body, Lymphangiectasia, Normal, Polyp, Ulcer, Worms**. Labels are integer indices 0–9. Probabilities must be finite, in [0, 1], and sum to one per row. NPZ arrays must not require pickle. When supplied, `dataset_index` must be 0, 1, …, N−1 in file order.

The optional `stratum` field enables source-specific scores and source exclusion. Use nonempty source names and Unicode string arrays for NPZ; quote CSV fields containing commas. CNN tensor exports support pooled metrics; use CSV or NPZ with `stratum` for source-specific analysis.

The command prints JSON metrics to standard output. Balanced accuracy averages recall over represented classes; macro AUC uses classes with positive and negative examples. Missing AUC support is reported as null. `combined_ovr` averages balanced accuracy and macro AUC; `null_acc` is the majority-class accuracy baseline. Source-balanced scores give sources equal weight. Primary/CNN macro-F1 uses represented true classes; the confidence-training recipe uses the union of true and predicted classes, selected with `--f1-class-set union`.

[`evaluation.py`](evaluation.py) also provides `paired_intervals` for matched seed differences and `cluster_bootstrap_intervals` for the local-versus-distant video-prefix bootstrap. Their docstrings specify the input arrays and contrasts. The bootstrap requires complete four-arm blocks with aligned rows and matched exposed/unflagged groups; its balanced accuracy uses present-class support within each draw.

## Code and data

| What you want to do | Start here |
|---|---|
| Compute pooled and source-specific prediction metrics | [Evaluation CLI](evaluate.py) and [input formats](#evaluate-predictions) |
| Calculate paired seed and video-prefix intervals | [Statistical functions](evaluation.py) |
| Train DINOv2 for primary or local-versus-distant comparisons | [DINOv2 training](training/README.md#dinov2) |
| Plan, run and summarize frozen-backbone CNN comparisons | [CNN comparisons](training/README.md#cnn-comparisons) |
| Check input, output and path interfaces without a GPU | [CPU checks](training/check_cpu.py) |
| Find dataset, model and software attribution | [Third-party notices](THIRD_PARTY_NOTICES.md) |

```text
evaluate.py                 Prediction evaluation CLI and metric self-check
evaluation.py               Metrics, paired intervals and video-prefix bootstrap
training/
  README.md                 Data schemas, environment and training commands
  crossmodel.py             CNN comparison planning, execution and summaries
  crossmodel_weight_hashes.json
                            Expected hashes of separately obtained CNN weights
  check_cpu.py              CPU interface and input-validation checks
  src/
    primary.py              Primary DINOv2 training
    confidence.py           DINOv2 local-versus-distant confidence comparisons
    cross_model.py          Frozen-backbone model and training implementations
    dataset_official.py     Shared Kvasir loader interface
    utils/                  Dataset loading and path resolution
requirements.txt            Numerical and image-processing dependencies
CITATION.cff                Machine-readable citation metadata
```

The repository provides **training and prediction-evaluation code**. Prepare the ordered training/evaluation lists and comparison pools before running; the scripts do not construct study splits. Obtain the original datasets and pretrained weights from their providers. Study-specific split lists and saved predictions are not bundled here.

## Citation

If you use ReuScope code or research, please cite:

```bibtex
@inproceedings{choi2026reuscope,
  title     = {{ReuScope: Auditing Data Reuse and Evaluation Sensitivity in Capsule Vision 2024}},
  author    = {Choi, Eunseob and Kim, Kyeonghun and Lee, Hyuk-Jae and Kim, Nam-Joon},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  note      = {NeurIPS 2026 Evaluations and Datasets Track},
  url       = {https://openreview.net/forum?id=69GCzBgr1S}
}
```

Citation metadata is also available in [CITATION.cff](CITATION.cff).

## Acknowledgements

This work was supported by Artificial intelligence industrial convergence cluster development project funded by the Ministry of Science and ICT(MSIT, Korea)&Gwangju Metropolitan City and the IITP AI Semiconductor Program and the ANCHOR Program, funded by the Korean government and the Seoul Metropolitan Government (IITP-2023-RS-2023-00256081; 2026-ANCHOR-01-110).

We thank the dataset, model and software authors whose resources support this work. Attribution and applicable terms are listed in [Third-party notices](THIRD_PARTY_NOTICES.md).

## Scope and terms

Manuscript sources, paper PDFs, raw images, model weights, saved prediction datasets and experiment logs are not included. The CPU quickstart checks code interfaces using synthetic inputs; reproducing study results requires the corresponding data, ordered lists, predictions or training runs.

Author-written code uses [MIT](LICENSE). External datasets, weights and dependencies retain their own terms; the code license does not grant rights to those assets. See [third-party notices](THIRD_PARTY_NOTICES.md) for sources and conditions of use.
