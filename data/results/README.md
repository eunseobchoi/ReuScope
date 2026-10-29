# ReuScope numerical evidence

Retained inputs and outputs for *ReuScope: Auditing Data Reuse and Evaluation
Sensitivity in Capsule Vision 2024* (NeurIPS 2026, Evaluations and Datasets).

From the repository root, with Python 3.12:

```bash
pip install -r requirements.txt
python replay.py --output replay.json
```

No images, weights or GPU are required. `replay.json` must be a new
file. Replay computes numerical quantities from these records; it does not
render the paper's figures or tables, and it does not rerun models or image
measurements. `python training/prepare.py lists`
exports the ordered training and evaluation lists used by the training code.

| Directory | Contents | Documentation |
|---|---|---|
| `inputs/` | Base public row lists; ordered primary pool/view memberships; local evaluation, parent cohorts and removal sets | `inputs/SCHEMA.md` |
| `audit/` | Retained byte-crossing members, source/prefix/hash measurements, counterpart pairs and SSIM controls | `audit/SCHEMA.md` |
| `numerical/` | Primary/CNN metric records, derived public-entry scores, official-test counts/class AUCs and local probabilities | `numerical/README.md` |

All 11 primary pools are specified, with Full using the base training table.
Public validation uses the base validation table. Membership positions preserve
order and repetition. Filenames and source/video identifiers come from public
benchmark artifacts; video prefixes do not identify unique patients.

The local array contains 10 seeds × 4 arms × 1,176 rows = 47,040 predictions,
with ten probabilities per row. These allow recomputing the local endpoints and
intervals from saved model outputs. In this folder, primary/CNN and participant
comparisons use per-seed or per-entry metrics; `../predictions` with
`recompute.py` recomputes the primary/CNN
metrics from per-image probabilities, and `rescore.py` recomputes the entry
scores from the upstream submission workbooks.
Official-test BA and accuracy come from confusion matrices; AUC aggregation uses
stored class AUCs. These evidence types support different levels of numerical
verification. Outcome-informed selection and the paper's conditional inference
limits remain unchanged.

Main replay coverage is Table 1's panels, Table 2, local Tables 27–28 and the
included relationship summaries, including SSIM joint-prefix intervals. The
stored records additionally permit inspection of class counts and recalls.
The alternate 256×256 counterpart result is an aggregate of retained prefix
counts, without per-pair alternate-resolution scores.

This folder does not include participant prediction workbooks (obtained
from the upstream challenge repository), auxiliary training results,
Kvasir-Capsule run histories, execution logs, medical images or model weights.
See `REPRODUCE.md` in the repository root and Appendix H of the paper for the
mapping from results to code and the remaining limits.

## Attribution

Eunseob Choi, Kyeonghun Kim, Hyuk-Jae Lee and Nam-Joon Kim. ReuScope (2026).
[Paper](https://openreview.net/forum?id=69GCzBgr1S) ·
[Code and data](https://github.com/eunseobchoi/ReuScope).

Public row paths, labels and source fields are adapted from Handa et al.,
[Training and Validation Dataset of Capsule Vision 2024 Challenge, v3](https://doi.org/10.6084/m9.figshare.26403469.v3).
The adaptation adds stable row IDs, ordered membership tables and audit fields;
released labels are retained. CV2024 combines frames from Kvasir-Capsule, the SEE-AI project
(Yokote et al., 2024, CC BY 4.0, [doi:10.1002/deo2.258](https://doi.org/10.1002/deo2.258)), KID
(Koulaouzidis et al., 2017, [doi:10.1055/s-0043-105488](https://doi.org/10.1055/s-0043-105488)) and an
AIIMS collection (Goel et al., 2022, [doi:10.1007/s00500-021-06546-y](https://doi.org/10.1007/s00500-021-06546-y));
the source labels in these tables retain that attribution. Official-test summaries concern
[Testing Dataset of Capsule Vision 2024 Challenge, v4](https://doi.org/10.6084/m9.figshare.27200664.v4).
Both release records state CC BY 4.0; the test record and challenge description
also specify research-only use. Source correspondence uses Smedsrud et al.,
[Kvasir-Capsule](https://doi.org/10.1038/s41597-021-00920-z), with
[data access](https://doi.org/10.17605/OSF.IO/DV2AG) and
[provider terms](https://github.com/simula/kvasir-capsule#terms-of-use).
The source reference paths and published video prefixes retain this attribution.

Author-created measurements, predictions and derived results use CC BY 4.0;
see `LICENSE.txt`. Provider terms remain applicable to their assets. No new
permission to images, weights or patient identities is granted by this repository.
