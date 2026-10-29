# Reproducing the paper's results

The repository supports four levels, from a seconds-long CPU check to model
training. Each later level adds inputs; none needs the earlier outputs.

| Level | Command | Inputs | Hardware | Time |
|---|---|---|---|---|
| 1. Replay | `python replay.py --output replay.json` | [`data/results`](data/results) (5 MB) | CPU | seconds |
| 2. Predictions | `python recompute.py --output recomputed.json` | + [`data/predictions`](data/predictions) (127 MB) | CPU | under a minute |
| 3. Participant entries | `python rescore.py --results-dir external/Results --output rescored.json` | + upstream challenge workbooks (58 MB) | CPU | under a minute |
| 4. Training and inference | see [`training/README.md`](training/README.md) | + images, pretrained weights | CUDA GPU | see [Compute](#compute) |

Levels 1–3 use `requirements.txt` (Python 3.12) and the repository's `data/`
folder. They stop with an error if an input or a checked value differs;
outputs must be new files.

For level 3, fetch only the needed upstream files at the fixed revision; no
organizer code is imported or redistributed:

```bash
git clone --filter=blob:none --no-checkout https://github.com/palaxydf/Capsule-Vision-2024-Challenge external
git -C external sparse-checkout init --no-cone
git -C external sparse-checkout set /Results/validation_data.xlsx /Results/submitted_excel_files/validation/ \
  /Results/metrics_reports/metrics_reports_val/ /Results/metrics_reports/metrics_reports_test/
git -C external checkout --detach de38787e8267f763a5f59695c67bb6d28ece70bf
```

`rescore.py` identifies the 27 workbooks and organizer reports by the SHA-256
values in `submissions.json`, scores the 25 readable entries without softmax
or other conversion, and forms the 25-, 24-, 19- and 18-entry cohorts.

For Appendix F.1, fetch the two official Kvasir-Capsule lists the same way and
recheck their row counts, video prefixes, intersections and frame gaps:

```bash
git clone --filter=blob:none --no-checkout https://github.com/simula/kvasir-capsule kvasir
git -C kvasir sparse-checkout init --no-cone
git -C kvasir sparse-checkout set /official_splits/
git -C kvasir checkout --detach 553601dc316fef82f39c952e28f60aaad66039e2
python kvasir_splits.py --splits kvasir/official_splits --output kvasir_splits.json
```

## Results map

"Check" names the command that recomputes the stated values from released
files and compares them with the recorded values in `data/results`. Values marked "computed"
are produced by the command without a stored reference; "supplied" means the
released files contain the values but no command recomputes them. Paths in
the third column are relative to `data/results` unless they start with `data/`.

| Paper result | Produced by | Released inputs and outputs | Check |
|---|---|---|---|
| Figure 1, Table 3 | Summaries of the results below | — | — |
| Figure 2, Table 4: byte identity, shared video prefixes | `reuse.py` (`byte_identity_summary`, `prefix_overlap`) on CV2024 images | `audit/byte_crossing_members`, `audit/kvasir_prefix41` | `replay.py` (group, row and prefix counts); renamed groups supplied |
| Table 5: independent hash minima to Kvasir-Capsule | `reuse.py` (`reference_minima`); PDQ columns not included | `audit/hash_screen` (per-row minima and screen flag) | supplied |
| Table 6, Section 3.2: NCC of selected source pairs | Counterpart selection from the original pipeline (ties in reference-cache order); NCC as in `reuse.py` | `audit/source_selected_pairs` | `replay.py` (pair, NCC ≥ 0.99 and distance 3–6 counts) |
| Section 3.2, Tables 9 and 11: KVASIR exact-pHash counterparts, NCC, SSIM | `reuse.py` (`internal_rows`) | `audit/internal_phash_pairs`, `internal_validation_counters`, `alternate256_prefix_counts`, `ssim_control_pairs` | `replay.py` (pair counts; 540, 799 and 387 NCC-positive rows; SSIM medians and prefix intervals) |
| Tables 9–10: non-KVASIR candidates, basename and joint-rule columns | Basename and joint-rule searches from the original pipeline | `audit/filter_assignments`, `audit/nonkvasir_internal_pairs` | supplied; the KVASIR joint-rule count (4,896) is not included |
| Pools and views; Table 17 | `controls.py` | `inputs/` memberships; `audit/filter_assignments`; `training/prepare.py` → 19 lists | `replay.py` (row counts, 225 filtered rows); `lists.sha256` |
| Table 1(a,b) DINOv2, Figure 3, Tables 18–20, 22 | `training/dinov2/primary.py`, 110 fits | `data/predictions` (110 × 16,132 rows); `numerical/seed_metrics` | `recompute.py` (450 per-seed records from per-image probabilities, Table 1(b) contrasts; table means computed); `replay.py` (from per-seed records) |
| Table 21: per-source results | same runs | `data/predictions` (`source` array) | supplied |
| Table 1(b) CNN rows, Table 23 | `training/cnn/`, 60 fits | `data/predictions` (60 × Public, Non-KV); `numerical/seed_metrics` | `recompute.py`; `replay.py` |
| Table 1(c), Table 16: official test | `training/official_test/producer.py`, 30 fits | `numerical/official_test` (confusion matrices, class AUCs) | computed by `replay.py` (BA and accuracy from counts, AUC from stored class AUCs, seed-paired contrasts) |
| Table 1(d), Tables 13–15: participant entries | upstream workbooks and organizer reports | `submissions.json`; `numerical/entry_scores` | `rescore.py` (72 entry/view scores checked against `data/results`; ranges, rank changes and correlations computed); Table 1(d) computed by `replay.py` from entry scores |
| Appendix F.1: official Kvasir-Capsule lists | upstream `official_splits` at revision 553601dc | — | `kvasir_splits.py` (rows, prefixes, filename and video/class intersections, minimum frame gap) |
| Table 1(e), Table 2, Tables 27–28: local removal | `training/dinov2/local.py`, 40 fits | `numerical/local_probabilities` (47,040 rows); `inputs/local_*` | `replay.py` (endpoints, seed and prefix intervals) |

Levels 1–3 verify the saved outputs of the reported runs. Training the models
again (level 4) follows the recorded recipes but is not expected to repeat
them bitwise; see `training/README.md`.

### Not included

- **Auxiliary CV2024 training analyses** (Tables 24–26; related-source
  add-back): their scripts were not verified against each run record, so none
  is published as their producer; the paper reports their settings and results.
- **Kvasir-Capsule experiments** (Tables 29–31): run records keep arguments
  and per-epoch metrics but not per-run hardware or a binding to the exact
  source revision, so no script is published as their producer.
- **Other measurements and analyses**: feature and transform audits
  (Tables 7–8), PDQ (Table 5), reference-counterpart selection (Table 6),
  the basename and joint-rule candidate searches (Tables 9–10), the 256×256
  NCC measurement (supplied as per-prefix counts), video-prefix resampling
  (Table 12; its ranges follow from `audit/kvasir_prefix41` under the rule
  stated in Appendix A.4), matched-SSIM control construction (the control pairs are
  supplied), the official-test byte scan (Appendix B.2) and the
  external-release screens of Appendix G
  (Table 32; ChestX-ray14 lists).
- **List construction**: local/distant sampling and controls B, C and D are
  released as exact lists, not as sampling code; the Appendix C.2 recipe
  reproduces B, C and D row for row from the released tables.
- **Run artifacts**: per-epoch histories (the source of the best-epoch values
  in Section 3.3 and Appendix C), model checkpoints, execution logs and the run
  records behind the compute accounting (Table 33). Images, pretrained weights
  and participant workbooks are obtained from their providers.

## Compute

Elapsed execution time of the reported runs from the run records, including
data loading and evaluation inside each run. Runs used a shared university
GPU cluster; capacities are 80 GB (A100-SXM4) and 48 GB (RTX A6000, L40S).

| Runs | Fits | GPU (fits) | Minutes per fit | Hours |
|---|---:|---|---|---:|
| Primary DINOv2 | 110 | A100 (110) | 10.7–32.2 | 22.23 |
| Local removal | 40 | RTX A6000 (24), A100 (8), L40S (8) | 28.8–64.7 | 24.34 |
| CNN | 60 | A100 (60) | 3.0–8.6 | 5.32 |
| Official test | 30 | L40S (28), A100 (1), RTX A6000 (1) | 7.8–20.9 | 6.06 |
| Auxiliary CV2024 training | 154 | A100, RTX A6000, L40S | mean 21.8 | 55.96 |
| DINOv2 feature extraction | 1 | RTX 4070 Ti SUPER 16 GB | 9.5 | 0.16 |
| Kvasir-Capsule training | 28 | not recorded | — | 2.92 training loop; **≈4.8 estimated** (upper ≈5.6) |
| **Reported total** | 422 + 1 | | | **≈118.8 GPU run-hours** (upper ≈119.6) |

- **GPU count.** Primary and CNN runs recorded one visible GPU; local runs were
  configured with one. The official-test launcher requested one GPU per run and
  two A100s for Full seed 0 (≈6.23 device-hours for the family if both were
  used); visible counts were not recorded. Auxiliary and Kvasir-Capsule runs
  did not record device counts.
- **Memory.** CNN runs allocated at most 3.3 GiB of CUDA memory; the DINOv2
  runs ran on 48 GB and 80 GB GPUs. Official-test launchers requested 6–16 CPU
  cores and 80–256 GB of host memory. The CV2024 training and validation
  images occupy about 0.4 GB.
- **Estimates.** Kvasir-Capsule runs timed only their training loops. Their
  evaluation time is estimated from each run's own training throughput times
  the evaluation-to-training cost ratio measured in the 30 official-test runs
  (median 0.89, max 1.21), plus per-run setup measured in the primary runs
  (median 9 s, max 30 s); the upper value uses the maxima. They train the same
  DINOv2-L/14 model with the batch sizes that ran on single 48 GB GPUs in the
  local runs.
- **CPU.** The image measurements in `reuse.py` take ≈1.2 core-hours for all
  105,362 images and 53,464 pairs (estimate: 800 real images and 200 pairs
  timed on one core of an AMD Ryzen 7 7800X3D, extrapolated). On that core,
  levels 1–3 took 5, 30 and 29 seconds with at most 210 MB of memory.
- **Additional work.** The project also ran 60 additional CNN diagnostic fits
  (5.26 h), preliminary experiments and failed runs; these are not part of the
  totals and were not fully recorded.
