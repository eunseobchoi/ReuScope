# Numerical evidence

Run `python replay.py --output replay.json` from the repository root; it reads
`data/results` with the core `analysis.py`.
Use the core requirements. CPU only; no images, weights or training are required.

- `seed_metrics.json.gz`: 330 primary records (11 pools × seeds 0–9 × Public,
  Non-KV, Filtered) and 120 CNN records (2 backbones × 3 pools × seeds 0–9 ×
  Public, Non-KV). Each contains family/backbone/arm/seed/view and metrics with
  per-class counts and recalls. Primary and CNN replay **reaggregates saved
  per-seed metrics**, not individual predictions. Public/Non-KV/Filtered contain
  16,132/4,551/4,326 rows. These are final-epoch results, not epoch selection.
- `official_test.json.gz`: 30 final-epoch records (3 pools × seeds 0–9), each
  with a true-by-predicted 10×10 confusion matrix and ten retained class AUCs.
  BA and accuracy are recomputed from counts; AUCs are averaged, not recomputed
  from per-image scores. Each evaluation has 4,385 rows. Reported spread is SE.
- `entry_scores.json.gz`: 24 entries × 3 views, containing pooled combined,
  equal-source BA and equal-source combined scores. Replay reaggregates these
  **derived summaries**, without third-party submission workbooks/predictions.
  Primary excludes Team G; the 19-entry sensitivity also excludes Teams U–Y.
- `local_probabilities.npz`: original float64 probabilities, shape
  `(10,4,1176,10)`, in `seeds`, `arms`, `row_id`, `classes` order. Seeds are
  `[14,15,16,17,4,5,6,7,8,9]`; arms are local/draw101/draw202/draw303.
  There are exactly 47,040 per-image prediction records, ten probabilities each.
  Shared row metadata: public filename `row_id`, relative public `image_path`, integer `labels`, `group` E/U
  (588 each), and `video_prefix` (32 unique). E/U counts match within video/class.
  These endpoints **are recomputed from the retained probabilities**.
- `reference_checks.json.gz`: retained interval targets for replay checks.

Class order is Angioectasia, Bleeding, Erosion, Erythema, Foreign Body,
Lymphangiectasia, Normal, Polyp, Ulcer, Worms. All scores use fractions; NLL
uses natural logarithms. Local BA uses classes 0,1,3,4,5,6. Distant arms are
averaged within seed before paired inference. Intervals condition on the fixed,
exploratory design. Prefix bootstrap uses 10,000 shared multinomial draws,
RNG seed 42, seeds averaged within draw, and present-class BA.
Only 914/10,000 draws retain all six classes in both groups (recomputed using
the same weights); it is not a fixed-six-class or patient-level confidence interval.
Local row order and metadata are checked against `inputs/local_evaluation.csv.gz`.

The script uses the public ReuScope `analysis.py` functions and the prefix
resampling rule in Appendix E. Values are retained
at full precision; paper rounding is applied only when comparing displayed cells.
Primary pools are Full, Random, Comp, Class-match-R, Removal, Source-only,
A, B, C, D-uniq and D-dup. CNN and official-test pools are Full, Random and
Source-only.

Replay does not rerun training or inference. Primary/CNN per-image
predictions are in `data/predictions` (checked by
`recompute.py`); participant workbooks stay upstream (`rescore.py`). Of the
per-image outputs, only the local predictions are in this folder.
