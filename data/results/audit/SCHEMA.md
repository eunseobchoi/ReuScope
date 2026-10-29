# Image-audit evidence

All tables are gzip-compressed UTF-8 CSV. `row_id`, `train_row_id`,
`val_row_id` and witness IDs join the base lists in `inputs/`; source, class,
split and Dataset-relative path are recovered there. Positions are zero-based
except the original SSIM `pair_index`, which starts at one within each group.
Flags are integer 0/1. Empty cells mean unavailable/not applicable, not zero.
All pixel/hash measurements are retained observations; counts and flags are
derived from those observations. No new image measurements are represented.

## Byte identity and recording prefixes

- `byte_crossing_members.csv.gz` — 444 members of 216 cross-split byte groups:
  210 SEE-AI groups and six AIIMS groups. `group_position,member_position`
  preserve the retained group/member ordering. `sha256` is the measured byte
  digest; `row_id` identifies the member. `label_sets_intersect` records whether
  the training and validation class sets intersect in that group. SEE-AI has
  215 validation members, 33 intersecting and 177 disjoint groups; 39 groups
  cross under different basenames. AIIMS has six validation members, all in
  intersecting groups with different basenames. A digest group is a byte
  relationship, not a patient identity.
- `kvasir_prefix41.csv.gz` — 41 rows, ordered by `video_prefix`, with
  `training_rows,validation_rows,exact_phash_validation_rows,selected_ncc99,any_ncc99`.
  Every prefix occurs in both splits. Totals are 27,011 training and 11,581
  validation frames; the last three fields sum to 1,381, 540 and 799.

## Reference screen and internal filter

- `hash_screen.csv.gz` — 53,739 rows, training base order then validation base
  order. Fields: `row_id,phash,dhash,min_phash,min_dhash,phash_ties,screen_positive`.
  Hashes are lowercase 16-character hexadecimal 64-bit hashes. Distances are
  integer Hamming distances in [0,64]. The minima independently range over the
  Kvasir reference collection; their witnesses need not coincide.
  `phash_ties` counts references attaining the minimum pHash distance.
  `screen_positive` means both independent minima are ≤6. This removes 27,016
  training rows, including all 27,011 KVASIR rows and five other rows, leaving
  the 10,591-row Removal pool.
- `filter_assignments.csv.gz` — all 4,551 non-KVASIR validation rows in base
  order. Fields: `row_id,basename_match_count,phash_match_count,basename_witness_row_id,
  phash_witness_row_id,removed,kept_position`. Match counts are against training
  within the same organizer source; a witness is the first matching path in
  lexicographic order. The filter first collects basenames with a same-source
  basename or exact-pHash match, then drops every non-KVASIR validation row
  carrying a collected basename. Thus removal can propagate to another row
  with that basename. There are 221 flagged basenames, 225 removed rows and
  4,326 retained rows. `kept_position` is the retained view order, blank when
  removed. This is a filename/hash filter, not byte-identity filtering.

## Exact-pHash counterparts and NCC

- `internal_phash_pairs.csv.gz` — 6,285 KVASIR train/validation candidate edges
  in retained `(phash,train_filename,val_filename)` order. Fields:
  `position,train_row_id,val_row_id,phash,ncc96,selected`. Candidates have equal
  pHash. `selected=1` identifies the lexicographically first training path for
  each validation query, giving 1,381 selected pairs. `ncc96` is the retained
  all-pair NCC measurement after grayscale resizing to 96×96 using the saved
  PIL preprocessing. There are 1,715 edges with NCC ≥0.99.
- `internal_validation_counters.csv.gz` — the 1,381 queried validation rows
  in base order. Fields: `row_id,representative_position,selected_train_row_id,
  phash_candidate_count,ncc99_candidate_count,selected_ncc96,selected_ssim96,
  selected_ncc99,any_ncc99`. `representative_position` recovers the original
  selected-pair inventory order. Candidate counts use all edges; `any_ncc99`
  means at least one candidate reaches 0.99. Selected NCC/SSIM values are the
  separately retained selected-pair measurements used by SSIM control group A.
  They must not replace the all-pair measurements: the maximum observed NCC
  difference for the same selected edge is 9.871845695341719e-7, with **zero**
  0.99-threshold disagreements. Floating-point implementation differences are
  a possible explanation, not an established cause. Selected confirms 540
  queries; any confirms 799, including 259 additional queries. The other 841
  selected queries and 582 any-candidate queries fail the threshold. There are
  747 queries with multiple exact-pHash training candidates.
- `alternate256_prefix_counts.csv.gz` — 41 retained prefix aggregates with
  `video_prefix,validation_rows,selected_ncc99_256`. The 256×256 sensitivity
  check recorded 387 confirmations over the same 1,381 selected candidate
  pairs and no skipped pairs. `validation_rows` counts **all** 11,581 KVASIR
  validation frames, not just those candidate pairs. Only these retained
  aggregates are available here: **256-pixel per-pair scores are unavailable**.
  The value 387 can be reaggregated, but cannot be independently thresholded
  from per-pair measurements in this folder.

NCC/SSIM values are unitless. Exact pHash alone does not establish pixel
identity, and selected-versus-any changes the counterpart rule, not the
validation population. No independence test or confidence interval follows
from these pair/frame counts. Frames share prefixes and counterpart edges.

## SSIM controls and source counterparts

- `ssim_control_pairs.csv.gz` — 8,286 retained pair measurements, 1,381 per
  group, in original order. Fields: `position,group,pair_index,train_row_id,
  val_row_id,train_video,val_video,frame_gap,ncc96,ssim96`. Prefixes are retained
  filename-derived video strings; `frame_gap` is the frame-suffix gap when
  applicable. The original group labels are `A_flagged_internal`,
  `B_adjacent_same_video`, `C_distant_same_video`,
  `D_different_video_same_class`, `E_different_video_any_class`, and
  `F_non_kvasir_same_source_class`. NCC ≥0.99 counts are 540, 68, 0, 0, 0, 0,
  respectively. Group A's separately measured selected NCC/SSIM values also
  appear in the query counters; all control pairs, including failures, remain.
- `source_selected_pairs.csv.gz` — 38,592 retained KVASIR-to-source selected
  pairs in original order. Fields: `position,row_id,reference_path,ncc96,
  independent_min_phash,independent_min_dhash,selected_pair_phash,selected_pair_dhash,
  phash_ties,byte_equal,pixel_equal_96`. `reference_path` is class/filename
  relative to the Kvasir-Capsule reference image root. The independent minima
  describe the screen; `selected_pair_*` describes the actual chosen pair.
  There are 35,625 NCC ≥0.99 pairs; none are byte-equal or 96-pixel-equal in
  these retained checks. Among 2,372 pairs with maximum selected-pair hash
  distance 3–6, 858 reach NCC ≥0.99. The data retain 6,275 pHash-tied queries
  and 4,665 selected dHash distances that differ from the independent minimum.
- `nonkvasir_internal_pairs.csv.gz` — 301 retained internal candidate pairs
  from the non-KVASIR source checks. Fields: `Dataset,source_position,
  train_row_id,val_row_id,phash_distance,dhash_distance,selected,byte_equal,ncc96,ncc256`.
  Distances are same-pair Hamming distances, both ≤6; `selected` chooses the
  first training path lexicographically per query. There are SEE-AI/KID/AIIMS
  candidate counts 288/7/6, selected-query counts 264/4/6, and selected NCC96
  ≥0.99 counts 220/2/6. These 256-pixel measurements are available; they do not
  fill the missing KVASIR exact-pHash 256-pixel per-pair scores above.

## Attribution and terms

Derived metadata and results are **CC BY 4.0**, attributed to ReuScope authors
Eunseob Choi, Kyeonghun Kim, Hyuk-Jae Lee and Nam-Joon Kim. Path, class and
organizer-source attribution is retained through the input base tables from
[Capsule Vision 2024](https://figshare.com/articles/dataset/Training_and_Validation_Dataset_of_Capsule_Vision_2024_Challenge/26403469/3).
Reference paths retain [Kvasir-Capsule](https://doi.org/10.17605/OSF.IO/DV2AG)
attribution. Upstream dataset terms remain separate from the derived-metadata
license. No images or model weights are included.
