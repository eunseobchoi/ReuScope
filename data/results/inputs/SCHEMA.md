# Ordered inputs

These gzip-compressed UTF-8 CSVs preserve retained row order, labels, source
assignment and multiplicities. `image_path` is relative to the organizer's
`Dataset/` directory, with `/` separators. No images or weights are included.
All `position` values are zero-based. Empty CSV cells mean unavailable/not
applicable; they are not zeros. Counts and schemas below describe every payload.

## Base lists and primary pools

| Payload | Rows | Fields and meaning |
| --- | ---: | --- |
| `training.csv.gz` | 37,607 | `row_id,image_path,Dataset`, then ten one-hot class columns; original training workbook order. IDs `t000000` through `t037606`. Also the Full pool. |
| `validation.csv.gz` | 16,132 | Same fields; original validation workbook order. IDs `v000000` through `v016131`. Also the Public evaluation view. |
| `primary_memberships.csv.gz` | 105,950 | `pool,position,row_id`; ten reduced/control pools join to training. Sort each pool by `position`; repeated row IDs must remain repeated. Full uses the base table directly. |
| `evaluation_memberships.csv.gz` | 8,877 | `view,position,row_id`; joins validation. `non_kvasir` has 4,551 rows; `filtered` has 4,326. Public uses the base table directly. |

The source field `Dataset` is the organizer attribution: `KVASIR`, `SEE-AI`,
`KID`, or `AIIMS`. Class order, also the zero-based numeric label order, is:
`Angioectasia`, `Bleeding`, `Erosion`, `Erythema`, `Foreign Body`,
`Lymphangiectasia`, `Normal`, `Polyp`, `Ulcer`, `Worms`.

Pool names are `Random`, `Source-only`, `Removal`, `Class-match-R`, `Comp`,
`A`, `B`, `C`, `D-dup`, `D-uniq`. Removal and Class-match-R have 10,591 rows;
the other eight have 10,596. C and D-dup each have 10,530 distinct row IDs,
including 66 additional repeated draws. Do not deduplicate or sort by row ID.
Source-only retains the non-KVASIR sources; Removal uses the
independent pHash/dHash reference-minimum screen. These are different pools.
The filtered validation view removes 225 rows from non-KVASIR; its exact
assignments are supplied in the audit payload.

## Local/distant experiment

| Payload | Rows | Fields and meaning |
| --- | ---: | --- |
| `local_evaluation.csv.gz` | 1,176 | `position,row_id,base_row_id,image_path,Dataset,label,group,video_prefix`. `row_id` is the **image filename**, unique in this cohort; `base_row_id` joins validation. `label` is the zero-based integer above. `group` is `E` (exposed) or `U` (matched-unflagged), 588 each. `Dataset` stays `KVASIR`; exposure does not replace source attribution. |
| `local_parent_memberships.csv.gz` | 1,598 | `group,position,row_id,retained_local`. Original 799-row E and 799-row U lists; `row_id` joins validation; `retained_local` is 1 for the retained 588 per group and 0 for the excluded 211 per group. |
| `local_removals.csv.gz` | 4,604 | `role,position,row_id,video,label,filename,frame,matched_run_id,requested_run_length,min_exposed_frame_gap,min_phash_distance,min_dhash_distance,min_joint_hash_distance,eligible_temporal_group`. `row_id` joins training; `label` here is the original class **name**. Remaining columns retain the original temporal selection CSV values. |

The local predictions (`numerical/local_probabilities.npz`) join on `(seed,arm,row_id)` where `row_id`
is the filename in `local_evaluation.csv.gz`. All 40 retained prediction files
(47,040 rows) were checked against its ordered paths, labels, groups and
zero-based `dataset_index` = `position`. The ten retained training seeds are
4–9 and 14–17; local and draw101/draw202/draw303 use the same cohort order.
The original `NCC99_EXPOSED`/`MATCHED_UNFLAGGED` pseudo-source values are mapped
to `group=E/U`; they are not new dataset sources.

Removal roles are `common1100` (1,100), `local876` (876), and `distant101`,
`distant202`, `distant303` (876 each). To reconstruct an ordered training arm,
start with `training.csv.gz` and retain, in base order, rows whose IDs are absent
from `common1100` plus the chosen additional removal role. Each resulting
local/distant arm has 35,631 rows; common-only has 36,507. All nine retained
training CSVs across the three draws match this reconstruction row for row.

`position` in removal records is the within-role order from the original
selection file. To reconstruct a draw's complete 2,852-row selection file,
map roles to original names and concatenate in original order:
`distant{draw}` → `distant_extra`, `common1100` → `exact_shared`,
`local876` → `local_extra`. Each role is ordered by `position`.
`video`/`video_prefix` is a released filename prefix, not a patient identifier.
`frame` is its numeric suffix; `matched_run_id` is local to each video/class.
`requested_run_length` counts contiguous positions in the source training
sequence. Blank run/gap/dHash cells on common rows are retained as blank.

For original local/distant rows, `min_exposed_frame_gap` is the minimum frame
suffix gap to exposed validation rows in that video/class;
`min_phash_distance` and `min_dhash_distance` are independent Hamming minima.
`min_joint_hash_distance` is the minimum over counterparts of the larger of
the two distances. Distant selections require gap ≥50 and joint distance >6.
`eligible_temporal_group` retains `True`/`False`. All 1,100 common rows are
removed, including 298 belonging to the excluded group. The excluded Normal
group needed 256 additional rows but had only 141 admissible distant rows;
its 211 E and 211 U rows remain visible in the parent memberships.

The local NPZ's ordered metadata matches this cohort, and all 330 primary
metric records' total and per-class evaluation counts match the three views.

These are ordered frame observations, not independent patients. Local E/U are
matched in 36 video/class strata across 32 prefixes. The model-comparison unit
is the paired training seed after averaging the three fixed distant arms;
frames/arms within a seed are not independent replicates. The cohort, controls,
seed set and endpoint were selected with access to results. This input package
does not calculate effects or uncertainty; the numerical evidence supplies those
conditional exploratory contrasts. Seed spread alone is not a significance test.

## Attribution and terms

Derived metadata and results are **CC BY 4.0**, attributed to the ReuScope
authors: Eunseob Choi, Kyeonghun Kim, Hyuk-Jae Lee and Nam-Joon Kim.
The base path/label/source fields retain the organizer's
[Capsule Vision 2024 training/validation dataset](https://figshare.com/articles/dataset/Training_and_Validation_Dataset_of_Capsule_Vision_2024_Challenge/26403469/3)
attribution. `KVASIR` paths also retain the
[Kvasir-Capsule](https://doi.org/10.17605/OSF.IO/DV2AG) source attribution.
Upstream dataset terms remain separate from the derived-metadata license;
no upstream images or weights are relicensed or redistributed here.
