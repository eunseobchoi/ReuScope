#!/usr/bin/env python3
"""Recheck the official Kvasir-Capsule list properties reported in Appendix F.1.

Fetch the two lists at the fixed revision (no images are needed):
  git clone --filter=blob:none --no-checkout https://github.com/simula/kvasir-capsule kvasir
  git -C kvasir sparse-checkout init --no-cone
  git -C kvasir sparse-checkout set /official_splits/
  git -C kvasir checkout --detach 553601dc316fef82f39c952e28f60aaad66039e2
  python kvasir_splits.py --splits kvasir/official_splits --output kvasir_splits.json

This check was written for the release to recompute the reported values from the
upstream lists; it is not the code that originally produced them.
"""
import argparse
import bisect
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

PINNED = {"split_0.csv": "fc7ed45cb061a378403c1b31375345462c90408960ee555727f7c00f8c3c0b3d",
          "split_1.csv": "5757b5ac99bdba4642edc2fde7eb2e1dbb2e0392b591b6866ec837720508134a"}
REPORTED = dict(rows=[23061, 24100], video_prefixes=[25, 25], union_prefixes=43, shared_prefixes=7,
                unique_filenames=[23061, 24092], filename_intersection=0, shared_video_class_pairs=0,
                min_cross_fold_frame_gap=201)
NAME = re.compile(r"([0-9a-f]+)_(\d+)\.jpg")


def read(path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != PINNED[path.name]:
        raise ValueError(f"{path.name} differs from revision 553601dc316f")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        match = NAME.fullmatch(row["filename"])
        if not match:
            raise ValueError(f"Unexpected filename {row['filename']!r} in {path.name}")
        row["video"], row["frame"] = match[1], int(match[2])
    return rows


def min_gap(first, second, videos):
    """Smallest frame-index distance between the folds within any shared video."""
    best = None
    for video in videos:
        frames = sorted({r["frame"] for r in second if r["video"] == video})
        for frame in {r["frame"] for r in first if r["video"] == video}:
            i = bisect.bisect_left(frames, frame)
            for j in (i - 1, i):
                if 0 <= j < len(frames):
                    gap = abs(frames[j] - frame)
                    best = gap if best is None else min(best, gap)
    return best


def summarize(splits):
    first, second = read(splits / "split_0.csv"), read(splits / "split_1.csv")
    videos = [{r["video"] for r in rows} for rows in (first, second)]
    shared = videos[0] & videos[1]
    return dict(
        rows=[len(first), len(second)],
        video_prefixes=[len(v) for v in videos],
        union_prefixes=len(videos[0] | videos[1]),
        shared_prefixes=len(shared),
        unique_filenames=[len({r["filename"] for r in rows}) for rows in (first, second)],
        filename_intersection=len({r["filename"] for r in first} & {r["filename"] for r in second}),
        shared_video_class_pairs=len({(r["video"], r["label"]) for r in first} & {(r["video"], r["label"]) for r in second}),
        min_cross_fold_frame_gap=min_gap(first, second, shared),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", type=Path, required=True, help="Kvasir-Capsule official_splits folder")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and (args.output.exists() or not args.output.parent.is_dir()):
        parser.error("--output must be a new file in an existing folder")
    try:
        values = summarize(args.splits)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(f"Input validation failed ({type(exc).__name__}: {exc})")
    mismatched = {k: (values[k], v) for k, v in REPORTED.items() if values[k] != v}
    if mismatched:
        sys.exit(f"Values differ from Appendix F.1 (computed, reported): {mismatched}")
    result = dict(status="all Appendix F.1 values reproduced", revision="553601dc316fef82f39c952e28f60aaad66039e2",
                  sha256=PINNED, **values)
    # Written only after every check has passed, so a failed run leaves no file.
    if args.output:
        with args.output.open("x", encoding="utf-8") as output:
            output.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
