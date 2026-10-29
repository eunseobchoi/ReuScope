#!/usr/bin/env python3
"""Rescore upstream validation workbooks as raw scores (Tables 1d and 13–15).

Install the existing ReuScope CPU requirements; no Excel engine is needed.
Obtain the external data without importing or redistributing organizer code:
  git clone --filter=blob:none --no-checkout https://github.com/palaxydf/Capsule-Vision-2024-Challenge external
  git -C external sparse-checkout init --no-cone
  git -C external sparse-checkout set /Results/validation_data.xlsx /Results/submitted_excel_files/validation/ /Results/metrics_reports/metrics_reports_val/ /Results/metrics_reports/metrics_reports_test/
  git -C external checkout --detach de38787e8267f763a5f59695c67bb6d28ece70bf
  python rescore.py --results-dir external/Results --output rescored.json

Keep submissions.json beside this script. Its aliases and hashes identify the
fixed 27 workbooks, 25 scoreable entries and organizer validation/test reports.
Workbooks remain upstream. Official-test scores are read from reports, not
recomputed from test images. BCa ranges are descriptive entry resampling with
fixed frames; independence of entries is not established.
"""
import argparse
from contextlib import ExitStack
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import sys
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.stats import bootstrap, rankdata, spearmanr
import analysis

# Minimal standard-library XLSX reader (no organizer code, no Excel engine).
CELL_REF = re.compile(r"([A-Z]+)")


def column_index(cell_reference):
    match = CELL_REF.match(cell_reference)
    if match is None:
        raise ValueError("Invalid spreadsheet cell reference")
    value = 0
    for character in match.group(1):
        value = value * 26 + ord(character) - ord("A") + 1
    return value - 1


def shared_strings(archive):
    name = "xl/sharedStrings.xml"
    if name not in archive.namelist():
        return []
    root = ElementTree.fromstring(archive.read(name))
    return ["".join(node.text or "" for node in item.iter() if node.tag.endswith("}t"))
            for item in root if item.tag.endswith("}si")]


def cell_value(cell, strings):
    if cell.attrib.get("t") == "inlineStr":
        return "".join(node.text or "" for node in cell.iter() if node.tag.endswith("}t"))
    node = next((node for node in cell if node.tag.endswith("}v")), None)
    if node is None or node.text is None:
        return ""
    return strings[int(node.text)] if cell.attrib.get("t") == "s" else node.text


def iter_xlsx_rows(path):
    with ZipFile(path) as archive:
        strings = shared_strings(archive)
        sheets = sorted(name for name in archive.namelist()
                        if name.startswith("xl/worksheets/sheet") and name.endswith(".xml"))
        if not sheets:
            raise ValueError("No worksheet")
        with archive.open(sheets[0]) as handle:
            for _, element in ElementTree.iterparse(handle, events=("end",)):
                if not element.tag.endswith("}row"):
                    continue
                values = {column_index(cell.attrib["r"]): cell_value(cell, strings)
                          for cell in element if cell.tag.endswith("}c")}
                if values:
                    yield [values.get(index, "") for index in range(max(values) + 1)]
                element.clear()


def workbook(path):
    rows = iter(iter_xlsx_rows(path))
    header = next(rows)
    if len(header) != len(set(header)) or not {"image_path", *analysis.CLASSES} <= set(header):
        raise ValueError("Invalid workbook columns")
    result = []
    for row in rows:
        if len(row) > len(header):
            raise ValueError("Unexpected workbook columns")
        result.append(dict(zip(header, row + [""] * (len(header) - len(row)))))
    return result


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def hashed_files(directory, pattern):
    result = {}
    for path in directory.glob(pattern):
        digest = sha256(path)
        result[digest] = path
    return result


def normalize(path):
    return path.replace("\\", "/")


def aligned_scores(rows, paths):
    keys = [normalize(row["image_path"]) for row in rows]
    if (len(keys) != len(set(keys)) or set(keys) != set(paths)
            or any(value == "" for row in rows for value in row.values())):
        raise ValueError("Incomplete or invalid validation workbook")
    bypath = dict(zip(keys, rows))
    values = np.array([[float(bypath[path][c]) for c in analysis.CLASSES] for path in paths])
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite validation scores")
    return values


def equal(actual, expected):
    if not np.allclose(actual, expected, atol=1e-12, rtol=0):
        raise ValueError("Retained numerical reference mismatch")


def report(path, expected):
    data = json.loads(path.read_text())
    result = {key: data[key] for key in ("mean_auc", "balanced_accuracy")}
    for key, value in result.items():
        if not isinstance(value, (float, int)) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid organizer metric")
        equal(value, expected[key])
    result["combined"] = (result["mean_auc"] + result["balanced_accuracy"]) / 2
    return result


def aggregate(entries, cohorts):
    # One advancing RNG (seed 42) shared across views, then cohorts, as in the paper's analysis.
    rng = np.random.default_rng(42)
    output = {}
    for view in ("Non-KV", "Filtered"):
        ordered = sorted(entries, key=lambda r: -r["views"][view]["pooled"]["combined_ovr"])
        for group, aliases in cohorts.items():
            rows = [r for r in ordered if r["entry"] in aliases]
            assert len(rows) == len(aliases)
            original = np.array([r["organizer_validation"]["combined"] for r in rows])
            subset = np.array([r["views"][view]["pooled"]["combined_ovr"] for r in rows])
            official = np.array([r["organizer_official"]["combined"] for r in rows])
            delta = subset - original
            ci = bootstrap((delta,), np.mean, n_resamples=10000, method="BCa",
                           random_state=rng).confidence_interval
            changes = {}
            for name, weighting, metric in (
                    ("pooled_combined", "pooled", "combined_ovr"),
                    ("equal_source_bal_acc", "equal_source", "bal_acc"),
                    ("equal_source_combined", "equal_source", "combined_ovr")):
                changes[name] = float(np.mean([
                    r["views"][view][weighting][metric] - r["views"]["Public"][weighting][metric]
                    for r in rows]))
            output[view + "/" + group] = dict(
                n=len(rows), entries=[r["entry"] for r in rows],
                mean_delta=float(delta.mean()), bca95=[float(ci.low), float(ci.high)],
                rank_changed=int(np.sum(rankdata(-original) != rankdata(-subset))),
                spearman_official_full=float(spearmanr(original, official).statistic),
                spearman_official_view=float(spearmanr(subset, official).statistic),
                raw_score_mean_changes=changes)
    return output


def rescore(results_dir, read, manifest):
    validation = pd.read_csv(io.BytesIO(read("inputs/validation.csv.gz")), compression="gzip")
    memberships = pd.read_csv(io.BytesIO(read("inputs/evaluation_memberships.csv.gz")), compression="gzip")
    assert len(validation) == 16132 and validation.row_id.is_unique and validation.image_path.is_unique
    paths = validation.image_path.tolist()
    onehot = validation[list(analysis.CLASSES)].to_numpy()
    assert np.isin(onehot, [0, 1]).all() and (onehot.sum(1) == 1).all()
    labels = onehot.argmax(1)
    gt_path = results_dir / "validation_data.xlsx"
    assert sha256(gt_path) == manifest["validation_workbook_sha256"]
    gt = workbook(gt_path)
    assert [normalize(r["image_path"]) for r in gt] == paths
    assert np.array_equal(aligned_scores(gt, paths), onehot)
    assert [r["Dataset"] for r in gt] == validation.Dataset.tolist()
    views = {}
    for name, rows in memberships.groupby("view", sort=False):
        assert rows.position.tolist() == list(range(len(rows))) and rows.row_id.is_unique
        assert rows.row_id.isin(validation.row_id).all()
        views[{"non_kvasir": "Non-KV", "filtered": "Filtered"}[name]] = rows.row_id.to_numpy()
    assert {k: len(v) for k, v in views.items()} == {"Non-KV": 4551, "Filtered": 4326}
    workbooks = hashed_files(results_dir / "submitted_excel_files/validation", "*.xlsx")
    validation_reports = hashed_files(results_dir / "metrics_reports/metrics_reports_val", "*.json")
    official_reports = hashed_files(results_dir / "metrics_reports/metrics_reports_test", "*.json")
    entries, excluded = [], []
    assert len(manifest["submissions"]) == 27
    for item in manifest["submissions"]:
        alias = item["alias"]
        if item["workbook_sha256"] not in workbooks:
            raise ValueError(f"{alias}: pinned workbook missing or modified")
        rows = workbook(workbooks[item["workbook_sha256"]])
        if not item["scoreable"]:
            try:
                aligned_scores(rows, paths)
            except (ValueError, KeyError):
                excluded.append(dict(entry=alias, reason="Incomplete or invalid validation workbook"))
                continue
            raise ValueError(f"{alias}: expected invalid workbook")
        scores = aligned_scores(rows, paths)
        scored = analysis.score_views(dict(row_id=validation.row_id.to_numpy(),
                    labels=labels, source=validation.Dataset.to_numpy(), scores=scores), views)
        scored["Public"] = scored.pop("full")
        entries.append(dict(entry=alias, views=scored,
            organizer_validation=report(validation_reports[item["validation_report_sha256"]],
                                        item["organizer_validation"]),
            organizer_official=report(official_reports[item["official_report_sha256"]],
                                      item["organizer_official"])))
    assert len(entries) == 25 and {r["entry"] for r in excluded} == {"Team Z", "Team AA"}
    drift = [r["entry"] for r in entries if max(
        abs(r["organizer_validation"][organizer] - r["views"]["Public"]["pooled"][metric])
        for organizer, metric in (("mean_auc", "mean_auc_ovr"), ("balanced_accuracy", "bal_acc"))
        ) > 1e-10]
    assert drift == ["Team G"]
    retained = json.loads(gzip.decompress(read("numerical/entry_scores.json.gz")))
    by_alias = {r["entry"]: r for r in entries}
    assert len(retained) == 72 and len({(r["entry"], r["view"]) for r in retained}) == 72
    for row in retained:
        actual = by_alias[row["entry"]]["views"][row["view"]]
        equal(actual["pooled"]["combined_ovr"], row["pooled_combined"])
        equal(actual["equal_source"]["bal_acc"], row["equal_source_bal_acc"])
        equal(actual["equal_source"]["combined_ovr"], row["equal_source_combined"])
    cohorts = manifest["cohorts"]
    assert list(cohorts) == ["all_25", "primary_24", "subset_19", "subset_without_A_18"]
    assert [len(v) for v in cohorts.values()] == [25, 24, 19, 18]
    return dict(status="passed", scoreable=25, excluded=excluded,
                reported_full_score_mismatch=drift, checked_core_entry_views=72,
                entries=entries, aggregates=aggregate(entries, cohorts),
                bca=dict(seed=42, resamples=10000, method="BCa",
                         order="Non-KV then Filtered; 25,24,19,18; descending unrounded view score",
                         scope="Descriptive entry resampling, fixed frames; dependence unestablished"),
                official_test="Organizer aggregate reports only; no test-image inference")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent / "data" / "results")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.flags.optimize:
        parser.error("Run without -O; validation assertions are required")
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("--output must be a new file in an existing folder")
    with ExitStack() as stack:
        try:
            manifest = json.loads(Path(__file__).with_name("submissions.json").read_text())
            read = ((lambda name: (args.data / name).read_bytes()) if args.data.is_dir()
                    else stack.enter_context(ZipFile(args.data)).read)
            result = rescore(args.results_dir, read, manifest)
        except (OSError, ValueError, KeyError, AssertionError, StopIteration) as exc:
            detail = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
            parser.error(f"Input validation failed ({detail}); check pinned upstream files, "
                         "submissions.json and data/results")
    # Written only after every check has passed, so a failed run leaves no file.
    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "scoreable", "excluded",
                     "reported_full_score_mismatch", "checked_core_entry_views")}, indent=2))


if __name__ == "__main__":
    main()
