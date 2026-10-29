#!/usr/bin/env python3
"""Replay retained numerical evidence, not training. Usage: python replay.py --output replay.json

Requires ReuScope analysis.py on the Python import path. Reads data/results by
default, or another copy given as a folder or zip. Primary/CNN/entry scores are summary reaggregations;
local endpoints are recomputed from probabilities; official BA from counts.
"""

import argparse
import csv
from collections import defaultdict
import gzip
import io
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import analysis


def close(a, b):
    if not np.allclose(a, b, atol=1e-10, rtol=0):
        raise ValueError(f"Reference mismatch: {a!r} != {b!r}")


def prefix_intervals(z):
    # Joint prefix draws share weights across arms/groups; seeds stay fixed.
    _, cluster = np.unique(z["video_prefix"], return_inverse=True)
    n = cluster.max() + 1
    weights = (
        np.random.default_rng(42)
        .multinomial(n, np.full(n, 1 / n), size=10000)
        .astype(float)
    )
    y, p = z["labels"], z["probabilities"]
    complete = np.ones(len(weights), dtype=bool)
    for group in ("E", "U"):
        for c in analysis.LOCAL_CLASSES:
            counts = np.bincount(cluster[(z["group"] == group) & (y == c)], minlength=n)
            complete &= (weights @ counts) > 0
    assert int(complete.sum()) == 914
    true = np.take_along_axis(
        p, np.broadcast_to(y[None, None, :, None], p.shape[:3] + (1,)), axis=3
    )[..., 0]
    other = p.copy()
    np.put_along_axis(
        other,
        np.broadcast_to(y[None, None, :, None], p.shape[:3] + (1,)),
        -np.inf,
        axis=3,
    )
    correct = (p.argmax(3) == y).astype(float)
    columns = dict(
        mean_true_class_prob=true,
        nll=-np.log(np.maximum(true, 1e-12)),
        confidence_margin=true - other.max(3),
        acc=correct,
        bal_acc=correct,
    )
    result = {}
    for endpoint, column in columns.items():
        values = np.empty((10000, len(z["seeds"]), 4, 2))
        for si in range(len(z["seeds"])):
            for ai in range(4):
                for gi, group in enumerate(("E", "U")):
                    mask = z["group"] == group
                    masks = (
                        [mask & (y == c) for c in np.unique(y[mask])]
                        if endpoint == "bal_acc"
                        else [mask]
                    )
                    recalls = []
                    for m in masks:
                        denom = weights @ np.bincount(cluster[m], minlength=n)
                        num = weights @ np.bincount(
                            cluster[m], weights=column[si, ai, m], minlength=n
                        )
                        recalls.append(
                            np.divide(
                                num,
                                denom,
                                out=np.full_like(num, np.nan),
                                where=denom > 0,
                            )
                        )
                    values[:, si, ai, gi] = np.nanmean(np.column_stack(recalls), axis=1)
        delta = values[:, :, 0] - values[:, :, 1:].mean(2)
        result[endpoint] = np.quantile(
            (delta[:, :, 0] - delta[:, :, 1]).mean(1), [0.025, 0.975]
        ).tolist()
    return dict(
        n_prefixes=int(n),
        resamples=10000,
        seed=42,
        fixed_six_support_draws=int(complete.sum()),
        ba_support="present classes per draw",
        intervals=result,
    )


def check_local_metadata(read, z):
    rows = list(
        csv.DictReader(
            io.StringIO(
                gzip.decompress(read("inputs/local_evaluation.csv.gz")).decode()
            )
        )
    )
    assert len(rows) == 1176
    assert [int(r["position"]) for r in rows] == list(range(1176))
    for key, column in (
        ("row_id", "row_id"),
        ("image_path", "image_path"),
        ("labels", "label"),
        ("group", "group"),
        ("video_prefix", "video_prefix"),
    ):
        assert np.array_equal(
            z[key], np.asarray([r[column] for r in rows], dtype=z[key].dtype)
        ), key
    return dict(
        rows=1176,
        aligned_fields=["row_id", "image_path", "labels", "group", "video_prefix"],
    )


def audit_replay(read):
    """Count released measurements and memberships; no fresh image processing."""
    import pandas as pd

    def csv(name):
        return pd.read_csv(io.BytesIO(read(name + ".csv.gz")), compression="gzip")

    train, val = (csv("inputs/" + name) for name in ("training", "validation"))
    base = pd.concat([train, val]).set_index("row_id", verify_integrity=True)
    assert len(train) == 37607 and len(val) == 16132
    assert base[list(analysis.CLASSES)].isin([0, 1]).all().all()
    assert base[list(analysis.CLASSES)].sum(axis=1).eq(1).all()
    pools = csv("inputs/primary_memberships")
    pool_counts = {
        "Full": len(train),
        **pools.groupby("pool", sort=False).size().to_dict(),
    }
    assert pools.row_id.isin(train.row_id).all()
    views = csv("inputs/evaluation_memberships")
    view_counts = {
        "public": len(val),
        **views.groupby("view", sort=False).size().to_dict(),
    }
    assert views.row_id.isin(val.row_id).all()
    for memberships in (pools, views):
        assert memberships.row_id.isin(base.index).all()
        key = "pool" if "pool" in memberships else "view"
        for _, rows in memberships.groupby(key, sort=False):
            assert rows.position.tolist() == list(range(len(rows)))
    members = csv("audit/byte_crossing_members").join(
        base[["image_path", "Dataset"]], on="row_id", validate="many_to_one"
    )
    byte_counts = {}
    for source, rows in members.groupby("Dataset"):
        groups = rows.groupby("group_position", sort=False)
        assert groups.sha256.nunique().eq(1).all()
        assert groups.label_sets_intersect.nunique().eq(1).all()
        split = rows.image_path.str.split("/").str[0]
        assert set(split) == {"training", "validation"}
        byte_counts[source] = dict(
            groups=int(groups.ngroups),
            validation_rows=int(rows.loc[split.eq("validation"), "row_id"].nunique()),
            intersect=int(groups.label_sets_intersect.first().sum()),
        )
    assert byte_counts == {
        "AIIMS": {"groups": 6, "validation_rows": 6, "intersect": 6},
        "SEE-AI": {"groups": 210, "validation_rows": 215, "intersect": 33},
    }
    prefixes = csv("audit/kvasir_prefix41")
    assert (
        len(prefixes) == 41
        and prefixes[["training_rows", "validation_rows"]].gt(0).all().all()
    )
    assert tuple(prefixes[["training_rows", "validation_rows"]].sum()) == (27011, 11581)
    pairs = csv("audit/internal_phash_pairs")
    selected = csv("audit/internal_validation_counters")
    any_count = int(pairs.loc[pairs.ncc96.ge(0.99), "val_row_id"].nunique())
    selected_count = int(selected.selected_ncc96.ge(0.99).sum())
    assert (len(pairs), pairs.val_row_id.nunique(), selected_count, any_count) == (
        6285,
        1381,
        540,
        799,
    )
    assert int((pairs.ncc96 >= 0.99).sum()) == 1715
    assert pairs.loc[pairs.ncc96.ge(0.99), "train_row_id"].nunique() == 1100
    alternate = csv("audit/alternate256_prefix_counts")
    assert alternate.selected_ncc99_256.sum() == 387
    source = csv("audit/source_selected_pairs")
    band = (
        source[["selected_pair_phash", "selected_pair_dhash"]].max(axis=1).between(3, 6)
    )
    assert (
        len(source),
        int(source.ncc96.ge(0.99).sum()),
        int(band.sum()),
        int((band & source.ncc96.ge(0.99)).sum()),
    ) == (38592, 35625, 2372, 858)
    filtered = csv("audit/filter_assignments")
    assert (
        len(filtered),
        int(filtered.removed.sum()),
        int(filtered.removed.eq(0).sum()),
    ) == (4551, 225, 4326)
    controls = csv("audit/ssim_control_pairs")
    ssim = {
        name: dict(
            rows=len(rows),
            median=float(rows.ssim96.median()),
            q05_q95=np.quantile(rows.ssim96, [0.05, 0.95]).tolist(),
            ncc99=int(rows.ncc96.ge(0.99).sum()),
        )
        for name, rows in controls.groupby("group")
    }
    assert all(x["rows"] == 1381 for x in ssim.values())
    # Joint prefix draws preserve both inventories' multiplicities (Appendix A.3).
    selected_rows = controls[controls.group.eq("A_flagged_internal")]
    for name, seed in [("B_adjacent_same_video", 42), ("C_distant_same_video", 43)]:
        groups = []
        for rows in (selected_rows, controls[controls.group.eq(name)]):
            assert rows.train_video.eq(rows.val_video).all()
            groups.append(
                {k: v.ssim96.to_numpy() for k, v in rows.groupby("val_video")}
            )
        names = sorted(set(groups[0]) | set(groups[1]))
        rng = np.random.default_rng(seed)
        gaps = []
        for _ in range(10000):
            draw = rng.choice(names, len(names), replace=True)
            gaps.append(
                np.median(
                    np.concatenate([groups[0][k] for k in draw if k in groups[0]])
                )
                - np.median(
                    np.concatenate([groups[1][k] for k in draw if k in groups[1]])
                )
            )
        ssim[name]["selected_minus_control_95"] = np.quantile(
            gaps, [0.025, 0.975]
        ).tolist()
    close(
        np.round(ssim["B_adjacent_same_video"]["selected_minus_control_95"], 3),
        [0.186, 0.285],
    )
    close(
        np.round(ssim["C_distant_same_video"]["selected_minus_control_95"], 3),
        [0.401, 0.487],
    )
    return dict(
        pool_rows=pool_counts,
        view_rows=view_counts,
        byte_crossings=byte_counts,
        shared_prefixes=41,
        exact_phash_pairs=6285,
        exact_phash_validation_rows=1381,
        selected_ncc99=selected_count,
        any_ncc99=any_count,
        alternate256_selected_ncc99=387,
        source_ncc99=35625,
        internal_removed=225,
        ssim=ssim,
    )


def replay(read):
    load = lambda name: json.loads(gzip.decompress(read("numerical/" + name)))
    records = load("seed_metrics.json.gz")
    assert len(records) == 450
    cells = defaultdict(dict)
    for r in records:
        m = r["metrics"]
        assert (
            m["n_test"]
            == {"Public": 16132, "Non-KV": 4551, "Filtered": 4326}[r["view"]]
        )
        assert set(m["per_class"]) == set(analysis.CLASSES)
        assert sum(c["n"] for c in m["per_class"].values()) == m["n_test"]
        close(m["bal_acc"], np.mean([c["recall"] for c in m["per_class"].values()]))
        key = (r["family"], r["backbone"], r["arm"], r["view"])
        assert r["seed"] not in cells[key]
        cells[key][r["seed"]] = r["metrics"]
    assert len(cells) == 45 and all(set(v) == set(range(10)) for v in cells.values())
    summary = {}
    contrasts = {}
    for key, values in cells.items():
        summary["/".join(key)] = {
            m: float(np.mean([values[s][m] for s in range(10)]))
            for m in ("bal_acc", "acc", "mean_auc_ovr", "combined_ovr")
        }
    reference = load("reference_checks.json.gz")
    targets = {
        ("primary", "dinov2_vitl14", name, "Public")
        for name in reference["primary_contrasts"]
    }
    targets.update(
        ("primary", "dinov2_vitl14", name, view)
        for name in ("Removal minus Full", "Source-only minus Random", "A minus Comp")
        for view in ("Public", "Non-KV")
    )
    targets.add(("primary", "dinov2_vitl14", "Removal minus Full", "Filtered"))
    targets.update(
        ("cnn", backbone, "Source-only minus Random", view)
        for backbone in ("resnet50", "convnext_tiny")
        for view in ("Public", "Non-KV")
    )
    for family, backbone, name, view in sorted(targets):
        left, right = name.split(" minus ")
        values = [
            cells[family, backbone, left, view][s]["bal_acc"]
            - cells[family, backbone, right, view][s]["bal_acc"]
            for s in range(10)
        ]
        contrasts["/".join((family, backbone, name, view))] = analysis.paired_intervals(
            values
        )
    for name, saved in reference["primary_contrasts"].items():
        actual = contrasts["primary/dinov2_vitl14/" + name + "/Public"]
        for k in ("values", "mean", "paired_t_95_ci", "paired_bootstrap_95_ci"):
            close(actual[k], saved[k])
    for r in reference["cnn_contrasts"]:
        if r["metric"] != "bal_acc" or r["contrast"] != "Source-only minus Random":
            continue
        left, right = r["contrast"].split(" minus ")
        view = {"orig_val": "Public", "dedup_val": "Non-KV"}[r["pool"]]
        v = [
            cells["cnn", r["backbone"], left, view][s][r["metric"]]
            - cells["cnn", r["backbone"], right, view][s][r["metric"]]
            for s in range(10)
        ]
        actual = analysis.paired_intervals(v)
        close(actual["mean"], r["expected"]["mean"])
        close(actual["paired_t_95_ci"], r["expected"]["ci95"])
    with np.load(
        io.BytesIO(read("numerical/local_probabilities.npz")), allow_pickle=False
    ) as data:
        z = dict(data)
    assert z["probabilities"].shape == (10, 4, 1176, 10)
    assert list(z["classes"]) == list(analysis.CLASSES) and list(z["arms"]) == list(
        analysis.ARMS
    )
    assert set(z["seeds"]) == {4, 5, 6, 7, 8, 9, 14, 15, 16, 17}
    metadata = check_local_metadata(read, z)
    by_seed_arm = {
        (int(s), str(a)): dict(
            seed=int(s),
            arm=str(a),
            probabilities=z["probabilities"][si, ai],
            **{k: z[k] for k in ("row_id", "labels", "group", "video_prefix")},
        )
        for si, s in enumerate(z["seeds"])
        for ai, a in enumerate(z["arms"])
    }
    local = analysis.local_contrast(by_seed_arm)
    prefix = prefix_intervals(z)
    names = dict(
        true_class_probability="mean_true_class_prob",
        balanced_accuracy="bal_acc",
        accuracy="acc",
        nll="nll",
        confidence_margin="confidence_margin",
    )
    for old, new in names.items():
        for k in ("values", "mean", "paired_t_95_ci", "paired_bootstrap_95_ci"):
            close(
                local["endpoints"][new]["E_minus_U"][k],
                reference["local_endpoints"][old][k],
            )
        close(
            prefix["intervals"][new],
            reference["local_endpoints"][old]["video_prefix_cluster_bootstrap_95_ci"],
        )
    official = load("official_test.json.gz")
    assert official["classes"] == list(analysis.CLASSES)
    assert len(official["records"]) == 30
    off = {}
    for arm in sorted({r["arm"] for r in official["records"]}):
        rows = [r for r in official["records"] if r["arm"] == arm]
        assert len(rows) == 10 and {r["seed"] for r in rows} == set(range(10))
        assert all(np.asarray(r["confusion"]).sum() == 4385 for r in rows)
        off[arm] = analysis.official_aggregates(
            {r["seed"]: r["confusion"] for r in rows},
            {r["seed"]: r["class_auc"] for r in rows},
        )
    # Seed-paired official-test contrasts (Table 1c, Appendix B.2).
    by_seed = {
        (arm, m): dict(zip(off[arm]["seeds"], off[arm][m]["values"]))
        for arm in off
        for m in ("bal_acc", "mean_auc_ovr", "combined_ovr")
    }
    official_paired = {
        f"{a} - {b}": {
            m: analysis.paired_intervals(
                [by_seed[a, m][s] - by_seed[b, m][s] for s in range(10)]
            )
            for m in ("bal_acc", "mean_auc_ovr", "combined_ovr")
        }
        for a, b in (("Source-only", "Full"), ("Source-only", "Random"))
    }
    entry = load("entry_scores.json.gz")
    assert len(entry) == 72
    scores = {(r["entry"], r["view"]): r for r in entry}
    assert len(scores) == 72
    cohorts = {}
    for n in (24, 19):
        entries = sorted(
            {r["entry"] for r in entry}
            - (set("Team " + c for c in "UVWXY") if n == 19 else set())
        )
        assert len(entries) == n
        means = {
            view: {
                m: float(np.mean([scores[e, view][m] for e in entries]))
                for m in (
                    "pooled_combined",
                    "equal_source_bal_acc",
                    "equal_source_combined",
                )
            }
            for view in ("Public", "Non-KV", "Filtered")
        }
        changes = {
            view: {
                m: float(
                    np.mean(
                        [scores[e, view][m] - scores[e, "Public"][m] for e in entries]
                    )
                )
                for m in means[view]
            }
            for view in ("Non-KV", "Filtered")
        }
        cohorts[str(n)] = dict(entries=entries, means=means, changes=changes)
    return dict(
        status="passed",
        coverage=dict(
            primary_fits=110,
            primary_seed_views=330,
            cnn_fits=60,
            cnn_seed_views=120,
            official_fits=30,
            local_fits=40,
            local_prediction_rows=47040,
            entry_views=72,
        ),
        summary_means=summary,
        paired_ba_contrasts=contrasts,
        official=off,
        official_paired=official_paired,
        entries=cohorts,
        local=local,
        local_prefix=prefix,
        local_metadata=metadata,
        audit=audit_replay(read),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="?", default=Path(__file__).resolve().parent / "data" / "results")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if __import__("sys").flags.optimize:
        parser.error("Run without -O; validation assertions are required")
    if args.output and (args.output.exists() or not args.output.parent.is_dir()):
        parser.error("--output must be a new file in an existing folder")
    # Known-answer metric check, including argmax ties.
    p = np.zeros((4, 10))
    p[:, :2] = [[0.8, 0.2], [0.5, 0.5], [0.2, 0.8], [0.6, 0.4]]
    toy = analysis.metrics(np.array([0, 0, 1, 1]), p, required_classes=None)
    close(toy["bal_acc"], 0.75)
    close(toy["mean_auc_ovr"], 0.75)
    if args.results.is_dir():
        result = replay(lambda name: (args.results / name).read_bytes())
    else:
        with ZipFile(args.results) as archive:
            result = replay(archive.read)
    # Written only after every check has passed, so a failed run leaves no file.
    if args.output:
        with args.output.open("x", encoding="utf-8") as output:
            output.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            dict(
                status=result["status"],
                coverage=result["coverage"],
                local_probability=result["local"]["endpoints"][
                    "mean_true_class_prob"
                ]["E_minus_U"],
                official_paired=result["official_paired"],
                entries=result["entries"],
                audit=result["audit"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
