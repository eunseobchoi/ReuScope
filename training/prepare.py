"""Export the ordered training and evaluation lists used by the paper's runs.

Reads data/results and writes 19 CSVs (11 primary pools, 3 validation
views, 4 local-removal training lists and the local evaluation list).
Compare the output with lists.sha256.
"""

import argparse
import io
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import pandas as pd

CLASSES = (
    "Angioectasia",
    "Bleeding",
    "Erosion",
    "Erythema",
    "Foreign Body",
    "Lymphangiectasia",
    "Normal",
    "Polyp",
    "Ulcer",
    "Worms",
)
POOLS = {
    "Random": 10596,
    "Source-only": 10596,
    "Removal": 10591,
    "Class-match-R": 10591,
    "Comp": 10596,
    "A": 10596,
    "B": 10596,
    "C": 10596,
    "D-dup": 10596,
    "D-uniq": 10596,
}
COLUMNS = ["image_path", "Dataset", *CLASSES]


def prepare(read):
    def csv(name):
        return pd.read_csv(
            io.BytesIO(read("inputs/" + name + ".csv.gz")), compression="gzip"
        )

    train, val = (csv(name) for name in ("training", "validation"))
    for frame, count in ((train, 37607), (val, 16132)):
        if len(frame) != count or not frame.row_id.is_unique:
            raise ValueError("Unexpected base population or duplicate row IDs")
        if (
            not frame[list(CLASSES)].isin([0, 1]).all().all()
            or not frame[list(CLASSES)].sum(axis=1).eq(1).all()
        ):
            raise ValueError(
                "Labels must be binary one-hot rows in the ten-class order"
            )
        if not frame.Dataset.isin(["KVASIR", "SEE-AI", "KID", "AIIMS"]).all():
            raise ValueError("Unknown source label")
        for path in frame.image_path:
            if (
                not isinstance(path, str)
                or not path.strip()
                or "\x00" in path
                or PurePosixPath(path).is_absolute()
                or ".." in PurePosixPath(path).parts
                or "\\" in path
                or ":" in path
            ):
                raise ValueError("Expected Dataset-relative POSIX image paths")
    t, v = (
        train.set_index("row_id", verify_integrity=True),
        val.set_index("row_id", verify_integrity=True),
    )
    outputs = {"primary/Full.csv": train[COLUMNS], "views/Public.csv": val[COLUMNS]}
    membership = csv("primary_memberships")
    if set(membership.pool) != set(POOLS):
        raise ValueError("Unexpected primary pool names")
    for name, rows in membership.groupby("pool", sort=False):
        if len(rows) != POOLS[name] or rows.position.tolist() != list(range(len(rows))):
            raise ValueError("Pool count/order mismatch: " + name)
        outputs["primary/" + name + ".csv"] = t.loc[rows.row_id, COLUMNS].reset_index(
            drop=True
        )
    views = csv("evaluation_memberships")
    names = {"non_kvasir": ("Non-KV", 4551), "filtered": ("Filtered", 4326)}
    if set(views.view) != set(names):
        raise ValueError("Unexpected evaluation view names")
    for key, rows in views.groupby("view", sort=False):
        name, count = names[key]
        if (
            len(rows) != count
            or rows.position.tolist() != list(range(count))
            or not rows.row_id.is_unique
        ):
            raise ValueError("View count/order mismatch: " + name)
        outputs["views/" + name + ".csv"] = v.loc[rows.row_id, COLUMNS].reset_index(
            drop=True
        )
    removals = csv("local_removals")
    roles = {name: set(rows.row_id) for name, rows in removals.groupby("role")}
    expected = {
        "common1100": 1100,
        "local876": 876,
        "distant101": 876,
        "distant202": 876,
        "distant303": 876,
    }
    if set(roles) != set(expected):
        raise ValueError("Unexpected local removal roles")
    for name, rows in removals.groupby("role"):
        if (
            len(rows) != expected[name]
            or len(roles[name]) != len(rows)
            or not roles[name] <= set(t.index)
        ):
            raise ValueError("Invalid removal set: " + name)
    for arm, role in [
        ("local", "local876"),
        ("draw101", "distant101"),
        ("draw202", "distant202"),
        ("draw303", "distant303"),
    ]:
        if roles["common1100"] & roles[role]:
            raise ValueError("Overlapping common/additional removals")
        kept = train.loc[~train.row_id.isin(roles["common1100"] | roles[role]), COLUMNS]
        if len(kept) != 35631:
            raise ValueError("Local training population mismatch")
        outputs["local/" + arm + ".csv"] = kept
    local = csv("local_evaluation")
    if (
        len(local) != 1176
        or not local.base_row_id.is_unique
        or local.position.tolist() != list(range(1176))
    ):
        raise ValueError("Local evaluation population/order mismatch")
    if local.group.value_counts().to_dict() != {"E": 588, "U": 588}:
        raise ValueError("Local E/U counts differ")
    paired = v.loc[local.base_row_id, COLUMNS].reset_index(drop=True)
    if (
        paired.image_path.tolist() != local.image_path.tolist()
        or not paired.Dataset.eq("KVASIR").all()
    ):
        raise ValueError("Local metadata do not align with validation")
    # The recorded local trainer uses Dataset for E/U strata in this one input.
    paired["Dataset"] = local.group.map(
        {"E": "NCC99_EXPOSED", "U": "MATCHED_UNFLAGGED"}
    ).to_numpy()
    outputs["local/Evaluation.csv"] = paired
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new output directory")
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "results",
                        help="results folder or zip (default: data/results)")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory already exists")
    if not args.output.parent.is_dir():
        parser.error("Output parent must exist")
    if args.data.is_dir():
        outputs = prepare(lambda name: (args.data / name).read_bytes())
    else:
        with ZipFile(args.data) as archive:
            outputs = prepare(archive.read)
    args.output.mkdir()
    for name, frame in outputs.items():
        p = args.output / name
        p.parent.mkdir(exist_ok=True)
        frame.to_csv(p, index=False, lineterminator="\n")
    print(json.dumps({name: len(frame) for name, frame in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
