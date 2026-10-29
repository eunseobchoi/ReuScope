"""Ordered main-paper controls from caller-owned one-hot pandas DataFrames."""
from pathlib import PurePosixPath

import numpy as np
import pandas as pd

CLASSES = ["Angioectasia", "Bleeding", "Erosion", "Erythema", "Foreign Body",
           "Lymphangiectasia", "Normal", "Polyp", "Ulcer", "Worms"]
SOURCES = ["KVASIR", "SEE-AI", "KID", "AIIMS"]


def _keys(frame):
    paths = frame.image_path
    if not paths.map(lambda p: isinstance(p, str) and bool(p.strip())
                     and p == p.strip() and "\x00" not in p).all():
        raise ValueError("image_path must contain nonempty path strings")
    keys = paths.str.replace("\\", "/", regex=False)
    if any(key == "." or ".." in PurePosixPath(key).parts for key in keys):
        raise ValueError("image_path must name a file without parent traversal")
    return keys


def _validate(frame):
    required = ["image_path", "Dataset", *CLASSES]
    if not frame.columns.is_unique or not set(required).issubset(frame.columns):
        raise ValueError("Expected image_path, Dataset and ten one-hot class columns")
    if frame.empty or not frame.Dataset.isin(SOURCES).all():
        raise ValueError("Nonempty input with literal KVASIR/SEE-AI/KID/AIIMS sources required")
    labels = frame[CLASSES]
    if not labels.isin([0, 1]).all().all() or not labels.sum(axis=1).eq(1).all():
        raise ValueError("Each row must have exactly one one-hot class")
    rows = frame[required].assign(image_path=_keys(frame)).drop_duplicates()
    if rows.image_path.duplicated().any():
        raise ValueError("Repeated paths must retain the same source and class")


def _align(measurements, frame, fields):
    """Accept reordered measurements, but reject missing/extra/conflicting paths."""
    if (not measurements.columns.is_unique
            or not {"image_path", *fields}.issubset(measurements.columns)):
        raise ValueError(f"Measurements require image_path and {fields}")
    identity = [c for c in ["Dataset", *CLASSES] if c in measurements]
    rows = measurements[["image_path", *identity, *fields]].copy()
    rows["image_path"] = _keys(rows)
    rows = rows.drop_duplicates()
    if rows.image_path.duplicated().any():
        raise ValueError("Conflicting measurements for the same path")
    keys = _keys(frame)
    if set(rows.image_path) != set(keys):
        raise ValueError("Measurement paths must exactly match the input")
    aligned = rows.set_index("image_path").loc[keys].reset_index(drop=True)
    for column in identity:
        if not aligned[column].eq(frame[column].reset_index(drop=True)).all():
            raise ValueError(f"Measurement {column} does not match input")
    return aligned


def _matched(full, target, advancing=False):
    rng = np.random.RandomState(0) if advancing else 0
    pieces = []
    for label in CLASSES:
        n = int(target[label].sum())
        kv = full[full[label].eq(1) & full.Dataset.eq("KVASIR")]
        other = full[full[label].eq(1) & full.Dataset.ne("KVASIR")]
        take = min(n, len(kv))
        if take:
            pieces.append(kv.sample(n=take, random_state=rng))
        if n > take:
            pieces.append(other.sample(n=n-take, random_state=rng))
    if not pieces:
        return full.iloc[:0].copy()
    return pd.concat(pieces, ignore_index=True).sample(frac=1, random_state=0).reset_index(drop=True)


def make_pools(train, validation, training_screen=None, train_hashes=None,
               validation_hashes=None):
    """Return Full/Source-only/Random/Comp/A and public/non-KVASIR views.

    Supply reference_minima(train, reference) for Removal/Class-match-R, and
    measure_images outputs for both splits for the filtered view. Independent
    min_phash AND min_dhash <= 6 define removal; the witnesses may differ.
    All draws use seed 0; A requires the paper's 66 KVASIR/66 AIIMS Ulcer rows.
    Inputs are never sorted or deduplicated. Repeated measurement paths must agree.
    """
    _validate(train)
    _validate(validation)
    train, validation = train.reset_index(drop=True), validation.reset_index(drop=True)
    source = train[train.Dataset.ne("KVASIR")].reset_index(drop=True)
    comp = _matched(train, source)
    kv_ulcer = comp.Dataset.eq("KVASIR") & comp.Ulcer.eq(1)
    aiims = train[train.Dataset.eq("AIIMS") & train.Ulcer.eq(1)]
    if kv_ulcer.sum() != 66 or len(aiims) != 66 or (comp.Dataset.eq("AIIMS") & comp.Ulcer.eq(1)).any():
        raise ValueError("Ulcer-swap requires exactly 66 KVASIR Ulcer in Comp and 66 AIIMS Ulcer in Full")
    nonkv = validation[validation.Dataset.ne("KVASIR")].reset_index(drop=True)
    pools = {"full": train.copy(), "source_only": source,
             "random": train.sample(n=len(source), random_state=0).reset_index(drop=True),
             "composition": comp,
             "ulcer_swap": pd.concat([comp[~kv_ulcer], aiims], ignore_index=True),
             "public": validation.copy(), "non_kvasir": nonkv}
    if training_screen is not None:
        distances = _align(training_screen, train, ["min_phash", "min_dhash"])
        values = distances[["min_phash", "min_dhash"]].apply(pd.to_numeric, errors="raise")
        if not np.isfinite(values).all().all() or not ((values >= 0) & (values <= 64) & (values % 1 == 0)).all().all():
            raise ValueError("Independent hash minima must be integers in [0, 64]")
        flagged = (values.min_phash <= 6) & (values.min_dhash <= 6)
        if not flagged[train.Dataset.eq("KVASIR")].all():
            raise ValueError("Removal requires the reference screen to flag every KVASIR training row")
        removal = train[~flagged].reset_index(drop=True)
        pools.update(removal=removal, class_match=_matched(train, removal, advancing=True))
    if (train_hashes is None) != (validation_hashes is None):
        raise ValueError("Supply both training and validation pHash measurements")
    if train_hashes is not None:
        hashes = [_align(rows, frame, ["phash"])
                  for rows, frame in [(train_hashes, train), (validation_hashes, validation)]]
        for rows, frame in zip(hashes, [train, validation]):
            if not rows.phash.map(lambda h: isinstance(h, str)).all() or not rows.phash.str.fullmatch(r"[0-9a-f]{16}").all():
                raise ValueError("Complete lowercase 64-bit pHash measurements required")
            rows["Dataset"] = frame.Dataset
            rows["basename"] = _keys(frame).map(lambda p: PurePosixPath(p).name)
        dropped = set()
        for source in SOURCES[1:]:
            tr, va = [rows[rows.Dataset.eq(source)] for rows in hashes]
            flagged = va.basename.isin(tr.basename) | va.phash.isin(tr.phash)
            dropped.update(va.loc[flagged, "basename"])
        pools["filtered"] = nonkv[~_keys(nonkv).map(lambda p: PurePosixPath(p).name).isin(dropped)].reset_index(drop=True)
    return pools
