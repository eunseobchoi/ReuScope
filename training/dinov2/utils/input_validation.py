"""Shared validation for externally supplied training lists and output paths."""
import csv
from pathlib import Path


def read_one_hot_csv(path, classes, *, allow_filename=False):
    """Read a nonempty one-hot list without changing its row order or duplicates."""
    import numpy as np
    import pandas as pd

    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle, strict=True)
        try:
            header = next(reader, [])
            if not header or len(header) != len(set(header)):
                raise ValueError(f"Expected unique CSV headers in {path}")
            path_column = "image_path" if "image_path" in header else "filename"
            if "image_path" not in header and not (allow_filename and "filename" in header):
                raise ValueError(f"Expected image_path column in {path}")
            label_cols = [c for c in header if c in classes]
            if not label_cols:
                raise ValueError(f"No class columns in {path}")
            for row in reader:
                if len(row) != len(header):
                    raise ValueError(f"CSV row width differs from header at line {reader.line_num} in {path}")
                for name in (path_column, "Dataset"):
                    if name in header:
                        value = row[header.index(name)]
                        if not value.strip() or "\x00" in value:
                            raise ValueError(f"Expected nonblank, NUL-free {name} strings in {path}")
        except csv.Error as exc:
            raise ValueError(f"Malformed CSV in {path}: {exc}") from exc
        handle.seek(0)
        df = pd.read_csv(handle, dtype={path_column: str, "Dataset": str}, keep_default_na=False, engine="python")
        if path_column != "image_path":
            df = df.rename(columns={path_column: "image_path"})
    try:
        labels = df[label_cols].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected finite binary one-hot labels in {path}") from exc
    if (not len(df) or not np.isfinite(labels).all()
            or not np.isin(labels, [0, 1]).all()
            or not (labels.sum(axis=1) == 1).all()):
        raise ValueError(f"Expected nonempty finite binary one-hot labels in {path}")
    df[label_cols] = labels
    return df.reset_index(drop=True)


def validate_output_paths(paths):
    """Reject output collisions and unusable parents before expensive work."""
    raw_paths = [Path(p) for p in paths if p is not None]
    outputs = [p.resolve() for p in raw_paths]
    if len(outputs) != len(set(outputs)):
        raise ValueError("Output paths must be distinct")
    for original, candidate in zip(raw_paths, outputs):
        if original.exists() or original.is_symlink() or candidate.exists():
            raise ValueError(f"Refusing to replace prior output: {original}; use a fresh run directory")
        if any(parent in outputs for parent in candidate.parents):
            raise ValueError("Output paths must not be ancestors of other outputs")
        for parent in original.absolute().parents:
            if (parent.exists() or parent.is_symlink()) and not parent.is_dir():
                raise ValueError(f"Output parent is not a directory: {parent}")
