"""Core reuse measurements for caller-supplied image tables (CPU only).

Public rows contain image_path, Dataset and exactly one active class in CLASSES.
Row-wise measurement functions preserve row order, index and multiplicity;
summary functions aggregate those rows. Relative paths are resolved against root.
Hashing reads files; hash comparisons trust supplied measurements.
Recording prefixes are not patient identities.
"""
from collections import Counter, defaultdict
import hashlib
import io
from pathlib import Path, PurePosixPath
import re

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

CLASSES = ["Angioectasia", "Bleeding", "Erosion", "Erythema", "Foreign Body",
           "Lymphangiectasia", "Normal", "Polyp", "Ulcer", "Worms"]
SOURCES = ["KVASIR", "SEE-AI", "KID", "AIIMS"]
POP16 = np.array([i.bit_count() for i in range(65536)], dtype=np.uint8)


def _key(path):
    return path.replace("\\", "/")


def _validate(rows, hashes=(), labels=True):
    fields = ["image_path", *(["Dataset", *CLASSES] if labels else []), *hashes]
    if not rows.columns.is_unique or not set(fields).issubset(rows.columns):
        raise ValueError(f"Expected unique columns including {fields}")
    for field in ["image_path", *(["Dataset"] if labels else [])]:
        if not rows[field].map(lambda x: isinstance(x, str) and bool(x.strip())
                               and x == x.strip() and "\x00" not in x).all():
            raise ValueError(f"{field} must contain nonempty strings")
    keys = rows.image_path.map(_key)
    if any(key == "." or ".." in PurePosixPath(key).parts for key in keys):
        raise ValueError("image_path must name a file without parent traversal")
    if labels:
        if not rows.Dataset.isin(SOURCES).all():
            raise ValueError("Dataset must contain literal KVASIR, SEE-AI, KID or AIIMS labels")
        values = rows[CLASSES].to_numpy()
        if not np.isin(values, [0, 1]).all() or not (values.sum(axis=1) == 1).all():
            raise ValueError("Each row must have exactly one active one-hot class")
    for field in hashes:
        width = 64 if field == "sha256" else 16
        if not rows[field].map(lambda x: isinstance(x, str) and
                               re.fullmatch(rf"[0-9a-f]{{{width}}}", x) is not None).all():
            raise ValueError(f"{field} must be a lowercase {width}-digit hexadecimal string")
    unique = rows[fields].assign(image_path=keys).drop_duplicates()
    if unique.image_path.duplicated().any():
        raise ValueError("Repeated image_path values have conflicting metadata or hashes")


def measure_images(rows, root=Path("."), *, reference=False):
    """Append SHA-256 and 64-bit pHash/dHash; reference=True needs only image_path."""
    _validate(rows, labels=not reference)
    measurements = []
    for path in rows.image_path:
        data = (Path(root) / _key(path)).read_bytes()
        with Image.open(io.BytesIO(data)) as image:
            rgb = image.convert("RGB")
            measurements.append((hashlib.sha256(data).hexdigest(),
                                 str(imagehash.phash(rgb, hash_size=8)),
                                 str(imagehash.dhash(rgb, hash_size=8))))
    result = rows.copy()
    result[["sha256", "phash", "dhash"]] = np.asarray(measurements, dtype=object).reshape(-1, 3)
    return result


def byte_identity_summary(train, validation, root=Path(".")):
    """Confirm repeated SHA groups byte for byte; count crossings and label sets."""
    groups = defaultdict(list)
    for split, frame in (("training", train), ("validation", validation)):
        _validate(frame, ("sha256",))
        for row in frame.to_dict("records"):
            groups[row["sha256"]].append(dict(row, split=split))
    sources = sorted(set(train.Dataset) | set(validation.Dataset))
    summary = dict(confirmed_sha256_groups=0, crossing_groups=0,
                   cross_source_crossing_groups=0, affected_validation_rows=0,
                   within_source={s: dict(validation_rows=int(validation.Dataset.eq(s).sum()),
                       crossing_groups=0, affected_validation_rows=0,
                       label_intersecting_groups=0, label_disjoint_groups=0) for s in sources})
    for sha, group in groups.items():
        if len(group) < 2:
            continue
        first = (Path(root) / _key(group[0]["image_path"])).read_bytes()
        if hashlib.sha256(first).hexdigest() != sha or any(
                (Path(root) / _key(row["image_path"])).read_bytes() != first for row in group[1:]):
            raise ValueError("SHA-256 group failed byte confirmation; inputs may have changed")
        summary["confirmed_sha256_groups"] += 1
        tr = [r for r in group if r["split"] == "training"]
        va = [r for r in group if r["split"] == "validation"]
        if not tr or not va:
            continue
        summary["crossing_groups"] += 1
        summary["affected_validation_rows"] += len(va)
        summary["cross_source_crossing_groups"] += int(len({r["Dataset"] for r in group}) > 1)
        for source in sources:
            st = [r for r in tr if r["Dataset"] == source]
            sv = [r for r in va if r["Dataset"] == source]
            if st and sv:
                counts = summary["within_source"][source]
                counts["crossing_groups"] += 1
                counts["affected_validation_rows"] += len(sv)
                a = {c for r in st for c in CLASSES if r[c] == 1}
                b = {c for r in sv for c in CLASSES if r[c] == 1}
                counts["label_intersecting_groups"] += int(bool(a & b))
                counts["label_disjoint_groups"] += int(not (a & b))
    return summary


def prefix_overlap(train, validation):
    """Count rows by KVASIR filename recording prefix for Figure 2; reject bad names."""
    counts = []
    for frame in (train, validation):
        _validate(frame)
        prefixes = []
        for path in frame.loc[frame.Dataset.eq("KVASIR"), "image_path"]:
            match = re.fullmatch(r"([0-9a-fA-F]{16})_([0-9]+)", PurePosixPath(_key(path)).stem)
            if match is None:
                raise ValueError(f"Invalid KVASIR recording filename: {path}")
            prefixes.append(match.group(1))
        counts.append(Counter(prefixes))
    tr, va = counts
    return pd.DataFrame([(p, tr[p], va[p]) for p in sorted(tr.keys() | va.keys())],
                        columns=["prefix", "training_rows", "validation_rows"])


def gray(path):
    """Direct 96x96 bicubic grayscale uint8 pixels, without cropping/registration."""
    with Image.open(path) as image:
        return np.asarray(image.convert("L").resize((96, 96), Image.Resampling.BICUBIC), dtype=np.uint8)


def normalized_cross_correlation(a, b):
    """The 96x96 float32 NCC calculation; constant images yield zero."""
    for pixels in (a, b):
        if (not isinstance(pixels, np.ndarray) or pixels.shape != (96, 96)
                or pixels.dtype.kind not in "uif" or not np.isfinite(pixels).all()
                or (pixels < 0).any() or (pixels > 255).any()):
            raise ValueError("NCC expects finite 96x96 grayscale arrays in [0, 255]")
    af, bf = a.astype(np.float32), b.astype(np.float32)
    af -= af.mean()
    bf -= bf.mean()
    denominator = np.sqrt(np.square(af).sum()) * np.sqrt(np.square(bf).sum())
    return float((af * bf).sum() / (denominator + 1e-12))


def internal_rows(train, validation, root=Path("."), ssim=False):
    """Same-source exact-pHash candidates: first lexicographic path versus any NCC>=.99.

    Optional SSIM measures only the selected pair, with uniform 7x7 windows and
    sample covariance. Neither predicate establishes byte identity or recall.
    """
    for frame in (train, validation):
        _validate(frame, ("phash",))
    if ssim:
        from skimage.metrics import structural_similarity
    by_hash = defaultdict(list)
    for row in sorted(train.to_dict("records"), key=lambda r: _key(r["image_path"])):
        by_hash[(row["Dataset"], row["phash"])].append(row["image_path"])
    measurements = []
    for row in validation.to_dict("records"):
        candidates = by_hash[(row["Dataset"], row["phash"])]
        selected_ncc, selected_ssim, passing = np.nan, np.nan, 0
        if candidates:
            pixels = gray(Path(root) / _key(row["image_path"]))
            for i, path in enumerate(candidates):
                other = gray(Path(root) / _key(path))
                ncc = normalized_cross_correlation(other, pixels)
                passing += int(ncc >= .99)
                if i == 0:
                    selected_ncc = ncc
                    if ssim:
                        selected_ssim = float(structural_similarity(other, pixels, data_range=255,
                            win_size=7, gaussian_weights=False, use_sample_covariance=True))
        measurements.append((len(candidates), candidates[0] if candidates else "", selected_ncc,
                             selected_ncc >= .99, passing > 0, passing, selected_ssim))
    columns = ["phash_candidate_count", "selected_train_path", "selected_ncc",
               "selected_ncc99", "any_ncc99", "ncc99_candidate_count", "selected_ssim"]
    result = validation.copy()
    for column in columns if ssim else columns[:-1]:
        result[column] = [m[columns.index(column)] for m in measurements]
    return result


def _distances(query, reference):
    xor = query[:, None] ^ reference[None, :]
    lanes = xor.view(np.uint16).reshape(len(query), len(reference), 4)
    result = POP16[lanes[:, :, 0]]
    for lane in (1, 2, 3):
        result += POP16[lanes[:, :, lane]]
    return result


def reference_minima(queries, references, query_chunk=32, reference_chunk=4096):
    """Exhaustive independent minima, bounded on both axes; no joint-rule implication.

    Queries use the canonical table; references need image_path, phash and dhash.
    screen_positive is true exactly when both independent minima are <=6, for every
    source. References need not share the queries' class taxonomy.
    """
    _validate(queries, ("phash", "dhash"))
    _validate(references, ("phash", "dhash"), labels=False)
    if references.empty or any(isinstance(n, bool) or not isinstance(n, (int, np.integer))
                               or n <= 0 for n in (query_chunk, reference_chunk)):
        raise ValueError("Nonempty references and positive integer chunk sizes are required")
    minima = np.full((len(queries), 2), 65, dtype=np.uint8)
    for h, column in enumerate(("phash", "dhash")):
        q = np.array([int(v, 16) for v in queries[column]], dtype=np.uint64)
        r = np.array([int(v, 16) for v in references[column]], dtype=np.uint64)
        for start in range(0, len(q), query_chunk):
            best = minima[start:start + query_chunk, h]
            for offset in range(0, len(r), reference_chunk):
                distances = _distances(q[start:start + query_chunk], r[offset:offset + reference_chunk])
                np.minimum(best, distances.min(axis=1), out=best)
    result = queries.copy()
    result[["min_phash", "min_dhash"]] = minima
    result["screen_positive"] = (minima <= 6).all(axis=1)
    return result
