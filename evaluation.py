"""CV2024 metrics and paired seed/video-prefix uncertainty calculations."""
from __future__ import annotations
import math
from typing import Any, Sequence

import numpy as np
from scipy.stats import t
from sklearn.metrics import f1_score, roc_auc_score

CLASSES = [
    'Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body',
    'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms',
]
EXPOSED = 'NCC99_EXPOSED'
MATCHED = 'MATCHED_UNFLAGGED'
STRATA = (EXPOSED, MATCHED)
ARMS = ('local', 'draw101', 'draw202', 'draw303')
ENDPOINT_COLUMNS = {
    'nll': 'nll',
    'true_class_probability': 'p_true',
    'confidence_margin': 'confidence_margin',
    'balanced_accuracy': 'correct',
    'accuracy': 'correct',
}


def validate_predictions(labels, probabilities):
    """Validate the ten-class probability interface without loading any dataset."""
    y, p = np.asarray(labels), np.asarray(probabilities)
    if y.ndim != 1 or not len(y) or y.dtype.kind not in 'iu':
        raise ValueError('Labels must be a nonempty integer vector')
    if not ((y >= 0) & (y < len(CLASSES))).all():
        raise ValueError('Labels must index the documented ten-class order')
    if p.shape != (len(y), len(CLASSES)) or p.dtype.kind != 'f':
        raise ValueError('Probabilities must be a floating N-by-10 array')
    if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError('Probabilities must be finite and in [0, 1]')
    if not np.allclose(p.sum(1), 1, atol=2e-5, rtol=0):
        raise ValueError('Probability rows must sum to one')
    # Keep the probability floor representable and support every integer label dtype.
    return y.astype(np.int64, copy=False), p.astype(np.result_type(p.dtype, np.float32), copy=False)


def metrics(labels, probabilities):
    """Score ordered predictions, averaging class metrics over supported classes."""
    labels, p = validate_predictions(labels, probabilities)
    pred = p.argmax(1)
    correct = pred == labels
    present = np.unique(labels)
    pt = p[np.arange(len(labels)), labels]
    other = p.copy()
    other[np.arange(len(labels)), labels] = -np.inf
    recalls = {
        name: {
            'n': int((labels == i).sum()),
            'recall': float(correct[labels == i].mean()) if (labels == i).any() else None,
        }
        for i, name in enumerate(CLASSES)
    }
    auc = ([float(roc_auc_score(labels == i, p[:, i])) for i in present]
           if len(present) > 1 else [])
    ba = float(np.mean([v['recall'] for v in recalls.values() if v['n']]))
    ma = float(np.mean(auc)) if auc else None
    return dict(
        n_test=len(labels),
        bal_acc=ba,
        acc=float(correct.mean()),
        f1_macro=float(f1_score(labels, pred, labels=present, average='macro', zero_division=0)),
        mean_auc_ovr=ma,
        combined_ovr=(ba + ma) / 2 if ma is not None else None,
        mean_true_class_prob=float(pt.mean()),
        nll=float(-np.log(np.clip(pt, 1e-12, 1)).mean()),
        confidence_margin=float((pt - other.max(1)).mean()),
        null_acc=float(max(np.bincount(labels)) / len(labels)),
        per_class=recalls,
        f1_macro_class_set=[CLASSES[i] for i in present],
        mean_auc_ovr_mode='all_classes' if len(present) == 10 else 'present_classes',
        mean_auc_ovr_classes=[CLASSES[i] for i in present] if auc else [],
        argmax_ties=int(((p == p.max(1)[:, None]).sum(1) > 1).sum()),
        max_probability_sum_error=float(np.max(np.abs(p.sum(1) - 1))),
    )


def paired_intervals(
    values: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> dict[str, Any]:
    """Return paired t and percentile-bootstrap intervals for seed differences.

    ``values`` contains one arm-minus-control difference per matched seed.
    ``bootstrap_indices`` is a B-by-n integer matrix of resampled seed positions.
    """
    values, bootstrap_indices = np.asarray(values), np.asarray(bootstrap_indices)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("paired intervals require a one-dimensional n>=2 vector")
    if not np.isfinite(values).all():
        raise ValueError('Paired differences must be finite')
    if (bootstrap_indices.ndim != 2 or not len(bootstrap_indices)
            or bootstrap_indices.shape[1] != len(values)
            or bootstrap_indices.dtype.kind not in 'iu'
            or not ((bootstrap_indices >= 0) & (bootstrap_indices < len(values))).all()):
        raise ValueError('Bootstrap indices must select complete valid seed blocks')
    mean = float(values.mean())
    sem = float(values.std(ddof=1) / math.sqrt(len(values)))
    half_width = float(t.ppf(0.975, len(values) - 1)) * sem
    bootstrap_means = values[bootstrap_indices].mean(axis=1)
    return {
        "mean": mean,
        "values": values.tolist(),
        "paired_t_95_ci": [mean - half_width, mean + half_width],
        "paired_bootstrap_95_ci": np.quantile(
            bootstrap_means,
            [0.025, 0.975],
        ).tolist(),
    }

def cluster_bootstrap_intervals(
    by_key: dict[tuple[int, str], dict[str, Any]],
    seed_order: Sequence[int],
    *,
    resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Bootstrap the local-versus-distant difference between exposure groups.

    ``by_key[(seed, arm)]['predictions']`` contains NumPy arrays named
    ``true_label_index``, ``probabilities``, ``stratum`` and ``video_prefix``.
    Every seed has all four ``ARMS``: local and the three distant control draws
    identified by their sampling seeds (101, 202, 303). The two ``STRATA`` are
    NCC >= 0.99 exposed frames and matched unflagged frames. Supply identical
    physical row order in every arm; label/group checks alone cannot establish
    frame identity. Every prefix must contain both exposure groups.

    The function derives per-frame endpoints in each predictions dictionary.
    It resamples prefixes, preserves paired arms, averages across training seeds,
    and returns 95% percentile intervals for (local - control) in the exposed
    group minus (local - control) in the matched group. Video prefixes are the
    sampling units, not verified patient or examination identifiers.
    """
    if len(seed_order) == 0 or len(set(seed_order)) != len(seed_order) or resamples < 1:
        raise ValueError('Unique nonempty seeds and a positive resample count required')
    if set(by_key) != {(seed, arm) for seed in seed_order for arm in ARMS}:
        raise ValueError('One complete four-arm block is required per seed')
    reference = by_key[(seed_order[0], "local")]["predictions"]
    for record in by_key.values():
        q = record['predictions']
        y, p = validate_predictions(q['true_label_index'], q['probabilities'])
        for key in ['true_label_index', 'stratum', 'video_prefix']:
            if q[key].shape != y.shape or not np.array_equal(q[key], reference[key]):
                raise ValueError('Seed/arm rows, labels and groups must align')
        if set(q['stratum']) != set(STRATA):
            raise ValueError('Both exposed and matched groups are required')
        if (q['video_prefix'].dtype.kind != 'U'
                or not all(value.strip() for value in q['video_prefix'])):
            raise ValueError('Video prefixes must be nonempty strings')
        true = p[np.arange(len(y)), y]
        other = p.copy(); other[np.arange(len(y)), y] = -np.inf
        derived = dict(correct=(p.argmax(1) == y).astype(float), p_true=true,
                       nll=-np.log(np.maximum(true, 1e-12)),
                       confidence_margin=true - other.max(1))
        for key, value in derived.items():
            if key in q and (np.shape(q[key]) != value.shape
                             or not np.allclose(q[key], value, atol=1e-10, rtol=0)):
                raise ValueError('Stored derived endpoint disagrees with probabilities')
            q[key] = value
    for prefix in set(reference['video_prefix']):
        for stratum in STRATA:
            if not ((reference['video_prefix'] == prefix) & (reference['stratum'] == stratum)).any():
                raise ValueError('Every resampled prefix must support both groups')
    cluster_names = sorted(set(reference["video_prefix"].tolist()))
    cluster_to_index = {
        cluster: index for index, cluster in enumerate(cluster_names)
    }
    cluster_index = np.asarray(
        [cluster_to_index[value] for value in reference["video_prefix"]],
        dtype=np.int64,
    )
    n_clusters = len(cluster_names)
    rng = np.random.default_rng(bootstrap_seed)
    cluster_weights = rng.multinomial(
        n_clusters,
        np.full(n_clusters, 1.0 / n_clusters),
        size=resamples,
    ).astype(np.float64)

    output: dict[str, Any] = {}
    for endpoint in ENDPOINT_COLUMNS:
        values = np.empty(
            (resamples, len(seed_order), len(ARMS), len(STRATA)),
            dtype=np.float64,
        )
        for seed_index, seed in enumerate(seed_order):
            for arm_index, arm in enumerate(ARMS):
                predictions = by_key[(seed, arm)]["predictions"]
                for stratum_index, stratum in enumerate(STRATA):
                    stratum_mask = predictions["stratum"] == stratum
                    if endpoint == "balanced_accuracy":
                        recalls = []
                        labels = sorted(
                            set(
                                predictions["true_label_index"][
                                    stratum_mask
                                ].tolist()
                            )
                        )
                        for label in labels:
                            mask = stratum_mask & (
                                predictions["true_label_index"] == label
                            )
                            counts = np.bincount(
                                cluster_index[mask],
                                minlength=n_clusters,
                            ).astype(np.float64)
                            correct = np.bincount(
                                cluster_index[mask],
                                weights=predictions["correct"][mask],
                                minlength=n_clusters,
                            )
                            denominator = cluster_weights @ counts
                            numerator = cluster_weights @ correct
                            recalls.append(
                                np.divide(
                                    numerator,
                                    denominator,
                                    out=np.full_like(numerator, np.nan),
                                    where=denominator > 0,
                                )
                            )
                        cell_values = np.nanmean(
                            np.column_stack(recalls),
                            axis=1,
                        )
                    else:
                        column = ENDPOINT_COLUMNS[endpoint]
                        counts = np.bincount(
                            cluster_index[stratum_mask],
                            minlength=n_clusters,
                        ).astype(np.float64)
                        sums = np.bincount(
                            cluster_index[stratum_mask],
                            weights=predictions[column][stratum_mask],
                            minlength=n_clusters,
                        )
                        denominator = cluster_weights @ counts
                        cell_values = (cluster_weights @ sums) / denominator
                    values[:, seed_index, arm_index, stratum_index] = (
                        cell_values
                    )
        local = values[:, :, 0, :]
        control = values[:, :, 1:, :].mean(axis=2)
        per_seed_did = (
            (local[:, :, 0] - control[:, :, 0])
            - (local[:, :, 1] - control[:, :, 1])
        )
        mean_did = per_seed_did.mean(axis=1)
        output[endpoint] = np.quantile(
            mean_did,
            [0.025, 0.975],
        ).tolist()
    output["metadata"] = {
        "cluster_unit": "KVASIR video prefix",
        "n_clusters": n_clusters,
        "resamples": resamples,
        "seed": bootstrap_seed,
        "seed_effects": "averaged within every cluster resample",
        "balanced_accuracy_support": "mean recall over classes present in each resample",
    }
    return output
