"""Selected in-memory calculations for Tables 1 and 2; no file I/O or training.

Rows use unique string row_id, integer labels in CLASSES order, and N-by-10
scores (raw organizer scores) OR probabilities (strict probability endpoints).
source names datasets; group names E/U exposure groups; video_prefix names videos.
Views map names to nonempty ordered row_id subsets; 'full' is added automatically.
Seed mappings preserve insertion order. Intervals condition on fixed lists and
controls; they do not estimate patient-level uncertainty or adjust for selection.
"""
import numpy as np
from scipy.stats import t
from sklearn.metrics import roc_auc_score

CLASSES = (
    'Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body',
    'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms',
)
ARMS = ('local', 'draw101', 'draw202', 'draw303')
ENDPOINTS = ('mean_true_class_prob', 'nll', 'confidence_margin', 'bal_acc', 'acc')
LOCAL_CLASSES = (0, 1, 3, 4, 5, 6)


def _arrays(labels, scores, probability=False):
    y, p = np.asarray(labels), np.asarray(scores)
    if (y.ndim != 1 or not len(y) or y.dtype.kind not in 'iu'
            or not ((y >= 0) & (y < 10)).all()):
        raise ValueError('labels must be a nonempty integer vector in [0, 9]')
    if p.shape != (len(y), 10) or p.dtype.kind not in 'iuf' or not np.isfinite(p).all():
        raise ValueError('scores must be a finite real N-by-10 array')
    p = p.astype(np.float64)
    if not np.isfinite(p).all():
        raise ValueError('scores must be representable in float64')
    if probability and (not ((p >= 0) & (p <= 1)).all()
                        or not np.allclose(p.sum(1), 1, atol=2e-5, rtol=0)):
        raise ValueError('probabilities must lie in [0, 1] and sum to one per row')
    return y.astype(np.int64, copy=False), p


def _strings(values, n, name, unique=False):
    values = np.asarray(values)
    if (values.shape != (n,) or not all(isinstance(v, str) and v.strip() for v in values)
            or (unique and len(set(values)) != n)):
        raise ValueError(f'{name} must contain {n} nonempty strings' +
                         (' without duplicates' if unique else ''))
    return values


def _record(record, field):
    y, p = _arrays(record['labels'], record[field], field == 'probabilities')
    q = dict(record, labels=y, **{field: p})
    if 'classes' in q and list(q['classes']) != list(CLASSES):
        raise ValueError('classes must follow the documented ten-column order')
    q['row_id'] = _strings(q['row_id'], len(y), 'row_id', unique=True)
    for key in ('source', 'group', 'video_prefix'):
        if key in q:
            q[key] = _strings(q[key], len(y), key)
    return q


def _aligned(records, field):
    records = [_record(q, field) for q in records]
    if not records:
        raise ValueError('at least one prediction record is required')
    for key in ('row_id', 'labels', 'source', 'group', 'video_prefix'):
        if any(key in q for q in records):
            if (not all(key in q for q in records)
                    or any(not np.array_equal(q[key], records[0][key]) for q in records)):
                raise ValueError(f'rows must match in identity, order and metadata: {key}')
    return records


def _seed(record, seed):
    value = np.asarray(record['seed'])
    if (value.ndim != 0 or value.dtype.kind not in 'iu' or value.item() < 0
            or isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or value.item() != seed):
        raise ValueError('record seed must be a nonnegative integer matching its mapping key')


def _classification(y, scores, required_classes=None):
    correct = scores.argmax(1) == y
    present = np.unique(y)
    classes = present if required_classes is None else np.asarray(required_classes)
    if (classes.ndim != 1 or not len(classes) or classes.dtype.kind not in 'iu'
            or not ((classes >= 0) & (classes < 10)).all() or len(set(classes)) != len(classes)):
        raise ValueError('required_classes must be distinct integer class indices in [0, 9]')
    per_class = {
        name: {'n': int((y == i).sum()),
               'recall': float(correct[y == i].mean()) if i in present else None}
        for i, name in enumerate(CLASSES)
    }
    ba = (float(np.mean([per_class[CLASSES[i]]['recall'] for i in classes]))
          if set(classes) <= set(present) else None)
    auc = (float(np.mean([roc_auc_score(y == i, scores[:, i]) for i in present]))
           if len(present) > 1 else None)
    return dict(n_test=len(y), bal_acc=ba, acc=float(correct.mean()), mean_auc_ovr=auc,
                combined_ovr=(ba + auc) / 2 if ba is not None and auc is not None else None,
                per_class=per_class, represented_classes=present.tolist(),
                ba_classes=classes.tolist(), auc_classes=present.tolist() if auc is not None else [])


def metrics(labels, probabilities, required_classes=range(10)):
    """Strict probabilities: fixed-ten BA, defined OVR AUC and row-mean endpoints.

    Missing-class recall is None; AUC/combined are None for a single-class view.
    BA defaults to all ten classes; missing required support gives None for
    BA/combined. Explicit required_classes=None uses represented-class BA.
    AUC uses defined OVR classes independently of the BA target.
    NLL is natural-log with floor 1e-12; argmax ties use the first class.
    """
    y, p = _arrays(labels, probabilities, probability=True)
    true = p[np.arange(len(y)), y]
    other = p.copy()
    other[np.arange(len(y)), y] = -np.inf
    return dict(_classification(y, p, required_classes), mean_true_class_prob=float(true.mean()),
                nll=float(-np.log(np.maximum(true, 1e-12)).mean()),
                confidence_margin=float((true - other.max(1)).mean()))


def score_views(record, views=None, required_classes=range(10)):
    """Score raw finite `scores` directly: no softmax, normalization or clipping.

    Optional source enables per-source and equal-source scores. Each source uses
    represented-class BA and defined OVR AUC; if any source has undefined AUC,
    equal-source AUC/combined are None (no silent dropping of sources).
    Pooled BA defaults to all ten classes; missing required support gives None
    for BA/combined. Explicit required_classes=None uses represented-class BA.
    required_classes affects pooled BA only; AUC uses defined OVR classes.
    """
    q = _record(record, 'scores')
    views = {} if views is None else views
    if 'full' in views:
        raise ValueError("'full' is reserved for the complete supplied record")
    positions = {row: i for i, row in enumerate(q['row_id'])}
    output = {}
    for name, ids in {'full': q['row_id'], **views}.items():
        ids = _strings(ids, len(ids), 'view row_id', unique=True)
        if not len(ids) or not set(ids) <= positions.keys():
            raise ValueError('views must be nonempty subsets of known row identities')
        index = np.array([positions[row] for row in ids])
        y, p = q['labels'][index], q['scores'][index]
        report = {'pooled': _classification(y, p, required_classes)}
        if 'source' in q:
            source = q['source'][index]
            report['per_source'] = {
                s: _classification(y[source == s], p[source == s]) for s in np.unique(source)
            }
            report['equal_source'] = {
                key: (float(np.mean(values)) if all(v is not None for v in values) else None)
                for key in ('bal_acc', 'mean_auc_ovr', 'combined_ovr')
                for values in [[m[key] for m in report['per_source'].values()]]
            }
        output[name] = report
    return output


def fixed_prediction_rescore(entries, views, required_classes=range(10)):
    """Table 1d: caller-selected cohort {entry: raw-score record}, fixed frames.

    Every entry requires source. Returns entry scores and descriptive mean changes
    from each metric's own full-view baseline; no inference over related entries.
    Call separately for the primary 24 and sensitivity 19 cohorts.
    Pooled BA defaults to all ten classes; missing support gives None for
    BA/combined and their changes. Explicit required_classes=None uses
    represented-class BA. Source BA and defined OVR AUC follow score_views.
    """
    records = _aligned(entries.values(), 'scores')
    if any('source' not in q for q in records):
        raise ValueError('source is required for equal-source re-scoring')
    scored = {name: score_views(q, views, required_classes) for name, q in zip(entries, records)}
    changes = {}
    for view in views:
        changes[view] = {}
        for weighting in ('pooled', 'equal_source'):
            changes[view][weighting] = {}
            for metric in ('bal_acc', 'mean_auc_ovr', 'combined_ovr'):
                pairs = [(s[view][weighting][metric], s['full'][weighting][metric])
                         for s in scored.values()]
                changes[view][weighting][metric] = (
                    float(np.mean([a - b for a, b in pairs]))
                    if all(a is not None and b is not None for a, b in pairs) else None)
    return {'n_entries': len(scored), 'entries': scored, 'mean_changes': changes}


def paired_intervals(values, bootstrap_indices=None):
    """One difference per paired training seed; default 10,000 draws, RNG seed 42.

    Optional B-by-n integer indices allow evaluation of supplied seed resamples.
    This low-level function cannot verify pairing; compare/local_contrast do so.
    """
    values = np.asarray(values)
    if values.ndim != 1 or len(values) < 2 or values.dtype.kind not in 'iuf':
        raise ValueError('at least two real paired seed differences are required')
    values = values.astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError('paired seed differences must be finite')
    n = len(values)
    indices = (np.random.default_rng(42).integers(0, n, size=(10000, n))
               if bootstrap_indices is None else np.asarray(bootstrap_indices))
    if (indices.ndim != 2 or not len(indices) or indices.shape[1] != n
            or indices.dtype.kind not in 'iu' or not ((indices >= 0) & (indices < n)).all()):
        raise ValueError('bootstrap indices must select complete valid seed blocks')
    mean = float(values.mean())
    half = float(t.ppf(.975, n - 1) * values.std(ddof=1) / np.sqrt(n))
    return dict(mean=mean, values=values.tolist(), paired_t_95_ci=[mean - half, mean + half],
                paired_bootstrap_95_ci=np.quantile(values[indices].mean(1), [.025, .975]).tolist(),
                signs=[int((values < -1e-12).sum()), int((np.abs(values) <= 1e-12).sum()),
                       int((values > 1e-12).sum())])


def compare(left_by_seed, right_by_seed, views=None, required_classes=range(10)):
    """Paired left-minus-right pooled BA on fixed views (raw-score records).

    Seed keys and each record's scalar seed must agree. Row identities, labels
    and supplied source/group/video metadata must match across all seed blocks.
    BA defaults to all ten classes; missing required support gives None for that
    view. Explicit required_classes=None uses represented-class BA.
    """
    seeds = list(left_by_seed)
    if len(seeds) < 2 or set(seeds) != set(right_by_seed):
        raise ValueError('at least two identical training-seed sets are required')
    records = []
    for side in (left_by_seed, right_by_seed):
        for seed in seeds:
            _seed(side[seed], seed)
            records.append(side[seed])
    scored = [score_views(q, views, required_classes) for q in _aligned(records, 'scores')]
    output = {}
    for view in scored[0]:
        if any(s[view]['pooled']['bal_acc'] is None for s in scored):
            output[view] = None
            continue
        left, right = np.array([s[view]['pooled']['bal_acc'] for s in scored]).reshape(2, -1)
        output[view] = dict(paired_intervals(left - right), seeds=seeds,
                            left_mean=float(left.mean()), right_mean=float(right.mean()))
    return output


def local_contrast(by_seed_arm, required_classes=LOCAL_CLASSES):
    """Eq. 3 / Tables 1e, 2: {(seed, arm): strict probability record}.

    Records require scalar seed/arm, group ('E' or 'U'), and video_prefix.
    Both groups must have identical video/class counts. Four arms are fixed in
    ARMS order; controls are averaged within each seed before any seed inference.
    Returns absolute levels, E and U differences, and E-minus-U seed intervals
    in fractions (natural-log NLL), with signs using zero tolerance 1e-12.
    BA defaults to the paper's fixed six classes; absent required support gives
    endpoints['bal_acc'] = None. Other endpoints remain defined.
    """
    seeds = list(dict.fromkeys(seed for seed, arm in by_seed_arm))
    keys = [(s, arm) for s in seeds for arm in ARMS]
    if len(seeds) < 2 or set(by_seed_arm) != set(keys):
        raise ValueError('at least two complete four-arm training-seed blocks are required')
    for seed, arm in keys:
        _seed(by_seed_arm[seed, arm], seed)
        if by_seed_arm[seed, arm]['arm'] != arm:
            raise ValueError('record arm must match its mapping key')
    records = _aligned([by_seed_arm[key] for key in keys], 'probabilities')
    q = records[0]
    if not {'group', 'video_prefix'} <= q.keys() or set(q['group']) != {'E', 'U'}:
        raise ValueError('group E/U and video_prefix are required')
    e, u = q['group'] == 'E', q['group'] == 'U'
    for prefix in np.unique(q['video_prefix']):
        mask = q['video_prefix'] == prefix
        if not np.array_equal(np.bincount(q['labels'][mask & e], minlength=10),
                              np.bincount(q['labels'][mask & u], minlength=10)):
            raise ValueError('E and U must match counts within every video/class stratum')
    cells = [[metrics(r['labels'][mask], r['probabilities'][mask], required_classes) for mask in (e, u)]
             for r in records]
    output = {}
    for endpoint in ENDPOINTS:
        if any(c[endpoint] is None for cell in cells for c in cell):
            output[endpoint] = None
            continue
        values = np.array([[c[endpoint] for c in cell] for cell in cells]).reshape(-1, 4, 2)
        local, distant = values[:, 0], values[:, 1:].mean(1)
        delta = local - distant
        output[endpoint] = {
            'absolute': {'E_local': float(local[:, 0].mean()), 'E_distant': float(distant[:, 0].mean()),
                         'U_local': float(local[:, 1].mean()), 'U_distant': float(distant[:, 1].mean())},
            'E': paired_intervals(delta[:, 0]), 'U': paired_intervals(delta[:, 1]),
            'E_minus_U': paired_intervals(delta[:, 0] - delta[:, 1]),
        }
    return {'seeds': seeds, 'n_per_group': int(e.sum()),
            'represented_classes': np.unique(q['labels']).tolist(),
            'ba_classes': cells[0][0]['ba_classes'], 'endpoints': output}


def official_aggregates(confusions, class_auc):
    """Table 1c: {seed: 10x10 true-by-predicted counts}, {seed: ten AUCs}.

    Requires all ten classes and identical class counts across seeds. Returns
    per-seed values, means and sample standard errors, not confidence intervals.
    """
    seeds = list(confusions)
    if len(seeds) < 2 or set(seeds) != set(class_auc):
        raise ValueError('confusions and AUCs need identical sets of at least two seeds')
    for seed in seeds:
        _seed({'seed': seed}, seed)
    counts, auc = np.asarray([confusions[s] for s in seeds]), np.asarray([class_auc[s] for s in seeds])
    if (counts.shape != (len(seeds), 10, 10) or counts.dtype.kind not in 'iuf'
            or not np.isfinite(counts).all() or (counts < 0).any() or (counts != np.floor(counts)).any()):
        raise ValueError('confusion matrices must contain nonnegative integer counts')
    counts = counts.astype(np.float64)
    support = counts.sum(2)
    if (support == 0).any() or not (support == support[0]).all():
        raise ValueError('all ten classes need identical nonzero support across seeds')
    if (auc.shape != (len(seeds), 10) or auc.dtype.kind not in 'iuf'
            or not np.isfinite(auc).all() or not ((auc >= 0) & (auc <= 1)).all()):
        raise ValueError('ten finite class AUCs in [0, 1] are required per seed')
    diagonal = np.diagonal(counts, axis1=1, axis2=2)
    ba, mean_auc = (diagonal / support).mean(1), auc.astype(np.float64).mean(1)
    values = dict(bal_acc=ba, mean_auc_ovr=mean_auc, combined_ovr=(ba + mean_auc) / 2,
                  acc=diagonal.sum(1) / support.sum(1))
    return {'seeds': seeds, **{key: dict(values=v.tolist(), mean=float(v.mean()),
                                        se=float(v.std(ddof=1) / np.sqrt(len(seeds))))
                              for key, v in values.items()}}
