#!/usr/bin/env python3
"""Evaluate user-supplied CV2024 predictions; write metrics to standard output."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import f1_score

from evaluation import (
    ARMS, CLASSES, STRATA, cluster_bootstrap_intervals, metrics,
    paired_intervals, validate_predictions,
)


def read_predictions(path):
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
    elif path.suffix == '.csv':
        with path.open(newline='', encoding='utf-8') as stream:
            reader = csv.DictReader(stream)
            header = reader.fieldnames
            required = ['true_label_index', *('p_' + c for c in CLASSES)]
            if header is None or len(set(header)) != len(header) or not set(required) <= set(header):
                raise ValueError('CSV needs unique headers, true_label_index and the ten p_CLASS columns')
            rows = list(reader)
        if any(None in row or any(value is None for value in row.values()) for row in rows):
            raise ValueError('CSV rows must have the same number of fields as the header')
        arrays = {
            'true_label_index': np.array([int(row['true_label_index']) for row in rows]),
            'probabilities': np.array([[float(row['p_' + c]) for c in CLASSES] for row in rows]),
        }
        for key in ['dataset_index', 'predictions']:
            if key in header:
                arrays[key] = np.array([int(row[key]) for row in rows])
        if 'stratum' in header:
            arrays['stratum'] = np.array([row['stratum'] for row in rows])
    elif path.suffix == '.pt':
        import torch
        data = torch.load(path, map_location='cpu', weights_only=True)
        arrays = {
            'true_label_index': np.asarray([label for _, label in data['items']]),
            'probabilities': data['probabilities'].numpy(),
            'classes': np.asarray(data['classes']),
            'predictions': data['predictions'].numpy(),
            'logits': data['logits'].float().numpy(),
        }
    else:
        raise ValueError('Use CSV, NPZ or a tensor export from training/crossmodel.py')
    validate_export(arrays)
    return arrays


def validate_export(arrays):
    y, p = validate_predictions(arrays['true_label_index'], arrays['probabilities'])
    if 'classes' in arrays and not np.array_equal(arrays['classes'], CLASSES):
        raise ValueError('Class order differs from CV2024')
    if 'dataset_index' in arrays and not np.array_equal(arrays['dataset_index'], np.arange(len(y))):
        raise ValueError('Dataset indices must preserve physical row order')
    if 'predictions' in arrays and not np.array_equal(arrays['predictions'], p.argmax(1)):
        raise ValueError('Predictions disagree with probability argmax')
    if 'stratum' in arrays:
        s = arrays['stratum']
        if s.shape != y.shape or s.dtype.kind != 'U' or not all(value.strip() for value in s):
            raise ValueError('Sources must be nonempty strings aligned with prediction rows')
    if 'logits' in arrays:
        logits = arrays['logits']
        if logits.shape != p.shape or not np.isfinite(logits).all():
            raise ValueError('Invalid logits')
        exps = np.exp(logits.astype(np.float64) - logits.max(1, keepdims=True))
        if not np.allclose(p, exps / exps.sum(1, keepdims=True), atol=1e-7, rtol=1e-5):
            raise ValueError('Probabilities disagree with logit softmax')


def evaluate(arrays, exclude_sources=(), f1_class_set='present'):
    validate_export(arrays)
    if f1_class_set not in ('present', 'union'):
        raise ValueError('F1 class set must be present or union')
    y, p = arrays['true_label_index'], arrays['probabilities']
    selected = np.ones(len(y), dtype=bool)
    if exclude_sources:
        if 'stratum' not in arrays or not set(exclude_sources) <= set(arrays['stratum']):
            raise ValueError('Every excluded source must exist in the supplied stratum field')
        selected = ~np.isin(arrays['stratum'], exclude_sources)
    if not selected.any():
        raise ValueError('The requested evaluation population is empty')

    def summarize(mask):
        result = metrics(y[mask], p[mask])
        if f1_class_set == 'union':
            pred = p[mask].argmax(1)
            result['f1_macro'] = float(f1_score(y[mask], pred, average='macro', zero_division=0))
            result['f1_macro_class_set'] = [CLASSES[i] for i in np.union1d(y[mask], pred)]
        return result

    report = {'pooled': summarize(selected), 'excluded_sources': list(exclude_sources)}
    if 'stratum' in arrays:
        per_source = {source: summarize(selected & (arrays['stratum'] == source))
                      for source in sorted(set(arrays['stratum'][selected]))}
        report['per_source'] = per_source
        combined = [m['combined_ovr'] for m in per_source.values()]
        report['source_balanced'] = {
            'bal_acc': float(np.mean([m['bal_acc'] for m in per_source.values()])),
            'combined_ovr': float(np.mean(combined)) if all(v is not None for v in combined) else None,
        }
    return report


def self_check():
    from tempfile import TemporaryDirectory

    p = np.zeros((4, 10))
    p[:, :2] = [[.8, .2], [.5, .5], [.2, .8], [.6, .4]]
    arrays = {'true_label_index': np.array([0, 0, 1, 1]), 'probabilities': p}
    result = evaluate(arrays)['pooled']
    assert result['bal_acc'] == result['mean_auc_ovr'] == .75 and result['argmax_ties'] == 1
    half = np.zeros_like(p, dtype=np.float16)
    half[:, 1] = 1
    half_arrays = {'true_label_index': arrays['true_label_index'].astype(np.uint64),
                   'probabilities': half}
    half_result = evaluate(half_arrays)['pooled']
    assert half_result == metrics(arrays['true_label_index'], half.astype(np.float32))
    assert np.isfinite(half_result['nll'])
    json.dumps(half_result, allow_nan=False)
    for key, value in [
        ('probabilities', p * 2), ('probabilities', np.full_like(p, np.nan)),
        ('true_label_index', np.array([-1, 0, 1, 1])),
        ('true_label_index', np.array([0., 0., 1., 1.])),
        ('dataset_index', np.array([1, 0, 2, 3])),
        ('predictions', np.array([1, 0, 1, 0])),
        ('classes', list(reversed(CLASSES))),
        ('stratum', np.array(['a', 'a', ' ', 'b'])),
    ]:
        try:
            evaluate({**arrays, key: value})
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid prediction export accepted: ' + key)
    sources = {**arrays, 'stratum': np.array(['a', 'b', 'a', 'b'])}
    assert evaluate(sources, ['a'])['pooled']['n_test'] == 2
    for excluded in [['unknown'], ['a', 'b']]:
        try:
            evaluate(sources, excluded)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid population selection accepted')
    paired = paired_intervals(np.ones(2), np.array([[0, 1], [1, 0]]))
    assert paired['paired_t_95_ci'] == paired['paired_bootstrap_95_ci'] == [1., 1.]
    block = {
        (0, arm): {'predictions': {
            **arrays, 'stratum': np.array(STRATA * 2),
            'video_prefix': np.array(['v1', 'v1', 'v2', 'v2']),
        }} for arm in ARMS
    }
    intervals = cluster_bootstrap_intervals(block, [0], resamples=32, bootstrap_seed=0)
    assert all(np.allclose(value, 0, atol=1e-15, rtol=0)
               for key, value in intervals.items() if key != 'metadata')
    half_block = {k: {'predictions': {**half_arrays,
                                    'stratum': v['predictions']['stratum'],
                                    'video_prefix': v['predictions']['video_prefix']}}
                  for k, v in block.items()}
    half_intervals = cluster_bootstrap_intervals(half_block, [0], resamples=32, bootstrap_seed=0)
    assert all(np.allclose(value, 0, atol=1e-14, rtol=0)
               for key, value in half_intervals.items() if key != 'metadata')
    json.dumps(half_intervals, allow_nan=False)
    for key, value in [('video_prefix', np.array([' ', ' ', 'v2', 'v2'])),
                       ('correct', np.array(1.))]:
        invalid = {k: {'predictions': {**v['predictions'], key: value}} for k, v in block.items()}
        try:
            cluster_bootstrap_intervals(invalid, [0], resamples=32, bootstrap_seed=0)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid bootstrap input accepted: ' + key)
    with TemporaryDirectory() as directory:
        path = Path(directory) / 'predictions.csv'
        header = ['true_label_index', *('p_' + c for c in CLASSES)]
        row = [0, *p[0]]
        for fields in [row, row[:-1], row + ['extra']]:
            with path.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.writer(stream)
                writer.writerows([header, fields])
            if fields == row:
                assert len(read_predictions(path)['true_label_index']) == 1
                continue
            try:
                read_predictions(path)
            except ValueError:
                pass
            else:
                raise AssertionError('Malformed CSV row accepted')
    print('PASS: metrics, input validation, row/class order, source selection and paired intervals')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', type=Path, nargs='?')
    parser.add_argument('--exclude-source', action='append', default=[])
    parser.add_argument('--f1-class-set', choices=['present', 'union'], default='present',
                        help='Use union for confidence-training macro-F1; present for primary/CNN metrics')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if sys.flags.optimize:
        parser.error('Run without -O: numerical checks require assertions')
    if args.self_check:
        self_check()
        return
    if args.predictions is None:
        parser.error('Supply a prediction file')
    print(json.dumps(evaluate(read_predictions(args.predictions), args.exclude_source,
                              args.f1_class_set), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
