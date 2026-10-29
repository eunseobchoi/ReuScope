#!/usr/bin/env python3
"""Recompute all 450 primary/CNN seed-view records from retained predictions.

Usage: python recompute.py --output recomputed.json
Reads data/predictions and data/results by default; other copies may be given as
folders or zips (the predictions folder's parent). Requires the public analysis.py
and its existing CPU dependencies; no torch, images, weights or training.
"""
import argparse
from collections import defaultdict
from contextlib import ExitStack
import gzip
import io
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
import pandas as pd
import analysis


def check_equal(actual, expected, path=''):
    """Check every published metric field, including supports and class recalls."""
    if isinstance(expected, dict):
        for key, value in expected.items():
            check_equal(actual[key], value, path + '/' + key)
    elif expected is None:
        if actual is not None:
            raise ValueError(f'{path}: expected undefined metric')
    elif not np.allclose(actual, expected, atol=1e-10, rtol=0):
        raise ValueError(f'{path}: metric mismatch: {actual!r} != {expected!r}')


def reader(path, stack):
    if path.is_dir():
        return lambda name: (path / name).read_bytes()
    return stack.enter_context(ZipFile(path)).read


def recompute(read_predictions, read_results):
    load = lambda name: json.loads(gzip.decompress(read_results('numerical/' + name)))
    expected = load('seed_metrics.json.gz')
    keys = ('family', 'backbone', 'arm', 'seed', 'view')
    references = {tuple(r[k] for k in keys): r['metrics'] for r in expected}
    assert len(expected) == len(references) == 450
    validation = pd.read_csv(io.BytesIO(read_results('inputs/validation.csv.gz')), compression='gzip')
    membership = pd.read_csv(io.BytesIO(read_results('inputs/evaluation_memberships.csv.gz')), compression='gzip')
    assert len(validation) == 16132 and validation.row_id.is_unique
    assert validation[list(analysis.CLASSES)].isin([0, 1]).all().all()
    assert validation[list(analysis.CLASSES)].sum(axis=1).eq(1).all()
    views = {'public': validation.row_id.to_numpy(dtype=str)}
    for name, rows in membership.groupby('view', sort=False):
        assert rows.position.tolist() == list(range(len(rows)))
        assert rows.row_id.is_unique and rows.row_id.isin(validation.row_id).all()
        views[name] = rows.row_id.to_numpy(dtype=str)
    assert {k: len(v) for k, v in views.items()} == {
        'public': 16132, 'non_kvasir': 4551, 'filtered': 4326}
    metadata = validation.set_index('row_id', verify_integrity=True)
    manifest = json.loads(read_predictions('predictions/manifest.json'))
    assert len(manifest) == 230 and len({r['file'] for r in manifest}) == 230
    records, seen = [], set()
    names = {'public': 'Public', 'non_kvasir': 'Non-KV', 'filtered': 'Filtered'}
    for entry in manifest:
        with np.load(io.BytesIO(read_predictions(entry['file'])), allow_pickle=False) as data:
            z = dict(data)
        assert set(z) == {'probabilities', 'labels', 'row_id', 'source', 'classes', 'seed', 'arm'}
        assert list(z['classes']) == list(analysis.CLASSES)
        assert z['seed'].shape == () and z['seed'].item() == entry['seed']
        assert z['arm'].shape == () and z['arm'].item() == entry['arm']
        assert np.array_equal(z['row_id'], views[entry['view']])
        assert len(z['row_id']) == entry['rows']
        assert str(z['probabilities'].dtype) == entry['dtype']
        assert entry['dtype'] == ('float64' if entry['family'] == 'primary' else 'float32')
        aligned = metadata.loc[z['row_id']]
        assert np.array_equal(z['labels'], aligned[list(analysis.CLASSES)].to_numpy().argmax(1))
        assert np.array_equal(z['source'], aligned.Dataset.to_numpy())
        if entry['family'] == 'primary':
            assert entry['view'] == 'public'
            subsets = views
        else:
            assert entry['family'] == 'cnn'
            subsets = {entry['view']: views[entry['view']]}
        positions = {row: i for i, row in enumerate(z['row_id'])}
        for view, ids in subsets.items():
            ix = np.array([positions[row] for row in ids])
            # One metric evaluation per seed/view; no redundant per-class calls.
            metrics = analysis.metrics(z['labels'][ix], z['probabilities'][ix])
            record = {k: entry[k] for k in keys[:-1]}
            record.update(view=names[view], metrics=metrics)
            key = tuple(record[k] for k in keys)
            assert key not in seen
            check_equal(metrics, references[key], '/'.join(map(str, key)))
            seen.add(key)
            records.append(record)
    assert seen == references.keys()
    cells = defaultdict(dict)
    for r in records:
        cells[r['family'], r['backbone'], r['arm'], r['view']][r['seed']] = r['metrics']
    assert len(cells) == 45 and all(set(v) == set(range(10)) for v in cells.values())
    summary = {'/'.join(key): {
        metric: float(np.mean([values[s][metric] for s in range(10)]))
        for metric in ('bal_acc', 'acc', 'mean_auc_ovr', 'combined_ovr')
    } for key, values in cells.items()}
    # Same contrast targets and intervals as replay.py.
    reference = load('reference_checks.json.gz')
    targets = {('primary', 'dinov2_vitl14', name, 'Public')
               for name in reference['primary_contrasts']}
    targets.update(('primary', 'dinov2_vitl14', name, view)
                   for name in ('Removal minus Full', 'Source-only minus Random', 'A minus Comp')
                   for view in ('Public', 'Non-KV'))
    targets.add(('primary', 'dinov2_vitl14', 'Removal minus Full', 'Filtered'))
    targets.update(('cnn', backbone, 'Source-only minus Random', view)
                   for backbone in ('resnet50', 'convnext_tiny')
                   for view in ('Public', 'Non-KV'))
    contrasts = {}
    for family, backbone, name, view in sorted(targets):
        left, right = name.split(' minus ')
        values = [cells[family, backbone, left, view][s]['bal_acc']
                  - cells[family, backbone, right, view][s]['bal_acc'] for s in range(10)]
        contrasts['/'.join((family, backbone, name, view))] = analysis.paired_intervals(values)
    for name, saved in reference['primary_contrasts'].items():
        actual = contrasts['primary/dinov2_vitl14/' + name + '/Public']
        for field in ('values', 'mean', 'paired_t_95_ci', 'paired_bootstrap_95_ci'):
            check_equal(actual[field], saved[field], name + '/' + field)
    for r in reference['cnn_contrasts']:
        if r['metric'] != 'bal_acc' or r['contrast'] != 'Source-only minus Random':
            continue
        view = {'orig_val': 'Public', 'dedup_val': 'Non-KV'}[r['pool']]
        actual = contrasts['/'.join(('cnn', r['backbone'], r['contrast'], view))]
        check_equal(actual['mean'], r['expected']['mean'])
        check_equal(actual['paired_t_95_ci'], r['expected']['ci95'])
    return dict(status='passed', coverage=dict(primary_fits=110, primary_seed_views=330,
                cnn_fits=60, cnn_seed_views=120, prediction_files=230),
                metric_tolerance=dict(atol=1e-10, rtol=0), seed_metrics=records,
                summary_means=summary, paired_ba_contrasts=contrasts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    data = Path(__file__).resolve().parent / 'data'
    parser.add_argument('predictions', type=Path, nargs='?', default=data)
    parser.add_argument('results', type=Path, nargs='?', default=data / 'results')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if sys.flags.optimize:
        parser.error('Run without -O; validation assertions are required')
    if args.output and (args.output.exists() or not args.output.parent.is_dir()):
        parser.error('--output must be a new file in an existing folder')
    with ExitStack() as stack:
        result = recompute(reader(args.predictions, stack), reader(args.results, stack))
    # Written only after every check has passed, so a failed run leaves no file.
    if args.output:
        with args.output.open('x', encoding='utf-8') as output:
            output.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'coverage', 'metric_tolerance')}, indent=2))


if __name__ == '__main__':
    main()
