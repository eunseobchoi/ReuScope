"""Small synthetic examples of the core operations: python example.py."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from PIL import Image

from analysis import (ARMS, CLASSES, LOCAL_CLASSES, compare, fixed_prediction_rescore,
                      local_contrast, metrics, score_views)
from controls import make_pools
from reuse import byte_identity_summary, internal_rows, measure_images, reference_minima


def image_table(paths, sources, labels):
    frame = pd.DataFrame({'image_path': paths, 'Dataset': sources})
    frame[list(CLASSES)] = np.eye(10, dtype=int)[labels]
    return frame


def main():
    # A synthetic file crossing, including distinct released labels.
    with TemporaryDirectory() as folder:
        pixels = np.arange(96 * 96, dtype=np.uint8).reshape(96, 96)
        for name in ('train.png', 'validation.png'):
            Image.fromarray(pixels).save(Path(folder) / name)
        train = measure_images(image_table(['train.png'], ['SEE-AI'], [0]), folder)
        validation = measure_images(image_table(['validation.png'], ['SEE-AI'], [1]), folder)
        identity = byte_identity_summary(train, validation, folder)
        assert identity['within_source']['SEE-AI']['label_disjoint_groups'] == 1
        matches = internal_rows(train, validation, folder)
        assert matches.selected_ncc99.sum() == matches.any_ncc99.sum() == 1
        assert reference_minima(validation, train).min_phash.iloc[0] == 0

    # The paper's 66-row Ulcer replacement rule, on generated path identifiers.
    train = image_table([f'image_{i}.jpg' for i in range(140)],
                        ['KVASIR'] * 66 + ['AIIMS'] * 66 + ['SEE-AI'] * 4 + ['KVASIR'] * 4,
                        [8] * 132 + [6] * 8)
    val = image_table(['eval_0.jpg', 'eval_1.jpg'], ['KVASIR', 'SEE-AI'], [6, 8])
    pools = make_pools(train, val)
    assert len(pools['source_only']) == len(pools['composition']) == len(pools['ulcer_swap']) == 70
    np.testing.assert_array_equal(pools['composition'][list(CLASSES)].sum(),
                                  pools['ulcer_swap'][list(CLASSES)].sum())

    # Losing a class must not silently shrink the pooled BA target.
    labels = np.arange(10)
    record = dict(row_id=[f'row_{i}' for i in labels], labels=labels,
                  scores=np.eye(10), source=['KVASIR'] * 5 + ['SEE-AI'] * 5)
    views = {'missing': record['row_id'][:-1]}
    for fixed in (True, False):
        kwargs = {} if fixed else {'required_classes': None}
        scored = score_views(record, views, **kwargs)
        assert scored['full']['pooled']['bal_acc'] == 1.
        for result in (metrics(labels[:-1], record['scores'][:-1], **kwargs),
                       scored['missing']['pooled']):
            assert result['ba_classes'] == list(range(10 if fixed else 9))
            assert result['bal_acc'] == result['combined_ovr'] == (None if fixed else 1.)
            assert result['mean_auc_ovr'] == result['acc'] == 1.
            assert result['auc_classes'] == list(range(9))
            assert result['per_class']['Worms']['recall'] is None
        assert scored['missing']['per_source']['SEE-AI']['ba_classes'] == [5, 6, 7, 8]
        assert scored['missing']['equal_source']['bal_acc'] == 1.
        rescore = fixed_prediction_rescore({'example': record}, views, **kwargs)
        changes = rescore['mean_changes']['missing']
        assert changes['pooled']['bal_acc'] == changes['pooled']['combined_ovr'] == (
            None if fixed else 0.)
        assert changes['pooled']['mean_auc_ovr'] == changes['equal_source']['bal_acc'] == 0.
        seeds = {seed: dict(record, seed=seed) for seed in (0, 1)}
        contrast = compare(seeds, seeds, views, **kwargs)
        assert contrast['full']['mean'] == 0.
        assert (contrast['missing'] is None if fixed
                else contrast['missing']['mean'] == 0.)

    # The same fitted-model predictions can reverse a contrast across views.
    labels = np.tile([0, 1], 4)
    ids = np.array([f'row_{i}' for i in range(8)])
    source = np.array(['KVASIR'] * 6 + ['SEE-AI'] * 2)
    views = {'non_kvasir': ids[source != 'KVASIR']}
    full, removal = {}, {}
    for seed in (0, 1):
        full_pred = labels.copy()
        full_pred[6:] = 1 - labels[6:]
        removal_pred = 1 - labels
        removal_pred[6:] = labels[6:]
        if seed == 1:
            removal_pred[:2] = labels[:2]
        base = dict(row_id=ids, labels=labels, source=source, seed=seed)
        full[seed] = dict(base, scores=np.eye(10)[full_pred])
        removal[seed] = dict(base, scores=np.eye(10)[removal_pred])
    contrast = compare(removal, full, views, required_classes=(0, 1))
    assert contrast['full']['mean'] == -.375
    assert contrast['non_kvasir']['mean'] == 1.
    rescore = fixed_prediction_rescore({'example': full[0]}, views, required_classes=(0, 1))
    assert rescore['mean_changes']['non_kvasir']['pooled']['bal_acc'] == -.75

    # Eq. 3: average three distant arms within a seed, then contrast E with U.
    labels = np.tile([0, 1], 2)
    group = np.array(['E', 'E', 'U', 'U'])
    blocks = {}
    for seed, effect in enumerate((.08, .10, .12)):
        for arm in ARMS:
            true_p = np.full(4, .6)
            if arm == 'local':
                true_p[group == 'E'] -= effect
            p = np.repeat(((1 - true_p) / 9)[:, None], 10, axis=1)
            p[np.arange(4), labels] = true_p
            blocks[seed, arm] = dict(row_id=['E0', 'E1', 'U0', 'U1'], labels=labels,
                probabilities=p, seed=seed, arm=arm, group=group,
                video_prefix=['video_a'] * 4)
    local_default = local_contrast(blocks)
    assert local_default['ba_classes'] == list(LOCAL_CLASSES)
    assert local_default['endpoints']['bal_acc'] is None
    local = local_contrast(blocks, required_classes=(0, 1))
    probability = local['endpoints']['mean_true_class_prob']['E_minus_U']
    np.testing.assert_allclose(probability['mean'], -.1, atol=1e-12)
    assert probability['paired_t_95_ci'][1] < 0
    assert local['endpoints']['acc']['E_minus_U']['mean'] == 0
    print('Synthetic checks passed: byte identity, counterpart screening, pool construction,')
    print('fixed-prediction views, matched seeds and local-versus-distant contrasts.')
    print('Training contrast: public = -0.375; non-KVASIR = +1.000.')
    print('Local probability contrast = -0.100; accuracy contrast = 0.')
    print('These are generated examples, not the reported experimental results.')


if __name__ == '__main__':
    main()
