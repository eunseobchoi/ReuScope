"""Plan, run and summarize fixed-list CNN comparisons with local weights.

Supply full-training, source-only (excluding Kvasir), and matched-random CSV lists.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import socket
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
SOURCES = {'fixed': ROOT / 'src/cross_model.py'}
ARMS = ('baseline', 'source_only', 'random')
BACKBONES = ('resnet50', 'convnext_tiny')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def load(variant):
    os.environ['CAPSULE_ROOT'] = str(ROOT)
    spec = importlib.util.spec_from_file_location('crossmodel_' + variant, SOURCES[variant])
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def versions():
    import torch, torchvision, numpy, pandas, sklearn, PIL
    return dict(python=platform.python_version(), torch=torch.__version__,
                torchvision=torchvision.__version__, numpy=numpy.__version__,
                pandas=pandas.__version__, sklearn=sklearn.__version__, pillow=PIL.__version__,
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())


def provenance():
    paths = [*SOURCES.values(), Path(__file__),
             ROOT / 'src/dataset_official.py',
             ROOT / 'src/utils/dataset_kvasir_official.py',
             ROOT / 'src/utils/path_resolution.py', ROOT / 'crossmodel_weight_hashes.json']
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def inventory(m, lists, skip):
    """Ordered labels and pixel identities; detect changed images as well as lists."""
    DS = m.build_dataset_class(m.TF_RESNET)
    pools = {k: DS(p) for k, p in lists.items()}
    if not skip:
        pools['kvasir_s1'] = m.KvasirCapsuleAdapter(1, m.TF_RESNET)
    hashes = {}
    result = {}
    for name, ds in pools.items():
        rows = []
        for path, label in ds.items:
            p = str(Path(path).resolve())
            if p not in hashes:
                hashes[p] = digest(p)
            rows.append([p, label, hashes[p]])
        result[name] = rows
    return result


def snapshot(model, binding):
    """Frozen weights are pinned separately; every buffer must travel with the head."""
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    names.update(n for n, _ in model.named_buffers())
    state = dict(model.named_parameters()) | dict(model.named_buffers())
    return {'binding': binding, 'state': {n: state[n].detach().cpu().clone() for n in sorted(names)},
            'trainable_names': sorted(n for n, p in model.named_parameters() if p.requires_grad),
            'buffer_names': sorted(n for n, _ in model.named_buffers())}


def observe(m, backbone, train, seed, args, orig, dedup, out, binding, skip=False):
    """Run the imported loop verbatim. Hooks use no random operations or extra forwards."""
    import torch
    built, steps, exports = [], [], {}
    build, evaluate, scaler_type = m.build_model, m.eval_one, torch.cuda.amp.GradScaler
    calls = 0

    def capture_build(name):
        model, tf = build(name)
        built.append(model)
        return model, tf

    class ObservedScaler(scaler_type):
        def update(self, *a, **kw):
            before = self.get_scale()
            result = super().update(*a, **kw)
            after = self.get_scale()
            steps.append({'scale_before': before, 'scale_after': after, 'skipped': after < before})
            return result

    def capture_eval(model, loader, n):
        nonlocal calls
        pool = ('orig_val', 'dedup_val', 'kvasir_s1')[calls % 3]
        final = calls // 3 == args.epochs - 1
        calls += 1
        if pool == 'kvasir_s1' and skip:
            # NaN is only a print-format sentinel; replaced before serialization.
            return {'bal_acc': float('nan'), 'status': 'skipped', 'reason': 'explicit plan skip'}
        probabilities, logits_saved, predictions = [], [], []
        def hook(_model, _inputs, logits):
            if not torch.isfinite(logits).all():
                raise RuntimeError('Nonfinite evaluation logits; diagnostic failed')
            if final:
                logits_saved.append(logits.detach().cpu())
                predictions.append(logits.detach().argmax(-1).cpu())
                probabilities.append(logits.detach().float().softmax(-1).cpu())
        handle = model.register_forward_hook(hook)
        try:
            metrics = evaluate(model, loader, n)
        finally:
            handle.remove()
        if final:
            p = torch.cat(probabilities)
            if len(p) != len(loader.dataset.items) or not torch.isfinite(p).all():
                raise ValueError('Invalid final probability export')
            export = dict(binding=binding, pool=pool, classes=m.CV2024_CLASSES,
                          items=loader.dataset.items, probabilities=p,
                          logits=torch.cat(logits_saved), predictions=torch.cat(predictions))
            torch.save(export, out / (pool + '.pt'))
            exports[pool] = digest(out / (pool + '.pt'))
        return metrics

    with patch.object(m, 'build_model', capture_build), patch.object(m, 'eval_one', capture_eval), \
         patch.object(torch.cuda.amp, 'GradScaler', ObservedScaler), \
         patch.object(m, 'KvasirCapsuleAdapter', (lambda **kw: []) if skip else m.KvasirCapsuleAdapter):
        result = m.train_one_seed(backbone, train, seed, args, orig, dedup)
    if calls != args.epochs * 3:
        raise ValueError('Unexpected evaluator call contract')
    if skip:
        for row in result['history']:
            row['kvasir_s1'] = {'status': 'skipped', 'reason': 'explicit plan skip'}
    torch.save(snapshot(built[0], binding), out / 'classifier.pt')
    exports['classifier'] = digest(out / 'classifier.pt')
    return dict(result=result, step_trace=steps, optimizer_steps=sum(not s['skipped'] for s in steps),
                amp_skipped_steps=sum(s['skipped'] for s in steps), exports=exports)


def pinned_builder(m, backbone, weight):
    """Use the original constructor/RNG path, replacing only the weight download."""
    import torch
    constructor = getattr(m.tvm, backbone)
    enum = m.tvm.ResNet50_Weights.DEFAULT if backbone == 'resnet50' else m.tvm.ConvNeXt_Tiny_Weights.DEFAULT
    def build(**kw):
        if kw != {'weights': enum}:
            raise ValueError('Unexpected pretrained constructor contract')
        net = constructor(weights=None)
        net.load_state_dict(torch.load(weight, map_location='cpu', weights_only=True), strict=True)
        return net
    return build


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('plan')
    p.add_argument('--output', type=Path, required=True)
    for key in (*ARMS, 'orig-val', 'dedup-val', 'resnet50-weights', 'convnext-tiny-weights'):
        p.add_argument('--' + key.replace('_', '-'), type=Path, required=True,
                       help='Training CSV excluding Kvasir images' if key == 'source_only' else None)
    p.add_argument('--skip-kvasir-eval', action='store_true')
    p.add_argument('--rig', required=True, help='Stable same GPU/allocation identity for all paired cells')
    p.add_argument('--seeds', nargs='+', type=int, default=list(range(10)))
    p.add_argument('--batch', type=int, default=128)
    r = sub.add_parser('run')
    r.add_argument('--plan', type=Path, required=True)
    r.add_argument('--output', type=Path, required=True)
    r.add_argument('--variant', choices=list(SOURCES), required=True,
                   help='Fixed frozen-backbone CNN configuration')
    r.add_argument('--backbone', choices=BACKBONES, required=True)
    r.add_argument('--arm', choices=ARMS, required=True)
    r.add_argument('--seed', type=int, required=True)
    r.add_argument('--rig', required=True)
    r.add_argument('--smoke-epochs', type=int, choices=[1, 2],
                   help='Short execution check; shortens the scheduler horizon and is excluded from summarize')
    s = sub.add_parser('summarize')
    s.add_argument('--plan', type=Path, required=True)
    s.add_argument('--runs', nargs='+', type=Path, required=True)
    s.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.command == 'summarize':
        summarize(args)
        return
    if args.command == 'plan':
        if args.batch < 1 or len(set(args.seeds)) != len(args.seeds) or any(not 0 <= s < 2**32 for s in args.seeds):
            ap.error('Positive batch and unique uint32 seeds required')
        m = load('fixed')
        lists = {k: str(getattr(args, k).resolve()) for k in (*ARMS, 'orig_val', 'dedup_val')}
        weights = {b: str(getattr(args, b + '_weights').resolve()) for b in BACKBONES}
        expected_hashes = json.loads((ROOT / 'crossmodel_weight_hashes.json').read_text())
        for b, path in weights.items():
            enum = m.tvm.ResNet50_Weights.DEFAULT if b == 'resnet50' else m.tvm.ConvNeXt_Tiny_Weights.DEFAULT
            if digest(path) != expected_hashes[Path(enum.url).name]:
                ap.error('Weights do not match pinned SHA256: ' + b)
        plan = dict(schema=1, diagnostic='fixed-list CNN training',
                    sources=provenance(), versions=versions(), classes=m.CV2024_CLASSES,
                    lists=lists, weights=weights, file_hashes={p: digest(p) for p in [*lists.values(), *weights.values()]},
                    inventory=inventory(m, lists, args.skip_kvasir_eval), seeds=args.seeds, batch=args.batch,
                    epochs=10, rig=args.rig, skip_kvasir_eval=args.skip_kvasir_eval,
                    config=dict(optimizer='AdamW', lr=3e-4, weight_decay=1e-4, label_smoothing=.05,
                                cbw_beta=.999, train_workers=6, drop_last=True, preprocessing=repr(m.TF_RESNET),
                                scheduler='CosineAnnealingLR', amp={'resnet50': 'float16', 'convnext_tiny': 'bfloat16'},
                                determinism='seeded warn-only; not bit-exact'),
                    weight_enums={b: str(m.tvm.ResNet50_Weights.DEFAULT if b == 'resnet50' else m.tvm.ConvNeXt_Tiny_Weights.DEFAULT) for b in BACKBONES})
        write(args.output, plan)
        return
    plan = json.loads(args.plan.read_text())
    if args.output.exists():
        ap.error('Fresh output directory required, including failed/smoke attempts')
    if provenance() != plan['sources'] or versions() != plan['versions']:
        ap.error('Stale source or mixed runtime')
    if args.rig != plan['rig'] or args.seed not in plan['seeds']:
        ap.error('Rig/seed outside pinned plan')
    if set(plan['file_hashes']) != set(plan['lists'].values()) | set(plan['weights'].values()):
        ap.error('Plan input paths do not match pinned file hashes')
    for p, h in plan['file_hashes'].items():
        if digest(p) != h:
            ap.error('Stale list or weights: ' + p)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not visible or ',' in visible or visible == '-1':
        ap.error('Exactly one explicit CUDA_VISIBLE_DEVICES entry required')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        ap.error('Exactly one actual visible GPU required')
    m = load(args.variant)
    if inventory(m, plan['lists'], plan['skip_kvasir_eval']) != plan['inventory']:
        ap.error('Stale image bytes, order, labels, or Kvasir split')
    args.output.mkdir(parents=True, exist_ok=False)
    binding = dict(plan_sha256=digest(args.plan), variant=args.variant, backbone=args.backbone,
                   arm=args.arm, seed=args.seed, epochs=args.smoke_epochs or 10,
                   tier='smoke' if args.smoke_epochs else 'full', rig=args.rig,
                   host=socket.gethostname(), gpu_name=torch.cuda.get_device_name(0), visible_gpu=visible,
                   gpu_uuid=str(getattr(torch.cuda.get_device_properties(0), 'uuid', 'unavailable')))
    write(args.output / 'binding.json', binding)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        with patch.object(m.tvm, args.backbone, pinned_builder(m, args.backbone, plan['weights'][args.backbone])):
            result = observe(m, args.backbone, plan['lists'][args.arm], args.seed,
                             SimpleNamespace(batch=plan['batch'], epochs=binding['epochs']),
                             plan['lists']['orig_val'], plan['lists']['dedup_val'], args.output,
                             binding, plan['skip_kvasir_eval'])
        torch.cuda.synchronize()
        result.update(binding=binding, runtime=dict(host=socket.gethostname(), pbs_jobid=os.environ.get('PBS_JOBID'),
                      torch_threads=torch.get_num_threads(), environment={k: os.environ.get(k) for k in
                      ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'CUBLAS_WORKSPACE_CONFIG', 'CUDA_VISIBLE_DEVICES')}),
                      scheduler_steps=result['optimizer_steps'],
                      wall_seconds=time.perf_counter()-start,
                      timing_scope='synchronized model construction, training, every-epoch evaluation and final exports; excludes preflight hashing',
                      peak_cuda_bytes=torch.cuda.max_memory_allocated())
        write(args.output / 'complete.json', result)
    except Exception as exc:
        write(args.output / 'failed.json', dict(binding=binding, error=repr(exc), wall_seconds=time.perf_counter()-start))
        raise


def summarize(args):
    plan = json.loads(args.plan.read_text())
    plan_hash = digest(args.plan)
    cells, rig = {}, None
    for directory in args.runs:
        r = json.loads((directory / 'complete.json').read_text())
        b = r['binding']
        if b['plan_sha256'] != plan_hash or b['tier'] != 'full' or b['epochs'] != 10:
            raise ValueError('Mixed/stale/smoke result')
        if b['rig'] != plan['rig']:
            raise ValueError('Rig outside pinned plan')
        identity = (b['rig'], b['host'], b['gpu_name'], b['visible_gpu'], b['gpu_uuid'])
        if rig is not None and identity != rig:
            raise ValueError('Mixed physical rig')
        rig = identity
        key = (b['backbone'], b['seed'], b['variant'], b['arm'])
        if key in cells:
            raise ValueError('Duplicate cell')
        if b['seed'] not in plan['seeds'] or b['backbone'] not in BACKBONES or b['variant'] not in SOURCES or b['arm'] not in ARMS:
            raise ValueError('Unknown cell')
        expected = {'classifier', 'orig_val', 'dedup_val'} | (set() if plan['skip_kvasir_eval'] else {'kvasir_s1'})
        if set(r['exports']) != expected or json.loads((directory / 'binding.json').read_text()) != b:
            raise ValueError('Incomplete exports or mixed binding')
        for name, h in r['exports'].items():
            if digest(directory / ('classifier.pt' if name == 'classifier' else name + '.pt')) != h:
                raise ValueError('Changed export')
        if (len(r['result']['history']) != 10 or r['result']['last'] != r['result']['history'][-1]
                or [h['epoch'] for h in r['result']['history']] != list(range(1, 11))
                or r['result']['seed'] != b['seed'] or r['result']['backbone'] != b['backbone']):
            raise ValueError('Incomplete final-epoch history')
        cells[key] = r
    rows = []
    for backbone in BACKBONES:
        for seed in plan['seeds']:
            needed = [(backbone, seed, v, a) for v in SOURCES for a in ARMS]
            if not all(k in cells for k in needed):
                rows.append(dict(backbone=backbone, seed=seed, status='missing', missing=[list(k) for k in needed if k not in cells]))
                continue
            for pool in ('orig_val', 'dedup_val'):
                for arm in ('source_only', 'random'):
                    delta = cells[backbone, seed, 'fixed', arm]['result']['last'][pool]['bal_acc'] - cells[backbone, seed, 'fixed', 'baseline']['result']['last'][pool]['bal_acc']
                    rows.append(dict(backbone=backbone, seed=seed, pool=pool, contrast=arm+'-baseline', fixed=delta))
    write(args.output, dict(plan_sha256=plan_hash, paired_contrasts=rows,
                           cell_costs=[dict(cell=list(k), seconds=r['wall_seconds']) for k,r in cells.items()],
                           note='Descriptive paired seed contrasts; missing blocks explicit'))


if __name__ == '__main__':
    main()
