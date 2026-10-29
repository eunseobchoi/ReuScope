"""Reported CNN runs: pin inputs in a plan, then train one backbone/pool/seed cell.

The archived trainer (src/trainer.py) runs unchanged; observation hooks add no
random operations or extra forward passes. See README.md and SOURCE.json.
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
TRAINER = ROOT / 'src/trainer.py'
ARMS = ('baseline', 'le6', 'random')
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


def load():
    os.environ['CAPSULE_ROOT'] = str(ROOT)
    spec = importlib.util.spec_from_file_location('cnn_trainer', TRAINER)
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
    paths = [TRAINER, Path(__file__), ROOT / 'src/dataset_official.py',
             ROOT / 'src/utils/dataset_kvasir_official.py']
    return {str(p): digest(p) for p in paths}


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


def restore(model, saved, binding):
    import torch
    if saved['binding'] != binding:
        raise ValueError('Checkpoint binding mismatch')
    expected = snapshot(model, binding)
    if any(saved[k] != expected[k] for k in ('trainable_names', 'buffer_names')) or set(saved['state']) != set(expected['state']):
        raise ValueError('Checkpoint state coverage mismatch')
    state = dict(model.named_parameters()) | dict(model.named_buffers())
    with torch.no_grad():
        for name, tensor in saved['state'].items():
            state[name].copy_(tensor)


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
    ap = argparse.ArgumentParser(__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('plan')
    p.add_argument('--output', type=Path, required=True)
    for key in (*ARMS, 'orig-val', 'dedup-val', 'resnet50-weights', 'convnext-tiny-weights'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--skip-kvasir-eval', action='store_true')
    p.add_argument('--rig', required=True, help='Label of the physical GPU shared by all cells of a seed')
    p.add_argument('--seeds', nargs='+', type=int, default=list(range(10)))
    p.add_argument('--batch', type=int, default=128)
    r = sub.add_parser('run')
    r.add_argument('--plan', type=Path, required=True)
    r.add_argument('--output', type=Path, required=True)
    r.add_argument('--backbone', choices=BACKBONES, required=True)
    r.add_argument('--arm', choices=ARMS, required=True)
    r.add_argument('--seed', type=int, required=True)
    r.add_argument('--rig', required=True)
    r.add_argument('--smoke-epochs', type=int, choices=[1, 2])
    args = ap.parse_args()
    if args.command == 'plan':
        if args.batch < 1 or len(set(args.seeds)) != len(args.seeds) or any(not 0 <= s < 2**32 for s in args.seeds):
            ap.error('Positive batch and unique uint32 seeds required')
        m = load()
        lists = {k: str(getattr(args, k).resolve()) for k in (*ARMS, 'orig_val', 'dedup_val')}
        weights = {b: str(getattr(args, b + '_weights').resolve()) for b in BACKBONES}
        pinned = json.loads((ROOT / 'SOURCE.json').read_text())['pretrained_weight_sha256']
        for b, path in weights.items():
            if digest(path) != pinned[b]:
                ap.error('Weights do not match the pinned SHA-256 in SOURCE.json: ' + b)
        plan = dict(schema=1, producer='reported CNN runs',
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
    for p, h in plan['file_hashes'].items():
        if digest(p) != h:
            ap.error('Stale list or weights: ' + p)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not visible or ',' in visible or visible == '-1':
        ap.error('Exactly one explicit CUDA_VISIBLE_DEVICES entry required')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        ap.error('Exactly one actual visible GPU required; CPU fakes are tests only')
    m = load()
    if inventory(m, plan['lists'], plan['skip_kvasir_eval']) != plan['inventory']:
        ap.error('Stale image bytes, order, labels, or Kvasir split')
    args.output.mkdir(parents=True, exist_ok=False)
    binding = dict(plan_sha256=digest(args.plan), backbone=args.backbone,
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


if __name__ == '__main__':
    main()
