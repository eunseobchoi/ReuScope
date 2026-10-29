"""Inference with a saved CNN classifier checkpoint and the pinned backbone weights."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--backbone-weights', type=Path, required=True)
    parser.add_argument('--input-csv', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=256)
    args = parser.parse_args()
    if args.batch < 1 or args.output.exists():
        parser.error('Positive batch and fresh output required')
    import torch
    m = run.load()
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    binding = saved['binding']
    backbone = binding['backbone']
    if backbone not in run.BACKBONES:
        raise ValueError('Expected a CNN classifier checkpoint from run.py')
    expected_hash = json.loads((run.ROOT / 'SOURCE.json').read_text())['pretrained_weight_sha256'][backbone]
    if run.digest(args.backbone_weights) != expected_hash:
        raise ValueError('Backbone weight SHA256 mismatch')
    with patch.object(m.tvm, backbone, run.pinned_builder(m, backbone, args.backbone_weights)):
        model, transform = m.build_model(backbone)
    reference = model.state_dict()
    for name, tensor in saved['state'].items():
        if (name not in reference or not isinstance(tensor, torch.Tensor)
                or tensor.shape != reference[name].shape or tensor.dtype != reference[name].dtype
                or not torch.isfinite(tensor).all()):
            raise ValueError('Invalid checkpoint tensor: ' + name)
    run.restore(model, saved, binding)
    model.to(m.DEVICE).eval()
    dataset = m.build_dataset_class(transform)(args.input_csv)
    loader = m.DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=0)
    logits = []
    def capture(_model, _inputs, output):
        logits.append(output.detach().cpu())
    handle = model.register_forward_hook(capture)
    try:
        metrics = m.eval_one(model, loader, len(m.CV2024_CLASSES))
    finally:
        handle.remove()
    logits = torch.cat(logits)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as stream:
        torch.save(dict(classes=m.CV2024_CLASSES, logits=logits,
                        predictions=logits.argmax(-1), probabilities=logits.float().softmax(-1),
                        labels=torch.tensor([label for _, label in dataset.items]),
                        row_indices=torch.arange(len(dataset)), metrics=metrics), stream)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
