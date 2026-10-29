"""Entry point for the archived official-test producer (producer.py, unchanged).

All options are forwarded to the producer. Set DINOV2_REPO to a local Torch Hub
checkout of DINOv2; the original runs did not record its revision or weights.
"""
import argparse
import os
from pathlib import Path
import runpy
import sys
from unittest.mock import patch


def main():
    root = Path(__file__).resolve().parent
    os.environ['CAPSULE_ROOT'] = str(root)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output')
    args, _ = parser.parse_known_args()
    if args.output:
        output = root / 'results' / args.output
        for path in (output, output.with_name(output.name + '.partial'), output.with_name(output.name + '.tmp')):
            if path.exists():
                parser.error('Fresh output required: ' + str(path))
    if '--help' in sys.argv or '-h' in sys.argv:
        runpy.run_path(str(root / 'producer.py'), run_name='__main__')
        return
    local_repo = os.environ.get('DINOV2_REPO')
    if not local_repo or not Path(local_repo).is_dir():
        parser.error('Set DINOV2_REPO to reader-provided local DINOv2 Torch Hub code')
    if not os.environ.get('CV2024_ROOT'):
        parser.error('Set CV2024_ROOT to the external Dataset root')
    import torch
    original_load = torch.hub.load
    def local_load(repo, model, *args, **kwargs):
        if repo != 'facebookresearch/dinov2' or model != 'dinov2_vitl14':
            raise ValueError('Unexpected backbone contract')
        return original_load(local_repo, model, *args, source='local', **kwargs)
    with patch.object(torch.hub, 'load', local_load):
        runpy.run_path(str(root / 'producer.py'), run_name='__main__')


if __name__ == '__main__':
    main()
