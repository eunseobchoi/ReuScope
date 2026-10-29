"""Export ordered probabilities from a learned DINOv2 checkpoint.

Supply the original frozen backbone weights locally with --backbone-weights.
Input CSVs use the same one-hot schema and external image roots as training.
"""
import argparse
import json
from pathlib import Path
import sys


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backbone-weights", type=Path, required=True)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--recipe", choices=("primary", "local"),
                        help="Required for producer checkpoints without recipe metadata")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch < 1:
        parser.error("batch must be positive")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from utils.input_validation import validate_output_paths
    try:
        validate_output_paths([args.output])
        for path in (args.checkpoint, args.backbone_weights, args.input_csv):
            if not path.is_file():
                raise ValueError(f"Expected an existing input file: {path}")
    except ValueError as exc:
        parser.error(str(exc))

    import torch
    from torch.utils.data import DataLoader
    import dinov2_training as dino

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    dataset = dino.CV2024DS(args.input_csv)
    if isinstance(checkpoint, dict) and checkpoint.get("format") == "cv2024_dinov2_lora_trainable_v1":
        recipe = checkpoint.get("recipe")
        if args.recipe is not None and recipe is not None and args.recipe != recipe:
            parser.error("--recipe conflicts with checkpoint recipe")
        recipe = recipe or args.recipe
        if recipe is None:
            parser.error("This checkpoint has no recipe metadata; supply --recipe primary or local")
        checkpoint = dict(checkpoint, recipe=recipe)
        model, transform = dino.restore_trainable_checkpoint(checkpoint, args.backbone_weights)
        use_amp = True
        f1_policy = "union" if checkpoint.get("recipe") == "local" else "present"
    else:
        raise ValueError("Unsupported checkpoint format")
    dataset.tf = transform
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=0)
    metrics = dino.ev(model.to(dino.DEVICE), loader, dataset=dataset,
                      prediction_output=args.output, use_amp=use_amp, f1_policy=f1_policy)
    print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
