"""DINOv2 primary recipe: macro-F1 over represented true classes."""
import argparse


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--orig_val_csv", required=True)
    ap.add_argument("--dedup_val_csv", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument(
        "--backbone",
        choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"],
        default="dinov2_vitl14",
        help="DINOv2 backbone; default: vitl14.",
    )
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--prediction_output",
        help=(
            "Write final-epoch per-frame probabilities for --dedup_val_csv. "
            "Single-seed invocations only."
        ),
    )
    ap.add_argument(
        "--trainable_checkpoint",
        help=(
            "Write the final learned head/LoRA tensors without duplicating "
            "the frozen DINOv2 backbone. Single-seed invocations only."
        ),
    )
    ap.add_argument("--no_batch128_assert", action="store_true")
    ap.add_argument(
        "--skip_kvasir_eval",
        action="store_true",
        help="Skip the external Kvasir split_1 evaluator for CV2024-only evaluation.",
    )
    return ap



def main():
    parser = build_parser()
    args = parser.parse_args()
    from dinov2_training import main as train
    train(parser, args, recipe="primary", script_path=__file__)


if __name__ == "__main__":
    main()
