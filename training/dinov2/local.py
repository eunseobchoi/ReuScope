"""DINOv2 local recipe: macro-F1 over true/predicted label union."""
from primary import build_parser


def main():
    parser = build_parser()
    parser.set_defaults(seeds=[14, 15, 16, 17, 4, 5, 6, 7, 8, 9])
    args = parser.parse_args()
    from dinov2_training import main as train
    train(parser, args, recipe="local", script_path=__file__)


if __name__ == "__main__":
    main()
