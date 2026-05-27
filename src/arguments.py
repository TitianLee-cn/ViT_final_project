"""Command line and YAML configuration parsing."""

import argparse
from pathlib import Path

import yaml


def str2bool_default(value):
    return bool(value)


class TrackDefaultsArgumentParser(argparse.ArgumentParser):
    """Parser that records explicitly supplied option destinations."""

    def parse_known_args(self, args=None, namespace=None):
        option_to_dest = {}
        for action in self._actions:
            for option in action.option_strings:
                option_to_dest[option] = action.dest
        import sys

        tokens = list(sys.argv[1:] if args is None else args)
        explicit = set()
        for token in tokens:
            option = token.split("=", 1)[0]
            if option in option_to_dest:
                explicit.add(option_to_dest[option])
        parsed, extras = super().parse_known_args(args=args, namespace=namespace)
        parsed._explicit_args = explicit
        return parsed, extras


def load_yaml_config(path):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def merge_config_with_args(config, args):
    for key, value in config.items():
        if key not in args._explicit_args and hasattr(args, key):
            setattr(args, key, value)
    return args


def build_parser():
    parser = TrackDefaultsArgumentParser(description="ViT image classification experiments")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save", type=str, default=None)

    parser.add_argument("--dataset", choices=["cifar10", "tiny-imagenet"], default="cifar10")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--download", action="store_true")

    parser.add_argument("--model", choices=["cnn", "vit", "hybrid"], default="vit")
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--cnn-depth", type=int, choices=[18, 34, 50], default=18)
    parser.add_argument("--pretrained", action="store_true")

    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--vit-dim", type=int, default=256)
    parser.add_argument("--vit-depth", type=int, default=6)
    parser.add_argument("--vit-heads", type=int, default=8)
    parser.add_argument("--vit-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--qkv-bias", dest="qkv_bias", action="store_true", default=True)
    parser.add_argument("--no-qkv-bias", dest="qkv_bias", action="store_false")
    parser.add_argument("--cls-token", dest="cls_token", action="store_true", default=True)
    parser.add_argument("--no-cls-token", dest="cls_token", action="store_false")

    parser.add_argument("--hybrid-stage", choices=["layer2", "layer3", "layer4"], default="layer3")
    parser.add_argument("--hybrid-freeze-cnn", action="store_true")
    parser.add_argument("--hybrid-patch-size", type=int, default=1)

    parser.add_argument("--random-crop", action="store_true")
    parser.add_argument("--crop-padding", type=int, default=4)
    parser.add_argument("--horizontal-flip", action="store_true")
    parser.add_argument("--hflip-prob", type=float, default=0.5)
    parser.add_argument("--randaugment", action="store_true")
    parser.add_argument("--ra-num-ops", type=int, default=2)
    parser.add_argument("--ra-magnitude", type=int, default=9)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--cutmix-alpha", type=float, default=0.0)
    parser.add_argument("--mixup-prob", type=float, default=1.0)
    parser.add_argument("--mixup-switch-prob", type=float, default=0.5)

    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--attention-dropout", type=float, default=0.0)
    parser.add_argument("--drop-path-rate", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--scheduler", choices=["cosine", "step", "none"], default="cosine")
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=None)
    return parser


def parse_args():
    parser = build_parser()
    args = parser.parse_args()
    if args.config:
        config_path = Path(args.config)
        config = load_yaml_config(config_path)
        args = merge_config_with_args(config, args)
    return args


def get_args():
    return parse_args()
