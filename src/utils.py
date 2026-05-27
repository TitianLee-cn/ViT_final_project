"""General utility functions."""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(args):
    if args.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(args.device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def to_serializable(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if hasattr(obj, "__dict__"):
        return {k: to_serializable(v) for k, v in vars(obj).items() if not k.startswith("_")}
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    return obj


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(obj), f, indent=2, ensure_ascii=False)


def create_experiment_name(args):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if args.model == "cnn":
        base = f"{args.dataset}_cnn_resnet{args.cnn_depth}"
    elif args.model == "vit":
        base = f"{args.dataset}_vit_p{args.patch_size}_d{args.vit_depth}_h{args.vit_heads}_dim{args.vit_dim}"
    else:
        base = (
            f"{args.dataset}_hybrid_resnet{args.cnn_depth}_{args.hybrid_stage}"
            f"_d{args.vit_depth}_h{args.vit_heads}_dim{args.vit_dim}"
        )
    augreg = any(
        [
            args.random_crop,
            args.horizontal_flip,
            args.randaugment,
            args.mixup_alpha > 0,
            args.cutmix_alpha > 0,
            args.drop_path_rate > 0,
            args.label_smoothing > 0,
        ]
    )
    if augreg:
        base += "_augreg"
    return f"{base}_{stamp}"
