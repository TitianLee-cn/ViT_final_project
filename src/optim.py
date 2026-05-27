"""Optimizer and learning-rate scheduler builders."""

import torch


def build_optimizer(args, model):
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def build_scheduler(args, optimizer, steps_per_epoch=None):
    if args.scheduler == "none":
        return None
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs), eta_min=args.min_lr
        )
    if args.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs // 3), gamma=0.1)
    raise ValueError(f"Unsupported scheduler: {args.scheduler}")
