"""Checkpoint save and load helpers."""

from pathlib import Path

import torch


def save_checkpoint(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint.get("epoch", 0) + 1, checkpoint.get("best_acc", 0.0)


def save_last_and_best(state, checkpoint_dir, is_best):
    checkpoint_dir = Path(checkpoint_dir)
    last_path = checkpoint_dir / "last.pt"
    best_path = checkpoint_dir / "best.pt"
    save_checkpoint(state, last_path)
    if is_best:
        save_checkpoint(state, best_path)
        return last_path, best_path
    return last_path, None
