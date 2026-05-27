"""CIFAR-10 dataset construction."""

from pathlib import Path

from torch.utils.data import random_split
from torchvision.datasets import CIFAR10
import torch


def build_cifar10_datasets(args, train_transform, test_transform):
    root = Path(args.data_dir) / "cifar10"
    full_train = CIFAR10(root=str(root), train=True, transform=train_transform, download=args.download)
    test_dataset = CIFAR10(root=str(root), train=False, transform=test_transform, download=args.download)

    val_ratio = max(0.0, min(1.0, args.val_ratio))
    if val_ratio > 0:
        val_size = int(len(full_train) * val_ratio)
        train_size = len(full_train) - val_size
        generator = torch.Generator().manual_seed(args.seed)
        train_subset, val_indices_subset = random_split(
            full_train, [train_size, val_size], generator=generator
        )
        val_base = CIFAR10(root=str(root), train=True, transform=test_transform, download=False)
        val_subset = torch.utils.data.Subset(val_base, val_indices_subset.indices)
    else:
        train_subset = full_train
        val_subset = test_dataset
    return train_subset, val_subset, test_dataset, 10
