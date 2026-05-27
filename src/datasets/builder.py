"""Unified DataLoader builder."""

from torch.utils.data import DataLoader

from .cifar10 import build_cifar10_datasets
from .tiny_imagenet import build_tiny_imagenet_datasets
from .transforms import build_test_transform, build_train_transform, get_dataset_stats


def build_dataloaders(args):
    dataset_name = args.dataset
    if args.image_size is None:
        args.image_size = 32 if dataset_name == "cifar10" else 64
    train_transform = build_train_transform(args, dataset_name)
    test_transform = build_test_transform(args, dataset_name)

    if dataset_name == "cifar10":
        train_dataset, val_dataset, test_dataset, num_classes = build_cifar10_datasets(
            args, train_transform, test_transform
        )
    elif dataset_name == "tiny-imagenet":
        train_dataset, val_dataset, test_dataset, num_classes = build_tiny_imagenet_datasets(
            args, train_transform, test_transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    pin_memory = str(args.device) != "cpu"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    mean, std = get_dataset_stats(dataset_name)
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "num_classes": num_classes,
        "image_size": args.image_size,
        "dataset_name": dataset_name,
        "mean": mean,
        "std": std,
    }
