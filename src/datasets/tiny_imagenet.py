"""Tiny ImageNet dataset interface."""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class TinyImageNetValDataset(Dataset):
    def __init__(self, root, transform=None, class_to_idx=None):
        self.root = Path(root)
        self.transform = transform
        self.class_to_idx = class_to_idx or {}
        ann_path = self.root / "val_annotations.txt"
        img_dir = self.root / "images"
        if not ann_path.exists():
            raise FileNotFoundError(f"Missing Tiny ImageNet val annotations: {ann_path}")
        self.samples = []
        with open(ann_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                filename, wnid = parts[0], parts[1]
                if wnid not in self.class_to_idx:
                    continue
                self.samples.append((img_dir / filename, self.class_to_idx[wnid]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def build_tiny_imagenet_datasets(args, train_transform, test_transform):
    root = Path(args.data_dir) / "tiny-imagenet-200"
    if not root.exists():
        raise FileNotFoundError(
            "Tiny ImageNet not found. Please place it at data/tiny-imagenet-200 "
            "with train/, val/, test/, wnids.txt and words.txt."
        )
    train_root = root / "train"
    val_root = root / "val"
    if not train_root.exists() or not val_root.exists():
        raise FileNotFoundError(f"Tiny ImageNet train/val folders are missing under {root}")

    train_dataset = ImageFolder(str(train_root), transform=train_transform)
    val_dataset = TinyImageNetValDataset(val_root, transform=test_transform, class_to_idx=train_dataset.class_to_idx)
    test_dataset = val_dataset
    return train_dataset, val_dataset, test_dataset, 200
