"""RandAugment wrapper."""

from torchvision import transforms


def build_randaugment(num_ops, magnitude):
    if not hasattr(transforms, "RandAugment"):
        raise RuntimeError("torchvision.transforms.RandAugment is unavailable in this torchvision version.")
    return transforms.RandAugment(num_ops=num_ops, magnitude=magnitude)
