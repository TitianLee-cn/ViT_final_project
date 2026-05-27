"""Image-level transforms."""

from torchvision import transforms

from src.augmentations.randaugment import build_randaugment


def get_dataset_stats(dataset_name):
    if dataset_name == "cifar10":
        return (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    if dataset_name == "tiny-imagenet":
        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _image_size(args, dataset_name):
    if args.image_size is not None:
        return args.image_size
    return 32 if dataset_name == "cifar10" else 64


def build_train_transform(args, dataset_name):
    size = _image_size(args, dataset_name)
    mean, std = get_dataset_stats(dataset_name)
    ops = []
    if dataset_name != "cifar10" or size != 32:
        ops.append(transforms.Resize((size, size)))
    if args.random_crop:
        ops.append(transforms.RandomCrop(size, padding=args.crop_padding))
    if args.horizontal_flip:
        ops.append(transforms.RandomHorizontalFlip(p=args.hflip_prob))
    if args.randaugment:
        ops.append(build_randaugment(args.ra_num_ops, args.ra_magnitude))
    ops.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transforms.Compose(ops)


def build_test_transform(args, dataset_name):
    size = _image_size(args, dataset_name)
    mean, std = get_dataset_stats(dataset_name)
    ops = []
    if dataset_name != "cifar10" or size != 32:
        ops.append(transforms.Resize((size, size)))
    ops.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transforms.Compose(ops)
