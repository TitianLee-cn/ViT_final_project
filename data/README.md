# Data

CIFAR-10 can be downloaded automatically:

```bash
python main.py --dataset cifar10 --download --dry-run
```

It will be stored under `data/cifar10`.

Tiny ImageNet is not downloaded by this project. Place it manually as:

```text
data/tiny-imagenet-200/
├── train/
├── val/
├── test/
├── wnids.txt
└── words.txt
```

The validation labels are read from `val/val_annotations.txt`.
