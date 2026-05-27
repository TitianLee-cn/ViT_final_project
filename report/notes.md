# Experiment Notes

## Goal

Compare CNN baseline, standard ViT, and Hybrid CNN+ViT on CIFAR-10, with Tiny ImageNet support for extension.

## Setup

- Dataset:
- Image size:
- Optimizer:
- Scheduler:
- Epochs:
- Batch size:
- Hardware:

## Model Comparison

| Model | Params | Augmentation | Regularization | Val Top-1 | Val Top-5 | Test Top-1 | Test Top-5 |
|---|---:|---|---|---:|---:|---:|---:|
| ResNet-18 | | | | | | | |
| ViT | | | | | | | |
| Hybrid CNN+ViT | | | | | | | |

## Ablation

| Experiment | Changed Setting | Val Top-1 | Test Top-1 | Notes |
|---|---|---:|---:|---|
| Patch size | 4 vs 8 | | | |
| DropPath | 0.0 vs 0.1 | | | |
| RandAugment | off vs on | | | |
| MixUp/CutMix | off vs on | | | |

## Analysis Prompts

- Compare ResNet and ViT in convergence speed and data efficiency.
- Explain whether strong augmentation improves ViT more than CNN.
- Discuss effects of DropPath, Label Smoothing, and Weight Decay.
- Check whether Hybrid CNN+ViT is more stable than pure ViT on small images.
