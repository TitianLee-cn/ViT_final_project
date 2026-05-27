# Vision Transformer for Image Classification

## 先看这里：各个 sh 文件怎么用

项目里有两类 shell 脚本，它们都可以在本地运行，也都可以写进远程服务器的提交脚本里。

```text
run.sh                  # 通用模板：一个文件里切换不同模型/配置
scripts/run_*.sh        # 专用模板：每个文件对应一个具体实验
```

推荐用法：

- 想跑某个明确模型：改对应的 `scripts/run_*.sh`
- 想跑服务器任务：可以直接提交对应的 `scripts/run_*.sh`
- 想用一个文件反复切换模型：改根目录的 `run.sh`
- 想做消融实验：改 `scripts/run_ablation.sh`

### 1. scripts 目录下每个 sh 文件的用途

每个脚本顶部都有完整参数区。实验过程中主要改顶部变量，不需要去改 Python 代码。

```text
scripts/run_resnet18.sh      # CNN baseline，默认 ResNet-18，可改 CNN_DEPTH=18/34/50
scripts/run_vit_basic.sh     # Basic ViT，不启用强增强，适合作为纯 ViT baseline
scripts/run_vit_augreg.sh    # AugReg ViT，启用 RandAugment/MixUp/CutMix/DropPath 等
scripts/run_hybrid.sh        # Hybrid CNN+ViT，调 CNN stage、ViT 深度、是否冻结 CNN
scripts/run_ablation.sh      # 消融实验，一次跑 patch size、DropPath、RandAugment、MixUp/CutMix 对比
```

运行方式：

```bash
bash scripts/run_resnet18.sh
bash scripts/run_vit_basic.sh
bash scripts/run_vit_augreg.sh
bash scripts/run_hybrid.sh
bash scripts/run_ablation.sh
```

远程服务器提交任务时也可以直接用对应脚本。例如提交 AugReg ViT：

```bash
cd /mnt/e/AI4Titi/0.8_DeepLearning/Final_Project/ViT_model
bash scripts/run_vit_augreg.sh
```

### 2. 每个模型脚本里应该改哪些参数

所有脚本都已经把可调参数显式写出来了。常改的在顶部：

```bash
EXPERIMENT_NAME="main_vit_augreg"
EPOCHS=100
BATCH_SIZE=128
NUM_WORKERS=4
LR=0.0003
DEVICE="auto"
AMP="--amp"
DRY_RUN=""
DOWNLOAD="--download"
RESUME=""
```

不同模型有各自重点：

`scripts/run_resnet18.sh`：

```bash
CNN_DEPTH=18
OPTIMIZER="sgd"
LR=0.1
WEIGHT_DECAY=0.0005
```

`scripts/run_vit_basic.sh` 和 `scripts/run_vit_augreg.sh`：

```bash
PATCH_SIZE=4
VIT_DIM=256
VIT_DEPTH=6
VIT_HEADS=8
VIT_MLP_RATIO=4.0
DROP_PATH_RATE=0.0
MIXUP_ALPHA=0.0
CUTMIX_ALPHA=0.0
RANDAUGMENT=""
```

`scripts/run_hybrid.sh`：

```bash
CNN_DEPTH=18
HYBRID_STAGE="layer3"
HYBRID_FREEZE_CNN=""
HYBRID_PATCH_SIZE=1
VIT_DEPTH=4
VIT_DIM=256
VIT_HEADS=8
```

`scripts/run_ablation.sh`：

```bash
EPOCHS=100
BATCH_SIZE=128
VIT_DIM=256
VIT_DEPTH=6
VIT_HEADS=8
```

然后在文件底部选择要跑哪些消融。不要跑的行可以注释掉，例如：

```bash
# run_vit "ablation_patch8" 8 "$DROP_PATH_RATE" "$RANDAUGMENT" "$MIXUP_ALPHA" "$CUTMIX_ALPHA"
```

### 3. run.sh 什么时候用

`run.sh` 是通用模板，不是服务器专用，也不是必须用。它适合这种情况：你想只维护一个文件，通过改 `MODEL`、`CONFIG` 等变量切换实验。

```text
run.sh
```

主要改：

```bash
CONFIG="configs/cifar10_vit_augreg.yaml"
EXPERIMENT_NAME="main_vit_augreg"
EPOCHS=100
BATCH_SIZE=128
NUM_WORKERS=4
LR=0.0003
AMP="--amp"
DRY_RUN=""
DOWNLOAD="--download"
```

运行：

```bash
bash run.sh
```

### 4. 常见参数应该在哪里调

| 你想调什么 | 推荐位置 | 参数名 |
|---|---|---|
| ResNet baseline | `scripts/run_resnet18.sh` | `CNN_DEPTH`、`LR`、`WEIGHT_DECAY` |
| Basic ViT | `scripts/run_vit_basic.sh` | `PATCH_SIZE`、`VIT_DEPTH`、`VIT_DIM`、`VIT_HEADS` |
| AugReg ViT | `scripts/run_vit_augreg.sh` | `RANDAUGMENT`、`MIXUP_ALPHA`、`CUTMIX_ALPHA`、`DROP_PATH_RATE` |
| Hybrid | `scripts/run_hybrid.sh` | `CNN_DEPTH`、`HYBRID_STAGE`、`HYBRID_FREEZE_CNN`、`VIT_DEPTH` |
| 消融实验 | `scripts/run_ablation.sh` | 文件底部的 `run_vit ...` 行 |
| epoch | 任意脚本顶部 | `EPOCHS` |
| batch size | 任意脚本顶部 | `BATCH_SIZE` |
| 学习率 | 任意脚本顶部 | `LR` |
| GPU/CPU | 任意脚本顶部 | `DEVICE` |
| 是否混合精度 | 任意脚本顶部 | `AMP="--amp"` 或 `AMP=""` |
| 是否 dry-run | 任意脚本顶部 | `DRY_RUN="--dry-run"` 或 `DRY_RUN=""` |
| 是否下载 CIFAR-10 | 任意脚本顶部 | `DOWNLOAD="--download"` 或 `DOWNLOAD=""` |
| 从 checkpoint 恢复 | 任意脚本顶部 | `RESUME="--resume path/to/last.pt"` |

### 5. YAML 和 sh 参数的关系

现在 `scripts/run_*.sh` 主要直接写完整命令参数，不依赖 YAML。`configs/*.yaml` 仍然可以直接配合 `python main.py --config ...` 使用。

如果同时用了 `--config configs/xxx.yaml` 和命令行参数，命令行参数会覆盖 YAML 里的同名设置。例如：

```bash
python main.py --config configs/cifar10_vit.yaml --batch-size 64 --epochs 50
```

这里实际会使用 `batch_size=64`、`epochs=50`。

### 6. 最推荐的实验执行顺序

先 dry-run：

```bash
bash scripts/run_vit_basic.sh
```

dry-run 时先打开对应脚本，把顶部改成：

```bash
EPOCHS=1
BATCH_SIZE=32
NUM_WORKERS=0
DRY_RUN="--dry-run"
```

确认成功后，正式训练前改回：

```bash
EPOCHS=100
BATCH_SIZE=128
NUM_WORKERS=4
DRY_RUN=""
```

正式主实验：

```bash
bash scripts/run_resnet18.sh
bash scripts/run_vit_basic.sh
bash scripts/run_vit_augreg.sh
bash scripts/run_hybrid.sh
```

最后跑消融：

```bash
bash scripts/run_ablation.sh
```

更详细的逐步流程看：

```text
DontReadMe.md
```

---

This project is a unified PyTorch experiment framework for image classification. It compares three model families on CIFAR-10 by default, with a complete Tiny ImageNet loading interface for larger experiments.

## Models

- CNN baseline: CIFAR-friendly ResNet-18, ResNet-34, ResNet-50 via `--cnn-depth`
- Standard Vision Transformer implemented in this repo
- Hybrid CNN+ViT: ResNet feature map tokens followed by Transformer encoder blocks

The ViT implementation includes patch embedding, optional CLS token, position embedding, multi-head self-attention, MLP blocks, Transformer encoder blocks, dropout, attention dropout, and DropPath.

## Datasets

- CIFAR-10: default dataset, can be downloaded with `--download`
- Tiny ImageNet: manual dataset placement under `data/tiny-imagenet-200`

Tiny ImageNet should be arranged as:

```text
data/tiny-imagenet-200/
├── train/
├── val/
├── test/
├── wnids.txt
└── words.txt
```

The validation split is read from `val/val_annotations.txt`. The project uses Tiny ImageNet val as the default evaluation set because the public test split has no labels.

## Augmentation

- RandomCrop: `--random-crop --crop-padding 4`
- RandomHorizontalFlip: `--horizontal-flip --hflip-prob 0.5`
- RandAugment: `--randaugment --ra-num-ops 2 --ra-magnitude 9`
- MixUp: `--mixup-alpha 0.2`
- CutMix: `--cutmix-alpha 1.0`

MixUp and CutMix are batch-level augmentations and are applied only during training.

## Regularization

- Dropout: `--dropout`
- Attention dropout: `--attention-dropout`
- Stochastic depth / DropPath: `--drop-path-rate`
- Label smoothing: `--label-smoothing`
- Weight decay: `--weight-decay`

## Installation

```bash
pip install -r requirements.txt
```

## Quick Dry Runs

```bash
python main.py --dataset cifar10 --model cnn --cnn-depth 18 --download --epochs 1 --dry-run
python main.py --config configs/cifar10_vit.yaml --download --dry-run
```

Required examples:

```bash
python main.py --dataset cifar10 --model cnn --cnn-depth 18 --download --epochs 1 --dry-run
python main.py --config configs/cifar10_vit.yaml --download --dry-run
python main.py --config configs/cifar10_vit_augreg.yaml --download
python main.py --config configs/cifar10_hybrid.yaml --download
```

## Formal Experiments

```bash
bash scripts/run_resnet18.sh
bash scripts/run_vit_basic.sh
bash scripts/run_vit_augreg.sh
bash scripts/run_hybrid.sh
bash scripts/run_ablation.sh
```

## Output Files

Each run creates:

- `outputs/logs/<experiment>/args.json`
- `outputs/logs/<experiment>/metrics.csv`
- `outputs/logs/<experiment>/final_result.json`
- `outputs/logs/<experiment>/train_log.txt`
- `outputs/checkpoints/<experiment>/last.pt`
- `outputs/checkpoints/<experiment>/best.pt`

Use `--resume path/to/checkpoint.pt` to continue training.

## Project Structure

```text
main.py                 # entry point
configs/                # YAML experiment configs
src/datasets/           # CIFAR-10 and Tiny ImageNet loading
src/augmentations/      # RandAugment, MixUp, CutMix
src/models/             # ResNet, ViT, Hybrid ViT
src/engine/             # training and evaluation loops
src/logger.py           # logs, metrics, json saving
scripts/                # runnable experiment scripts
run.sh                  # editable run template for server jobs
report/notes.md         # report template
```

## Suggested Report Experiments

- ResNet-18 vs standard ViT vs Hybrid CNN+ViT
- ViT without augmentation vs ViT with RandAugment, MixUp, CutMix
- Patch size 4 vs 8
- DropPath 0.0 vs 0.1
- Label smoothing and weight decay ablation
