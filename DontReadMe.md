# ViT 图像分类大作业实验流程

本实验流程用于完成 Vision Transformer 图像分类大作业。

核心任务是：在 CIFAR-10 上比较 CNN baseline、Basic ViT、AugReg ViT 和 Hybrid CNN+ViT 的性能，并通过消融实验分析数据增强、正则化和结构设计的影响。

项目路径：

    cd /mnt/e/AI4Titi/0.8_DeepLearning/Final_Project/ViT_model

---

# 0. 实验总目标

本项目围绕以下问题展开：

1. CNN baseline，也就是 ResNet-18，在 CIFAR-10 上能达到怎样的性能？
2. 标准 Vision Transformer 在 CIFAR-10 这种小数据集上表现如何？
3. 数据增强和正则化是否能提升 ViT 的性能？
4. Hybrid CNN+ViT 是否能结合 CNN 的局部特征提取能力和 ViT 的全局建模能力？
5. patch size、DropPath、RandAugment、MixUp/CutMix 等设计对模型性能有什么影响？

---

# 1. 实验前检查

## 1.1 检查项目结构

目的：确认 Codex 已经生成完整代码文件和目录。

运行：

    find /mnt/e/AI4Titi/0.8_DeepLearning/Final_Project/ViT_model -maxdepth 3 -type f | sort

重点检查是否存在：

    main.py
    src/arguments.py
    src/datasets/
    src/models/
    src/engine/
    src/losses.py
    src/optim.py
    src/metrics.py
    configs/
    scripts/
    outputs/

如果这些文件和目录都存在，说明项目框架已经完整。

---

## 1.2 检查 Python 文件是否能编译

目的：确认代码没有语法错误。

运行：

    python -m py_compile main.py
    python -m py_compile src/*.py
    python -m py_compile src/datasets/*.py
    python -m py_compile src/augmentations/*.py
    python -m py_compile src/models/*.py
    python -m py_compile src/engine/*.py

成功标准：

    没有报错输出。

如果出现 SyntaxError，说明代码存在语法错误，需要先修复再继续实验。

---

## 1.3 查看命令行参数

目的：确认所有可调参数都已经暴露出来。

运行：

    python main.py --help

重点检查这些参数是否存在：

    --dataset
    --model
    --cnn-depth
    --patch-size
    --vit-depth
    --vit-heads
    --vit-dim
    --random-crop
    --horizontal-flip
    --randaugment
    --mixup-alpha
    --cutmix-alpha
    --dropout
    --attention-dropout
    --drop-path-rate
    --label-smoothing
    --weight-decay
    --amp
    --dry-run
    --resume

这一步测试的是：项目是否真的支持“通过命令行切换数据集、模型、增强方法、正则化方法和训练参数”。

---

# 2. Dry-run 快速检查

Dry-run 的目的不是训练出高准确率，而是确认完整流程能跑通。

它主要检查：

1. 数据集加载；
2. 模型构建；
3. forward；
4. loss 计算；
5. backward；
6. validation；
7. checkpoint 保存；
8. metrics.csv 和 final_result.json 保存。

所有正式实验开始前，必须先跑 dry-run。

---

## 2.1 CNN baseline dry-run

测试内容：检查 ResNet-18 baseline 是否能正常训练和验证。

运行：

    python main.py \
      --dataset cifar10 \
      --model cnn \
      --cnn-depth 18 \
      --download \
      --epochs 1 \
      --batch-size 32 \
      --num-workers 0 \
      --dry-run

这一步测试的是：

    CIFAR-10 数据集读取
    CNN baseline 构建
    ResNet-18 forward
    CrossEntropy loss
    train / val / test 流程
    checkpoint 保存
    日志保存

成功标准：

    1. 不报错；
    2. 能看到 train loss / train acc；
    3. 能看到 val loss / val acc1 / val acc5；
    4. outputs/logs/ 下生成对应实验目录；
    5. outputs/checkpoints/ 下生成 last.pt 和 best.pt。

---

## 2.2 Basic ViT dry-run

测试内容：检查手写 ViT 是否能正常工作。

运行：

    python main.py \
      --dataset cifar10 \
      --model vit \
      --patch-size 4 \
      --vit-depth 2 \
      --vit-heads 4 \
      --vit-dim 128 \
      --download \
      --epochs 1 \
      --batch-size 32 \
      --num-workers 0 \
      --dry-run

这一步测试的是：

    图像切 patch
    Patch Embedding
    CLS token
    Position Embedding
    Transformer Encoder
    Multi-Head Self-Attention
    MLP
    Classification Head

成功标准：

    能完整跑完 train / val / test，且没有维度错误。

---

## 2.3 AugReg ViT dry-run

测试内容：检查数据增强和正则化是否能正常接入 ViT。

运行：

    python main.py \
      --dataset cifar10 \
      --model vit \
      --patch-size 4 \
      --vit-depth 2 \
      --vit-heads 4 \
      --vit-dim 128 \
      --random-crop \
      --horizontal-flip \
      --randaugment \
      --mixup-alpha 0.2 \
      --cutmix-alpha 1.0 \
      --dropout 0.1 \
      --attention-dropout 0.1 \
      --drop-path-rate 0.1 \
      --label-smoothing 0.1 \
      --weight-decay 0.05 \
      --download \
      --epochs 1 \
      --batch-size 32 \
      --num-workers 0 \
      --dry-run

这一步测试的是：

    RandomCrop
    RandomHorizontalFlip
    RandAugment
    MixUp
    CutMix
    SoftTargetCrossEntropy
    Dropout
    Attention Dropout
    DropPath / Stochastic Depth
    Label Smoothing
    Weight Decay

成功标准：

    1. MixUp / CutMix 开启后 loss 能正常计算；
    2. 不出现 label shape 错误；
    3. 训练、验证、测试流程完整跑通。

注意：

    MixUp / CutMix 会把 hard label 变成 soft label。
    因此这一步尤其能检查 SoftTargetCrossEntropy 是否正确。

---

## 2.4 Hybrid CNN+ViT dry-run

测试内容：检查 CNN backbone + ViT 的混合模型是否能正常运行。

运行：

    python main.py \
      --dataset cifar10 \
      --model hybrid \
      --cnn-depth 18 \
      --hybrid-stage layer3 \
      --vit-depth 2 \
      --vit-heads 4 \
      --vit-dim 128 \
      --download \
      --epochs 1 \
      --batch-size 32 \
      --num-workers 0 \
      --dry-run

这一步测试的是：

    ResNet feature extractor
    forward_features
    CNN feature map 转 token sequence
    Hybrid patch embedding
    Transformer Encoder
    Hybrid classification head

成功标准：

    能完整跑完，说明 Hybrid 模型结构没有维度错误。

---

# 3. 正式实验设计

正式实验分成四组主实验和若干组消融实验。

主实验：

    主实验 1：ResNet-18 baseline
    主实验 2：Basic ViT
    主实验 3：AugReg ViT
    主实验 4：Hybrid CNN+ViT

消融实验：

    消融实验 A：Patch Size
    消融实验 B：DropPath
    消融实验 C：RandAugment
    消融实验 D：MixUp / CutMix
    消融实验 E：Hybrid Stage

建议先完成四个主实验，再做消融实验。

---

# 4. 主实验 1：CNN Baseline

## 4.1 实验目的

测试传统 CNN 模型在 CIFAR-10 上的表现，作为后续 ViT 和 Hybrid 模型的对照组。

这个实验回答的问题是：

    ResNet-18 作为 CNN baseline，在相同数据集和训练框架下能达到什么准确率？

---

## 4.2 推荐运行命令

优先使用脚本：

    bash scripts/run_resnet18.sh

或者手动运行：

    python main.py \
      --config configs/cifar10_resnet18.yaml \
      --download \
      --experiment-name cifar10_resnet18_baseline

---

## 4.3 建议参数

    dataset: cifar10
    model: cnn
    cnn_depth: 18
    random_crop: true
    horizontal_flip: true
    optimizer: sgd
    scheduler: cosine
    weight_decay: 0.0005
    epochs: 100
    batch_size: 128

---

## 4.4 这一步在测试什么

这一组实验主要测试：

    CNN baseline 的稳定性能；
    CIFAR-friendly ResNet-18 是否适合 CIFAR-10；
    后续 ViT / Hybrid 是否能接近或超过 CNN baseline。

ResNet 的优势在于：

    卷积结构天然适合图像；
    局部感受野能捕捉边缘、纹理等局部模式；
    参数共享带来更强的归纳偏置；
    在 CIFAR-10 这种小数据集上通常比较稳定。

---

## 4.5 记录内容

训练结束后记录：

    outputs/logs/cifar10_resnet18_baseline/final_result.json
    outputs/logs/cifar10_resnet18_baseline/metrics.csv
    outputs/checkpoints/cifar10_resnet18_baseline/best.pt

需要整理到报告中的指标：

| Model | Test Loss | Test Top-1 Acc | Test Top-5 Acc | Params |
|---|---:|---:|---:|---:|
| ResNet-18 |  |  |  |  |

---

# 5. 主实验 2：Basic ViT

## 5.1 实验目的

测试标准 Vision Transformer 在 CIFAR-10 上的基础表现。

这个实验回答的问题是：

    不使用强数据增强和复杂正则化时，纯 ViT 在小数据集 CIFAR-10 上表现如何？

---

## 5.2 推荐运行命令

优先使用脚本：

    bash scripts/run_vit_basic.sh

或者手动运行：

    python main.py \
      --config configs/cifar10_vit.yaml \
      --download \
      --experiment-name cifar10_vit_basic

---

## 5.3 建议参数

    dataset: cifar10
    model: vit
    patch_size: 4
    vit_dim: 256
    vit_depth: 6
    vit_heads: 8
    vit_mlp_ratio: 4.0
    dropout: 0.1
    attention_dropout: 0.1
    drop_path_rate: 0.0
    label_smoothing: 0.0
    weight_decay: 0.05
    optimizer: adamw
    scheduler: cosine
    epochs: 100
    batch_size: 128

---

## 5.4 这一步在测试什么

这一组实验主要测试：

    标准 ViT 结构本身是否能完成 CIFAR-10 图像分类；
    patch embedding + self-attention 是否能学习图像特征；
    不引入强增强时，ViT 是否容易过拟合；
    ViT 与 ResNet-18 的性能差距。

Basic ViT 的核心流程是：

    Image
    -> Patch Embedding
    -> Add CLS Token
    -> Add Position Embedding
    -> Transformer Encoder
    -> Classification Head

---

## 5.5 重点观察

对比 ResNet-18：

    1. ViT 的收敛速度是否更慢？
    2. ViT 的 train acc 和 val acc 差距是否更大？
    3. ViT 是否更容易过拟合？
    4. Basic ViT 的最终 test acc 是否低于 ResNet-18？

---

## 5.6 记录表格

| Model | Test Loss | Test Top-1 Acc | Test Top-5 Acc | Params |
|---|---:|---:|---:|---:|
| ResNet-18 |  |  |  |  |
| Basic ViT |  |  |  |  |

---

# 6. 主实验 3：AugReg ViT

## 6.1 实验目的

测试数据增强和正则化是否能提升 ViT。

这个实验回答的问题是：

    MixUp、CutMix、RandAugment、DropPath、Label Smoothing 和 Weight Decay 是否能改善 ViT 在 CIFAR-10 上的泛化性能？

---

## 6.2 推荐运行命令

优先使用脚本：

    bash scripts/run_vit_augreg.sh

或者手动运行：

    python main.py \
      --config configs/cifar10_vit_augreg.yaml \
      --download \
      --experiment-name cifar10_vit_augreg

---

## 6.3 建议参数

    dataset: cifar10
    model: vit
    patch_size: 4
    vit_dim: 256
    vit_depth: 6
    vit_heads: 8
    random_crop: true
    horizontal_flip: true
    randaugment: true
    ra_num_ops: 2
    ra_magnitude: 9
    mixup_alpha: 0.2
    cutmix_alpha: 1.0
    dropout: 0.1
    attention_dropout: 0.1
    drop_path_rate: 0.1
    label_smoothing: 0.1
    weight_decay: 0.05
    optimizer: adamw
    scheduler: cosine
    epochs: 100
    batch_size: 128

---

## 6.4 这一步在测试什么

这一组实验主要测试：

    数据增强是否能提升 ViT 泛化能力；
    正则化是否能缓解 ViT 在小数据集上的过拟合；
    AugReg 思路是否比 Basic ViT 更适合 CIFAR-10。

具体模块作用：

    RandomCrop:
        让模型看到不同裁剪位置，提高位置鲁棒性。

    RandomHorizontalFlip:
        让模型学习左右翻转不变性。

    RandAugment:
        自动组合多种图像增强，提高数据多样性。

    MixUp:
        把两张图片线性混合，同时混合标签，让分类边界更平滑。

    CutMix:
        把一张图片的局部区域替换为另一张图片，同时按区域比例混合标签。

    DropPath:
        随机丢弃残差分支，减少模型对某些路径的依赖。

    Label Smoothing:
        避免模型对单一类别过度自信。

    Weight Decay:
        限制参数过大，改善泛化能力。

---

## 6.5 重点观察

重点对比 Basic ViT：

    1. AugReg ViT 的 val acc 是否高于 Basic ViT？
    2. train acc 是否可能下降？
    3. val acc 和 test acc 是否更稳定？
    4. loss 曲线是否更平滑？
    5. 过拟合是否减轻？

注意：

    MixUp / CutMix 会让训练阶段的 train accuracy 变得不那么直观。
    因此 AugReg ViT 更应该看 val acc 和 test acc，而不是只看 train acc。

---

## 6.6 记录表格

| Model | Augmentation | Regularization | Test Top-1 Acc | Test Top-5 Acc |
|---|---|---|---:|---:|
| Basic ViT | weak / none | weak |  |  |
| AugReg ViT | RandAug + MixUp + CutMix | DropPath + LS + WD |  |  |

---

# 7. 主实验 4：Hybrid CNN+ViT

## 7.1 实验目的

测试 CNN 局部特征提取和 ViT 全局建模结合后，是否能得到更好的性能。

这个实验回答的问题是：

    Hybrid CNN+ViT 是否比纯 ViT 更适合 CIFAR-10 这种小规模图像分类任务？

---

## 7.2 推荐运行命令

优先使用脚本：

    bash scripts/run_hybrid.sh

或者手动运行：

    python main.py \
      --config configs/cifar10_hybrid.yaml \
      --download \
      --experiment-name cifar10_hybrid_resnet18_layer3

---

## 7.3 建议参数

    dataset: cifar10
    model: hybrid
    cnn_depth: 18
    hybrid_stage: layer3
    hybrid_freeze_cnn: false
    hybrid_patch_size: 1
    vit_dim: 256
    vit_depth: 4
    vit_heads: 8
    vit_mlp_ratio: 4.0
    random_crop: true
    horizontal_flip: true
    dropout: 0.1
    attention_dropout: 0.1
    drop_path_rate: 0.1
    label_smoothing: 0.1
    weight_decay: 0.05
    optimizer: adamw
    scheduler: cosine
    epochs: 100
    batch_size: 128

---

## 7.4 这一步在测试什么

这一组实验主要测试：

    CNN 的局部归纳偏置是否能帮助 ViT；
    ResNet feature map 转 token 后是否适合 Transformer 处理；
    Hybrid 是否比 Basic ViT 更稳定；
    Hybrid 是否能接近或超过 ResNet-18。

Hybrid 模型流程：

    Image
    -> ResNet Backbone
    -> Feature Map
    -> Token Embedding
    -> Transformer Encoder
    -> Classification Head

---

## 7.5 重点观察

重点比较：

    ResNet-18 vs Basic ViT vs AugReg ViT vs Hybrid CNN+ViT

分析问题：

    1. Hybrid 是否比 Basic ViT 更稳定？
    2. Hybrid 是否比 AugReg ViT 更好？
    3. Hybrid 是否接近或超过 ResNet-18？
    4. CNN feature map 作为 token 是否减少了 ViT 对大量数据的依赖？

---

## 7.6 记录表格

| Model | Structure | Test Top-1 Acc | Test Top-5 Acc | Params |
|---|---|---:|---:|---:|
| ResNet-18 | CNN |  |  |  |
| Basic ViT | Pure Transformer |  |  |  |
| AugReg ViT | Pure Transformer + AugReg |  |  |  |
| Hybrid CNN+ViT | CNN + Transformer |  |  |  |

---

# 8. 消融实验

消融实验的目标是解释：

    为什么最终模型有效？
    哪些模块真的带来了提升？
    哪些超参数影响比较明显？

消融实验不一定全部都要跑完。建议至少跑 3 组。

---

# 8.1 消融实验 A：Patch Size

## 实验目的

测试 patch size 对 ViT 性能的影响。

这个实验回答：

    图像切成更小 patch 是否能带来更好的细节建模？

---

## 运行命令

Patch size = 4：

    python main.py \
      --config configs/cifar10_vit.yaml \
      --patch-size 4 \
      --experiment-name ablation_vit_patch4 \
      --download

Patch size = 8：

    python main.py \
      --config configs/cifar10_vit.yaml \
      --patch-size 8 \
      --experiment-name ablation_vit_patch8 \
      --download

---

## 这一步在测试什么

    patch_size = 4:
        CIFAR-10 的 32x32 图像会被切成 8x8 = 64 个 patch。
        token 数更多，细节保留更多，但计算量更大。

    patch_size = 8:
        CIFAR-10 的 32x32 图像会被切成 4x4 = 16 个 patch。
        token 数更少，训练更快，但可能损失局部细节。

---

## 记录表格

| Patch Size | Token Number on 32x32 Image | Test Top-1 Acc | Training Cost | Observation |
|---:|---:|---:|---|---|
| 4 | 64 |  |  |  |
| 8 | 16 |  |  |  |

---

# 8.2 消融实验 B：DropPath

## 实验目的

测试 Stochastic Depth / DropPath 是否能提升 ViT 泛化能力。

这个实验回答：

    随机丢弃残差分支是否能减少过拟合？

---

## 运行命令

DropPath = 0：

    python main.py \
      --config configs/cifar10_vit_augreg.yaml \
      --drop-path-rate 0.0 \
      --experiment-name ablation_droppath0 \
      --download

DropPath = 0.1：

    python main.py \
      --config configs/cifar10_vit_augreg.yaml \
      --drop-path-rate 0.1 \
      --experiment-name ablation_droppath01 \
      --download

DropPath = 0.2：

    python main.py \
      --config configs/cifar10_vit_augreg.yaml \
      --drop-path-rate 0.2 \
      --experiment-name ablation_droppath02 \
      --download

---

## 这一步在测试什么

    drop_path_rate = 0.0:
        没有 Stochastic Depth，模型可能更容易过拟合。

    drop_path_rate = 0.1:
        常用且比较稳妥的正则化强度。

    drop_path_rate = 0.2:
        正则化更强，但过大可能导致欠拟合。

---

## 记录表格

| DropPath Rate | Test Top-1 Acc | Train-Val Gap | Observation |
|---:|---:|---:|---|
| 0.0 |  |  |  |
| 0.1 |  |  |  |
| 0.2 |  |  |  |

---

# 8.3 消融实验 C：RandAugment

## 实验目的

测试 RandAugment 是否能提升 ViT 的泛化能力。

这个实验回答：

    更强的图像级数据增强是否能弥补 ViT 在小数据集上的数据不足问题？

---

## 运行命令

不使用 RandAugment：

    python main.py \
      --config configs/cifar10_vit_augreg.yaml \
      --randaugment false \
      --experiment-name ablation_no_randaugment \
      --download

如果命令行不支持 --randaugment false，则直接复制 configs/cifar10_vit_augreg.yaml，新建一个配置文件，把 randaugment 改成 false，然后运行：

    python main.py \
      --config configs/cifar10_vit_augreg_no_ra.yaml \
      --experiment-name ablation_no_randaugment \
      --download

使用 RandAugment：

    python main.py \
      --config configs/cifar10_vit_augreg.yaml \
      --randaugment \
      --experiment-name ablation_with_randaugment \
      --download

---

## 这一步在测试什么

    不使用 RandAugment:
        数据变化较少，模型更容易记住训练集。

    使用 RandAugment:
        输入图像扰动更多，可能提升 val/test accuracy。

---

## 记录表格

| RandAugment | Test Top-1 Acc | Test Top-5 Acc | Observation |
|---|---:|---:|---|
| Off |  |  |  |
| On |  |  |  |

---

# 8.4 消融实验 D：MixUp / CutMix

## 实验目的

测试 batch-level 数据增强对 ViT 的影响。

这个实验回答：

    混合样本和混合标签是否能让 ViT 学到更平滑的分类边界？

---

## 运行命令

不使用 MixUp / CutMix：

    python main.py \
      --config configs/cifar10_vit_augreg.yaml \
      --mixup-alpha 0.0 \
      --cutmix-alpha 0.0 \
      --experiment-name ablation_no_mixup_cutmix \
      --download

只使用 MixUp：

    python main.py \
      --config configs/cifar10_vit_augreg.yaml \
      --mixup-alpha 0.2 \
      --cutmix-alpha 0.0 \
      --experiment-name ablation_mixup_only \
      --download

只使用 CutMix：

    python main.py \
      --config configs/cifar10_vit_augreg.yaml \
      --mixup-alpha 0.0 \
      --cutmix-alpha 1.0 \
      --experiment-name ablation_cutmix_only \
      --download

同时使用 MixUp 和 CutMix：

    python main.py \
      --config configs/cifar10_vit_augreg.yaml \
      --mixup-alpha 0.2 \
      --cutmix-alpha 1.0 \
      --experiment-name ablation_mixup_cutmix \
      --download

---

## 这一步在测试什么

    MixUp:
        图像整体线性混合，鼓励模型学习平滑决策边界。

    CutMix:
        局部区域替换，更符合图像局部语义特点。

    MixUp + CutMix:
        可能进一步提升泛化，但训练 accuracy 解释性会下降。

---

## 记录表格

| MixUp Alpha | CutMix Alpha | Test Top-1 Acc | Observation |
|---:|---:|---:|---|
| 0.0 | 0.0 |  |  |
| 0.2 | 0.0 |  |  |
| 0.0 | 1.0 |  |  |
| 0.2 | 1.0 |  |  |

---

# 8.5 消融实验 E：Hybrid Stage

## 实验目的

测试 Hybrid 模型中使用 ResNet 哪一层 feature map 更合适。

这个实验回答：

    Transformer 应该接收更浅层、更高分辨率的特征，还是更深层、更语义化的特征？

---

## 运行命令

layer2：

    python main.py \
      --config configs/cifar10_hybrid.yaml \
      --hybrid-stage layer2 \
      --experiment-name ablation_hybrid_layer2 \
      --download

layer3：

    python main.py \
      --config configs/cifar10_hybrid.yaml \
      --hybrid-stage layer3 \
      --experiment-name ablation_hybrid_layer3 \
      --download

layer4：

    python main.py \
      --config configs/cifar10_hybrid.yaml \
      --hybrid-stage layer4 \
      --experiment-name ablation_hybrid_layer4 \
      --download

---

## 这一步在测试什么

    layer2:
        feature map 分辨率较高，token 数更多，保留更多局部细节，但计算量更大。

    layer3:
        折中方案，通常比较稳。

    layer4:
        feature map 分辨率较低，语义更强，但 token 太少，可能限制 Transformer 的建模能力。

---

## 记录表格

| Hybrid Stage | Feature Level | Test Top-1 Acc | Observation |
|---|---|---:|---|
| layer2 | shallow / high resolution |  |  |
| layer3 | middle |  |  |
| layer4 | deep / low resolution |  |  |

---

# 9. 可选实验：Tiny ImageNet 接口测试

## 9.1 实验目的

这个实验不是主线，只用于说明项目支持 Tiny ImageNet 接口。

如果没有 Tiny ImageNet 数据，可以不跑。

Tiny ImageNet 数据需要放在：

    data/tiny-imagenet-200/
    ├── train/
    ├── val/
    ├── test/
    ├── wnids.txt
    └── words.txt

---

## 9.2 运行命令

    python main.py \
      --config configs/tinyimagenet_vit.yaml \
      --experiment-name tinyimagenet_vit_test

---

## 9.3 这一步测试什么

    1. dataset 参数是否能从 cifar10 切换到 tiny-imagenet；
    2. Tiny ImageNet train folder 是否能读取；
    3. val_annotations.txt 是否能正确映射验证集标签；
    4. num_classes 是否变为 200；
    5. image_size 是否变为 64；
    6. top-5 accuracy 是否更有实际意义。

---

# 10. 实验结果整理方式

每个实验结束后，查看：

    ls outputs/logs/

进入对应实验目录：

    ls outputs/logs/<experiment_name>/

重点文件：

    args.json
    metrics.csv
    final_result.json
    train_log.txt

checkpoint 文件：

    outputs/checkpoints/<experiment_name>/best.pt
    outputs/checkpoints/<experiment_name>/last.pt

---

# 11. 训练曲线应该看什么

打开 metrics.csv，重点关注：

    epoch
    train_loss
    train_acc1
    val_loss
    val_acc1
    val_acc5
    lr

分析时重点看：

    1. train_loss 是否下降；
    2. val_loss 是否下降；
    3. train_acc1 和 val_acc1 差距是否过大；
    4. val_acc1 是否在后期震荡；
    5. AugReg 是否减轻过拟合；
    6. Hybrid 是否比纯 ViT 更稳定。

---

# 12. 最终主结果表

报告中建议放这张主表：

| Model | Dataset | Main Setting | Test Top-1 Acc | Test Top-5 Acc | Params | Comment |
|---|---|---|---:|---:|---:|---|
| ResNet-18 | CIFAR-10 | CNN baseline |  |  |  | baseline |
| Basic ViT | CIFAR-10 | patch=4, depth=6 |  |  |  | pure ViT |
| AugReg ViT | CIFAR-10 | MixUp/CutMix/RandAug + regularization |  |  |  | bonus |
| Hybrid CNN+ViT | CIFAR-10 | ResNet-18 layer3 + ViT |  |  |  | structure improvement |

---

# 13. 最终消融实验表

报告中建议放这张消融表：

| Experiment | Variant | Test Top-1 Acc | Conclusion |
|---|---|---:|---|
| Patch Size | 4 |  |  |
| Patch Size | 8 |  |  |
| DropPath | 0.0 |  |  |
| DropPath | 0.1 |  |  |
| RandAugment | Off |  |  |
| RandAugment | On |  |  |
| MixUp/CutMix | Off |  |  |
| MixUp/CutMix | On |  |  |
| Hybrid Stage | layer2 |  |  |
| Hybrid Stage | layer3 |  |  |
| Hybrid Stage | layer4 |  |  |

---

# 14. 推荐报告分析逻辑

## 14.1 CNN baseline 分析

    ResNet-18 作为 CNN baseline，在 CIFAR-10 上通常表现稳定。
    它具有卷积结构带来的局部归纳偏置，因此在小规模图像数据集上容易训练。

## 14.2 Basic ViT 分析

    Basic ViT 直接将图像切成 patch，并使用 self-attention 建模 patch 之间的关系。
    但是 ViT 缺少 CNN 的局部归纳偏置，因此在 CIFAR-10 这种小数据集上可能不如 ResNet 稳定。

## 14.3 AugReg ViT 分析

    AugReg ViT 通过 RandAugment、MixUp、CutMix、DropPath、Label Smoothing 和 Weight Decay 增强泛化能力。
    如果实验结果显示 AugReg ViT 高于 Basic ViT，可以说明数据增强和正则化对 ViT 尤其重要。

## 14.4 Hybrid CNN+ViT 分析

    Hybrid CNN+ViT 先用 CNN 提取局部特征，再用 Transformer 进行全局建模。
    如果它优于 Basic ViT，说明 CNN 的局部特征提取能力可以弥补 ViT 在小数据集上的不足。

## 14.5 消融实验分析

    Patch size 控制 token 粒度；
    DropPath 控制残差分支随机丢弃强度；
    RandAugment 控制图像级增强；
    MixUp/CutMix 控制 batch-level 增强；
    Hybrid stage 控制 CNN 特征的深浅。

---

# 15. 建议实验优先级

如果时间充足，按完整流程跑：

    1. ResNet-18 baseline
    2. Basic ViT
    3. AugReg ViT
    4. Hybrid CNN+ViT
    5. Patch size ablation
    6. DropPath ablation
    7. RandAugment ablation
    8. MixUp/CutMix ablation
    9. Hybrid stage ablation

如果时间有限，至少跑：

    1. ResNet-18 baseline
    2. Basic ViT
    3. AugReg ViT
    4. Hybrid CNN+ViT

如果时间非常紧，最小可交付实验是：

    1. ResNet-18 baseline
    2. Basic ViT
    3. AugReg ViT

这三组已经可以满足：

    1. ViT 图像分类；
    2. CNN baseline 对比；
    3. Bonus 数据增强和正则化改进。

---

# 16. 推荐最终结论写法模板

实验完成后，可以根据结果选择下面的结论表述。

如果 AugReg ViT 提升明显：

    Compared with Basic ViT, AugReg ViT achieves higher validation and test accuracy, showing that data augmentation and regularization are crucial for training Vision Transformers on relatively small datasets such as CIFAR-10.

如果 Hybrid 提升明显：

    The Hybrid CNN+ViT model outperforms the pure ViT baseline, suggesting that CNN-based local feature extraction can provide useful inductive bias before global self-attention modeling.

如果 ResNet 仍然最好：

    Although ViT provides a flexible global modeling framework, ResNet-18 remains highly competitive on CIFAR-10 because convolutional networks naturally encode locality and translation equivariance, which are especially useful for small-scale image datasets.

如果 ViT 训练不稳定：

    The Basic ViT model is more sensitive to training settings than ResNet-18. This indicates that Vision Transformers usually require stronger augmentation, regularization, or larger-scale data to fully realize their potential.

---

# 17. 实验完成检查清单

正式写报告前，确认已经完成：

    [ ] CNN baseline dry-run 通过
    [ ] Basic ViT dry-run 通过
    [ ] AugReg ViT dry-run 通过
    [ ] Hybrid dry-run 通过

    [ ] ResNet-18 正式实验完成
    [ ] Basic ViT 正式实验完成
    [ ] AugReg ViT 正式实验完成
    [ ] Hybrid CNN+ViT 正式实验完成

    [ ] 每个实验都有 final_result.json
    [ ] 每个实验都有 metrics.csv
    [ ] 每个实验都有 best.pt
    [ ] 已整理主结果表
    [ ] 已整理至少 3 组消融实验
    [ ] 已保存训练曲线或结果截图
    [ ] 已总结 ResNet / ViT / AugReg / Hybrid 的对比结论
```