#!/usr/bin/env bash
set -e

# Run from the project root no matter where this script is launched.
cd "$(dirname "$0")"

# ============================================================
# Edit these variables before running or submitting to a server.
# ============================================================

DATASET="cifar10"
MODEL="vit"                  # cnn / vit / hybrid
CONFIG=""                    # optional, e.g. configs/cifar10_vit_augreg.yaml
EXPERIMENT_NAME=""           # optional, e.g. my_vit_run_001

EPOCHS=100
BATCH_SIZE=128
NUM_WORKERS=4
LR=0.0003
DEVICE="auto"                # auto / cpu / cuda

# CNN options
CNN_DEPTH=18                 # 18 / 34 / 50

# ViT options
PATCH_SIZE=4
VIT_DIM=256
VIT_DEPTH=6
VIT_HEADS=8
VIT_MLP_RATIO=4.0

# Hybrid options
HYBRID_STAGE="layer3"        # layer2 / layer3 / layer4
HYBRID_PATCH_SIZE=1
HYBRID_FREEZE_CNN=""         # set to "--hybrid-freeze-cnn" to enable

# Augmentation and regularization
RANDOM_CROP="--random-crop"          # set to "" to disable
HORIZONTAL_FLIP="--horizontal-flip"  # set to "" to disable
RANDAUGMENT=""                      # set to "--randaugment" to enable
MIXUP_ALPHA=0.0
CUTMIX_ALPHA=0.0
DROPOUT=0.1
ATTENTION_DROPOUT=0.1
DROP_PATH_RATE=0.0
LABEL_SMOOTHING=0.0
WEIGHT_DECAY=0.05

# Training control
OPTIMIZER="adamw"             # adamw / sgd
SCHEDULER="cosine"            # cosine / step / none
AMP="--amp"                   # set to "" to disable
DOWNLOAD="--download"         # CIFAR-10 only; set to "" if data already exists
DRY_RUN=""                    # set to "--dry-run" for a quick test
RESUME=""                     # e.g. "--resume outputs/checkpoints/xxx/last.pt"

# ============================================================
# Build command.
# If CONFIG is set, values in CONFIG are used first, then the
# explicit command-line options below override them.
# ============================================================

CMD=(python main.py)

if [ -n "$CONFIG" ]; then
  CMD+=(--config "$CONFIG")
else
  CMD+=(--dataset "$DATASET" --model "$MODEL")
fi

if [ -n "$EXPERIMENT_NAME" ]; then
  CMD+=(--experiment-name "$EXPERIMENT_NAME")
fi

CMD+=(
  --device "$DEVICE"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --lr "$LR"
  --optimizer "$OPTIMIZER"
  --scheduler "$SCHEDULER"
  --cnn-depth "$CNN_DEPTH"
  --patch-size "$PATCH_SIZE"
  --vit-dim "$VIT_DIM"
  --vit-depth "$VIT_DEPTH"
  --vit-heads "$VIT_HEADS"
  --vit-mlp-ratio "$VIT_MLP_RATIO"
  --hybrid-stage "$HYBRID_STAGE"
  --hybrid-patch-size "$HYBRID_PATCH_SIZE"
  --mixup-alpha "$MIXUP_ALPHA"
  --cutmix-alpha "$CUTMIX_ALPHA"
  --dropout "$DROPOUT"
  --attention-dropout "$ATTENTION_DROPOUT"
  --drop-path-rate "$DROP_PATH_RATE"
  --label-smoothing "$LABEL_SMOOTHING"
  --weight-decay "$WEIGHT_DECAY"
)

# Optional flags. Keep these unquoted so empty strings disappear.
CMD+=(
  $RANDOM_CROP
  $HORIZONTAL_FLIP
  $RANDAUGMENT
  $HYBRID_FREEZE_CNN
  $AMP
  $DOWNLOAD
  $DRY_RUN
  $RESUME
)

echo "Running command:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"

# ============================================================
# Common presets:
#
# 1. ResNet-18 baseline:
#    CONFIG="configs/cifar10_resnet18.yaml"
#
# 2. Basic ViT:
#    CONFIG="configs/cifar10_vit.yaml"
#
# 3. AugReg ViT:
#    CONFIG="configs/cifar10_vit_augreg.yaml"
#
# 4. Hybrid:
#    CONFIG="configs/cifar10_hybrid.yaml"
#
# 5. Quick dry-run:
#    EPOCHS=1
#    BATCH_SIZE=32
#    NUM_WORKERS=0
#    DRY_RUN="--dry-run"
# ============================================================
