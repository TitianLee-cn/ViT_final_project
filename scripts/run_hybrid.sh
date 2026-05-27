#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

# ============================================================
# Hybrid CNN + ViT experiment.
# Edit variables below before running locally or submitting to a server.
# ============================================================

# Basic
SEED=42
DEVICE="auto"
OUTPUT_DIR="outputs"
EXPERIMENT_NAME="main_hybrid"
DRY_RUN=""
AMP="--amp"
RESUME=""
SAVE=""

# Dataset
DATASET="cifar10"
DATA_DIR="data"
NUM_WORKERS=4
VAL_RATIO=0.1
IMAGE_SIZE=""
DOWNLOAD="--download"

# Model
MODEL="hybrid"
NUM_CLASSES=""
CNN_DEPTH=18
PRETRAINED=""

# ViT part
VIT_DIM=256
VIT_DEPTH=4
VIT_HEADS=8
VIT_MLP_RATIO=4.0
QKV_BIAS="--qkv-bias"
CLS_TOKEN="--cls-token"

# Hybrid part
HYBRID_STAGE="layer3"         # layer2 / layer3 / layer4
HYBRID_FREEZE_CNN=""          # set to "--hybrid-freeze-cnn" to freeze CNN
HYBRID_PATCH_SIZE=1

# Standard ViT patch-size is not used by hybrid, but main.py accepts it.
PATCH_SIZE=4

# Image-level augmentation
RANDOM_CROP="--random-crop"
CROP_PADDING=4
HORIZONTAL_FLIP="--horizontal-flip"
HFLIP_PROB=0.5
RANDAUGMENT=""
RA_NUM_OPS=2
RA_MAGNITUDE=9

# Batch-level augmentation
MIXUP_ALPHA=0.0
CUTMIX_ALPHA=0.0
MIXUP_PROB=1.0
MIXUP_SWITCH_PROB=0.5

# Regularization
DROPOUT=0.1
ATTENTION_DROPOUT=0.1
DROP_PATH_RATE=0.1
LABEL_SMOOTHING=0.1
WEIGHT_DECAY=0.05

# Training
EPOCHS=100
BATCH_SIZE=128
LR=0.0003
OPTIMIZER="adamw"
SCHEDULER="cosine"
WARMUP_EPOCHS=5
MIN_LR=0.000001
MOMENTUM=0.9
LOG_INTERVAL=50
EVAL_INTERVAL=1
GRAD_CLIP=""

CMD=(python main.py
  --seed "$SEED"
  --device "$DEVICE"
  --output-dir "$OUTPUT_DIR"
  --experiment-name "$EXPERIMENT_NAME"
  --dataset "$DATASET"
  --data-dir "$DATA_DIR"
  --num-workers "$NUM_WORKERS"
  --val-ratio "$VAL_RATIO"
  --model "$MODEL"
  --cnn-depth "$CNN_DEPTH"
  --patch-size "$PATCH_SIZE"
  --vit-dim "$VIT_DIM"
  --vit-depth "$VIT_DEPTH"
  --vit-heads "$VIT_HEADS"
  --vit-mlp-ratio "$VIT_MLP_RATIO"
  --hybrid-stage "$HYBRID_STAGE"
  --hybrid-patch-size "$HYBRID_PATCH_SIZE"
  --crop-padding "$CROP_PADDING"
  --hflip-prob "$HFLIP_PROB"
  --ra-num-ops "$RA_NUM_OPS"
  --ra-magnitude "$RA_MAGNITUDE"
  --mixup-alpha "$MIXUP_ALPHA"
  --cutmix-alpha "$CUTMIX_ALPHA"
  --mixup-prob "$MIXUP_PROB"
  --mixup-switch-prob "$MIXUP_SWITCH_PROB"
  --dropout "$DROPOUT"
  --attention-dropout "$ATTENTION_DROPOUT"
  --drop-path-rate "$DROP_PATH_RATE"
  --label-smoothing "$LABEL_SMOOTHING"
  --weight-decay "$WEIGHT_DECAY"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --lr "$LR"
  --optimizer "$OPTIMIZER"
  --scheduler "$SCHEDULER"
  --warmup-epochs "$WARMUP_EPOCHS"
  --min-lr "$MIN_LR"
  --momentum "$MOMENTUM"
  --log-interval "$LOG_INTERVAL"
  --eval-interval "$EVAL_INTERVAL"
)

CMD+=($IMAGE_SIZE $NUM_CLASSES $PRETRAINED $QKV_BIAS $CLS_TOKEN $HYBRID_FREEZE_CNN)
CMD+=($RANDOM_CROP $HORIZONTAL_FLIP $RANDAUGMENT)
CMD+=($DOWNLOAD $DRY_RUN $AMP $RESUME $SAVE $GRAD_CLIP)

echo "Running command:"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"
