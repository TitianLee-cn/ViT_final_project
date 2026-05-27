#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

# ============================================================
# ResNet baseline experiment.
# Edit variables below before running locally or submitting to a server.
# ============================================================

# Basic
SEED=42
DEVICE="auto"                 # auto / cpu / cuda
OUTPUT_DIR="outputs"
EXPERIMENT_NAME="main_resnet18"
DRY_RUN=""                    # set to "--dry-run" to test quickly
AMP=""                        # set to "--amp" to enable mixed precision
RESUME=""                     # e.g. "--resume outputs/checkpoints/main_resnet18/last.pt"
SAVE=""                       # e.g. "--save outputs/checkpoints/custom_dir"

# Dataset
DATASET="cifar10"             # cifar10 / tiny-imagenet
DATA_DIR="data"
NUM_WORKERS=4
VAL_RATIO=0.1
IMAGE_SIZE=""                 # empty means dataset default; or set to "--image-size 32"
DOWNLOAD="--download"         # CIFAR-10 only; set to "" if data already exists

# Model
MODEL="cnn"
NUM_CLASSES=""                # empty means auto; or set to "--num-classes 10"
CNN_DEPTH=18                  # 18 / 34 / 50
PRETRAINED=""                 # kept for interface; set to "--pretrained" if needed

# Image-level augmentation
RANDOM_CROP="--random-crop"
CROP_PADDING=4
HORIZONTAL_FLIP="--horizontal-flip"
HFLIP_PROB=0.5
RANDAUGMENT=""                # set to "--randaugment" to enable
RA_NUM_OPS=2
RA_MAGNITUDE=9

# Batch-level augmentation
MIXUP_ALPHA=0.0
CUTMIX_ALPHA=0.0
MIXUP_PROB=1.0
MIXUP_SWITCH_PROB=0.5

# Regularization
DROPOUT=0.0
ATTENTION_DROPOUT=0.0
DROP_PATH_RATE=0.0
LABEL_SMOOTHING=0.0
WEIGHT_DECAY=0.0005

# Training
EPOCHS=100
BATCH_SIZE=128
LR=0.1
OPTIMIZER="sgd"               # adamw / sgd
SCHEDULER="cosine"            # cosine / step / none
WARMUP_EPOCHS=5
MIN_LR=0.000001
MOMENTUM=0.9
LOG_INTERVAL=50
EVAL_INTERVAL=1
GRAD_CLIP=""                  # e.g. "--grad-clip 1.0"

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

CMD+=($IMAGE_SIZE $NUM_CLASSES $PRETRAINED $RANDOM_CROP $HORIZONTAL_FLIP $RANDAUGMENT)
CMD+=($DOWNLOAD $DRY_RUN $AMP $RESUME $SAVE $GRAD_CLIP)

echo "Running command:"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"
