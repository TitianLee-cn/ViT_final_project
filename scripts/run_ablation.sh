#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

# ============================================================
# ViT ablation experiments.
# This script runs several ViT experiments. Edit the shared
# variables first, then edit individual experiment commands below.
# ============================================================

# Basic
SEED=42
DEVICE="auto"
OUTPUT_DIR="outputs"
DRY_RUN=""                    # set to "--dry-run" to test quickly
AMP="--amp"
RESUME=""                     # usually keep empty for ablation
SAVE=""

# Dataset
DATASET="cifar10"
DATA_DIR="data"
NUM_WORKERS=4
VAL_RATIO=0.1
IMAGE_SIZE=""
DOWNLOAD="--download"

# Model
MODEL="vit"
NUM_CLASSES=""
PRETRAINED=""

# ViT default parameters
PATCH_SIZE=4
VIT_DIM=256
VIT_DEPTH=6
VIT_HEADS=8
VIT_MLP_RATIO=4.0
QKV_BIAS="--qkv-bias"
CLS_TOKEN="--cls-token"

# Image-level augmentation defaults
RANDOM_CROP=""
CROP_PADDING=4
HORIZONTAL_FLIP=""
HFLIP_PROB=0.5
RANDAUGMENT=""
RA_NUM_OPS=2
RA_MAGNITUDE=9

# Batch-level augmentation defaults
MIXUP_ALPHA=0.0
CUTMIX_ALPHA=0.0
MIXUP_PROB=1.0
MIXUP_SWITCH_PROB=0.5

# Regularization defaults
DROPOUT=0.1
ATTENTION_DROPOUT=0.1
DROP_PATH_RATE=0.0
LABEL_SMOOTHING=0.0
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

run_vit() {
  local experiment_name="$1"
  local patch_size="$2"
  local drop_path_rate="$3"
  local randaugment_flag="$4"
  local mixup_alpha="$5"
  local cutmix_alpha="$6"

  local cmd=(python main.py
    --seed "$SEED"
    --device "$DEVICE"
    --output-dir "$OUTPUT_DIR"
    --experiment-name "$experiment_name"
    --dataset "$DATASET"
    --data-dir "$DATA_DIR"
    --num-workers "$NUM_WORKERS"
    --val-ratio "$VAL_RATIO"
    --model "$MODEL"
    --patch-size "$patch_size"
    --vit-dim "$VIT_DIM"
    --vit-depth "$VIT_DEPTH"
    --vit-heads "$VIT_HEADS"
    --vit-mlp-ratio "$VIT_MLP_RATIO"
    --crop-padding "$CROP_PADDING"
    --hflip-prob "$HFLIP_PROB"
    --ra-num-ops "$RA_NUM_OPS"
    --ra-magnitude "$RA_MAGNITUDE"
    --mixup-alpha "$mixup_alpha"
    --cutmix-alpha "$cutmix_alpha"
    --mixup-prob "$MIXUP_PROB"
    --mixup-switch-prob "$MIXUP_SWITCH_PROB"
    --dropout "$DROPOUT"
    --attention-dropout "$ATTENTION_DROPOUT"
    --drop-path-rate "$drop_path_rate"
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

  cmd+=($IMAGE_SIZE $NUM_CLASSES $PRETRAINED $QKV_BIAS $CLS_TOKEN)
  cmd+=($RANDOM_CROP $HORIZONTAL_FLIP $randaugment_flag)
  cmd+=($DOWNLOAD $DRY_RUN $AMP $RESUME $SAVE $GRAD_CLIP)

  echo "Running command:"
  printf ' %q' "${cmd[@]}"
  echo
  "${cmd[@]}"
}

# ============================================================
# Experiments. Comment out lines you do not want to run.
# ============================================================

# Patch size ablation
run_vit "ablation_patch4" 4 "$DROP_PATH_RATE" "$RANDAUGMENT" "$MIXUP_ALPHA" "$CUTMIX_ALPHA"
run_vit "ablation_patch8" 8 "$DROP_PATH_RATE" "$RANDAUGMENT" "$MIXUP_ALPHA" "$CUTMIX_ALPHA"

# DropPath ablation
run_vit "ablation_droppath0" "$PATCH_SIZE" 0.0 "$RANDAUGMENT" "$MIXUP_ALPHA" "$CUTMIX_ALPHA"
run_vit "ablation_droppath01" "$PATCH_SIZE" 0.1 "$RANDAUGMENT" "$MIXUP_ALPHA" "$CUTMIX_ALPHA"

# RandAugment ablation
run_vit "ablation_no_randaugment" "$PATCH_SIZE" "$DROP_PATH_RATE" "" "$MIXUP_ALPHA" "$CUTMIX_ALPHA"
run_vit "ablation_randaugment" "$PATCH_SIZE" "$DROP_PATH_RATE" "--randaugment" "$MIXUP_ALPHA" "$CUTMIX_ALPHA"

# MixUp/CutMix ablation
run_vit "ablation_no_mix" "$PATCH_SIZE" "$DROP_PATH_RATE" "$RANDAUGMENT" 0.0 0.0
run_vit "ablation_mix_cutmix" "$PATCH_SIZE" "$DROP_PATH_RATE" "$RANDAUGMENT" 0.2 1.0
