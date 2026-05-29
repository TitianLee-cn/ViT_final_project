#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

# ============================================================
# Small grid search for ViT hyperparameters.
#
# Usage:
#   bash scripts/run_grid_search.sh
#
# Recommended workflow:
#   1. Run this short grid search with EPOCHS=30.
#   2. Compare outputs/logs/grid_*/final_result.json.
#   3. Retrain the best 1-3 settings for 100 epochs.
# ============================================================

# Basic
SEED=42
DEVICE="auto"                 # auto / cpu / cuda
OUTPUT_DIR="outputs"
AMP="--amp"                   # set to "" to disable mixed precision
DRY_RUN=""                    # set to "--dry-run" for a quick test

# Dataset
DATASET="cifar10"
DATA_DIR="data"
NUM_WORKERS=6
VAL_RATIO=0.1
DOWNLOAD=""                   # set to "--download" only if the cluster can download CIFAR-10

# Fixed ViT settings
MODEL="vit"
VIT_HEADS=8
VIT_MLP_RATIO=4.0
QKV_BIAS="--qkv-bias"
CLS_TOKEN="--cls-token"

# Augmentation and regularization used during search
RANDOM_CROP="--random-crop"
HORIZONTAL_FLIP="--horizontal-flip"
RANDAUGMENT="--randaugment"
MIXUP_ALPHA=0.2
CUTMIX_ALPHA=1.0
DROPOUT=0.1
ATTENTION_DROPOUT=0.1
LABEL_SMOOTHING=0.1
WEIGHT_DECAY=0.05

# Training
EPOCHS=30
BATCH_SIZE=128
OPTIMIZER="adamw"
SCHEDULER="cosine"
MIN_LR=0.000001
MOMENTUM=0.9
LOG_INTERVAL=50
EVAL_INTERVAL=1

# Search space: 2 * 2 * 2 * 2 * 2 = 32 runs
PATCH_SIZES=(4 8)
VIT_DEPTHS=(4 6)
VIT_DIMS=(128 256)
LRS=(0.0001 0.0003)
DROP_PATH_RATES=(0.0 0.1)

for PATCH_SIZE in "${PATCH_SIZES[@]}"; do
  for VIT_DEPTH in "${VIT_DEPTHS[@]}"; do
    for VIT_DIM in "${VIT_DIMS[@]}"; do
      for LR in "${LRS[@]}"; do
        for DROP_PATH_RATE in "${DROP_PATH_RATES[@]}"; do
          if (( VIT_DIM % VIT_HEADS != 0 )); then
            echo "Skip invalid setting: vit_dim=${VIT_DIM}, vit_heads=${VIT_HEADS}"
            continue
          fi

          LR_TAG="${LR//./p}"
          DP_TAG="${DROP_PATH_RATE//./p}"
          EXPERIMENT_NAME="grid_p${PATCH_SIZE}_d${VIT_DEPTH}_dim${VIT_DIM}_lr${LR_TAG}_dp${DP_TAG}"

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
            --patch-size "$PATCH_SIZE"
            --vit-dim "$VIT_DIM"
            --vit-depth "$VIT_DEPTH"
            --vit-heads "$VIT_HEADS"
            --vit-mlp-ratio "$VIT_MLP_RATIO"
            --mixup-alpha "$MIXUP_ALPHA"
            --cutmix-alpha "$CUTMIX_ALPHA"
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
            --min-lr "$MIN_LR"
            --momentum "$MOMENTUM"
            --log-interval "$LOG_INTERVAL"
            --eval-interval "$EVAL_INTERVAL"
          )

          CMD+=($QKV_BIAS $CLS_TOKEN)
          CMD+=($RANDOM_CROP $HORIZONTAL_FLIP $RANDAUGMENT)
          CMD+=($DOWNLOAD $AMP $DRY_RUN)

          echo "============================================================"
          echo "Running grid search experiment: ${EXPERIMENT_NAME}"
          printf ' %q' "${CMD[@]}"
          echo
          "${CMD[@]}"
        done
      done
    done
  done
done
