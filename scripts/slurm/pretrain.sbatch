#!/usr/bin/env bash
#SBATCH --job-name=pfm-pretrain
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

set -euo pipefail

: "${DATA_PATH:?Set DATA_PATH to a PerturbDataset artifact}"
: "${OUT_PATH:?Set OUT_PATH to encoder checkpoint (.pt)}"

EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-1e-3}"
DEVICE="${DEVICE:-cuda}"

python scripts/pretrain_cell_encoder.py \
  --data "$DATA_PATH" \
  --out "$OUT_PATH" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --device "$DEVICE"
