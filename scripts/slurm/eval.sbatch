#!/usr/bin/env bash
#SBATCH --job-name=pfm-eval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

set -euo pipefail

: "${DATA_PATH:?Set DATA_PATH to a PerturbDataset artifact}"
: "${SPLIT_HASH:?Set SPLIT_HASH to a stored split hash}"
: "${PREDS_PATH:?Set PREDS_PATH to predictions.npz}"

OUT_DIR="${OUT_DIR:-runs/eval_${SLURM_JOB_ID}}"

python -m perturbfm.cli eval predictions \
  --data "$DATA_PATH" \
  --split "$SPLIT_HASH" \
  --preds "$PREDS_PATH" \
  --out "$OUT_DIR"
