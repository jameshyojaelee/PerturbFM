#!/usr/bin/env bash
#SBATCH --job-name=pfm-train
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

set -euo pipefail

: "${DATA_PATH:?Set DATA_PATH to a PerturbDataset artifact}"
: "${SPLIT_HASH:?Set SPLIT_HASH to a stored split hash}"

MODEL_KIND="${MODEL_KIND:-baseline}"
OUT_DIR="${OUT_DIR:-runs/slurm_${SLURM_JOB_ID}}"
DEVICE="${DEVICE:-cuda}"
USE_TORCHRUN="${USE_TORCHRUN:-0}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"

BASE_CMD=(python -m perturbfm.cli train)

case "$MODEL_KIND" in
  baseline)
    BASELINE_NAME="${BASELINE_NAME:-global_mean}"
    CMD=("${BASE_CMD[@]}" baseline --data "$DATA_PATH" --split "$SPLIT_HASH" --baseline "$BASELINE_NAME" --out "$OUT_DIR")
    ;;
  perturbfm-v0)
    CMD=("${BASE_CMD[@]}" perturbfm-v0 --data "$DATA_PATH" --split "$SPLIT_HASH" --device "$DEVICE" --out "$OUT_DIR")
    ;;
  perturbfm-v1)
    : "${ADJACENCY_PATH:?Set ADJACENCY_PATH for v1}"
    : "${PERT_MASKS_PATH:?Set PERT_MASKS_PATH for v1}"
    CMD=("${BASE_CMD[@]}" perturbfm-v1 --data "$DATA_PATH" --split "$SPLIT_HASH" --adjacency "$ADJACENCY_PATH" --pert-masks "$PERT_MASKS_PATH" --device "$DEVICE" --out "$OUT_DIR")
    ;;
  perturbfm-v2)
    CMD=("${BASE_CMD[@]}" perturbfm-v2 --data "$DATA_PATH" --split "$SPLIT_HASH" --device "$DEVICE" --out "$OUT_DIR")
    if [[ -n "${ADJACENCY_PATHS:-}" ]]; then
      IFS=',' read -r -a ADJ <<< "$ADJACENCY_PATHS"
      for path in "${ADJ[@]}"; do
        CMD+=("--adjacency" "$path")
      done
    fi
    ;;
  *)
    echo "Unknown MODEL_KIND=$MODEL_KIND" >&2
    exit 1
    ;;
esac

if [[ "$USE_TORCHRUN" -eq 1 ]]; then
  MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
  MASTER_PORT="${MASTER_PORT:-29500}"
  NNODES="${SLURM_NNODES:-1}"
  NODE_RANK="${SLURM_NODEID:-0}"
  torchrun \
    --nnodes "$NNODES" \
    --nproc_per_node "$GPUS_PER_NODE" \
    --node_rank "$NODE_RANK" \
    --rdzv_id "${SLURM_JOB_ID}" \
    --rdzv_backend c10d \
    --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
    "${CMD[@]}"
else
  "${CMD[@]}"
fi
