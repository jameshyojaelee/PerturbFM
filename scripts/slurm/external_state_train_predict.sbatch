#!/usr/bin/env bash
#SBATCH --job-name=pfm-state
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=120G
#SBATCH --time=04:00:00

set -euo pipefail

: "${ENV_PREFIX:?Set ENV_PREFIX to micromamba env prefix (e.g., third_party/mamba/env_scgen)}"
: "${TOML_CONFIG:?Set TOML_CONFIG to a STATE dataset split TOML}"
: "${DATA_ARTIFACT:?Set DATA_ARTIFACT to a PerturbDataset artifact dir}"
: "${SPLIT_HASH:?Set SPLIT_HASH to a SplitStore hash (test_idx used for export)}"
: "${RUNS_ROOT:?Set RUNS_ROOT for STATE run output (will create RUNS_ROOT/NAME)}"
: "${NAME:?Set NAME for the experiment folder name under RUNS_ROOT}"

export PYTHONNOUSERSITE=1
export PYTHONPATH=src

RUN_DIR="${RUNS_ROOT}/${NAME}"

echo "[info] Training STATE -> ${RUN_DIR}"
micromamba run -p "${ENV_PREFIX}" state tx train \
  data.kwargs.toml_config_path="${TOML_CONFIG}" \
  data.kwargs.embed_key="X_hvg" \
  data.kwargs.batch_col="batch_id" \
  data.kwargs.pert_col="condition" \
  data.kwargs.cell_type_key="cell_type" \
  data.kwargs.control_pert="ctrl" \
  data.kwargs.num_workers="${SLURM_CPUS_PER_TASK}" \
  data.kwargs.pin_memory="true" \
  model.device="cuda" \
  model.kwargs.hidden_dim=128 \
  model.kwargs.cell_set_len=128 \
  training.batch_size=8 \
  training.max_steps=2000 \
  training.val_freq=200 \
  training.ckpt_every_n_steps=200 \
  use_wandb=false \
  overwrite=true \
  output_dir="${RUNS_ROOT}" \
  name="${NAME}"

echo "[info] Predicting (writes adata_pred.h5ad)"
micromamba run -p "${ENV_PREFIX}" state tx predict \
  --output-dir "${RUN_DIR}" \
  --checkpoint "last.ckpt" \
  --profile "anndata" \
  --predict-only

PRED_ADATA="${RUN_DIR}/eval_last.ckpt/adata_pred.h5ad"
if [[ ! -f "${PRED_ADATA}" ]]; then
  echo "[error] Missing ${PRED_ADATA}" >&2
  exit 1
fi

OUT_PREDS="runs/external_predictions/state/${NAME}/${SPLIT_HASH}/predictions.npz"
mkdir -p "$(dirname "${OUT_PREDS}")"

echo "[info] Converting to PerturbFM predictions.npz -> ${OUT_PREDS}"
micromamba run -p "${ENV_PREFIX}" python scripts/external/convert_adata_pred_to_predictions.py \
  --data "${DATA_ARTIFACT}" \
  --split "${SPLIT_HASH}" \
  --adata-pred "${PRED_ADATA}" \
  --pert-col "condition" \
  --celltype-col "cell_type" \
  --batch-col "batch_id" \
  --out "${OUT_PREDS}"

echo "[ok] Done. Predictions: ${OUT_PREDS}"
