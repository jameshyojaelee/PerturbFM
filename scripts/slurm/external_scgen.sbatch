#!/usr/bin/env bash
#SBATCH --job-name=pfm-scgen
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=120G
#SBATCH --time=06:00:00

set -euo pipefail

: "${ENV_PREFIX:?Set ENV_PREFIX to micromamba env prefix (env_scgen)}"
: "${ADATA:?Set ADATA to .h5ad path}"
: "${ARTIFACT:?Set ARTIFACT to PerturbDataset dir}"
: "${SPLIT_HASH:?Set SPLIT_HASH}"
: "${HOLDOUT_CONTEXT:?Set HOLDOUT_CONTEXT id}"
: "${OUT_PREDS:?Set OUT_PREDS path}"

export PYTHONNOUSERSITE=1
export PYTHONPATH=src
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

micromamba run -p "${ENV_PREFIX}" python scripts/external/run_scgen.py \
  --adata "${ADATA}" \
  --artifact "${ARTIFACT}" \
  --split "${SPLIT_HASH}" \
  --out-preds "${OUT_PREDS}" \
  --holdout-context "${HOLDOUT_CONTEXT}" \
  --max-epochs 50 \
  --batch-size 128 \
  --n-hidden 256 \
  --n-latent 64 \
  --n-layers 2
