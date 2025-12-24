#!/usr/bin/env bash
#SBATCH --job-name=pfm-gears
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=120G
#SBATCH --time=04:00:00

set -euo pipefail

: "${ENV_PREFIX:?Set ENV_PREFIX to micromamba env prefix (env_gears)}"
: "${ADATA:?Set ADATA to .h5ad path}"
: "${ARTIFACT:?Set ARTIFACT to PerturbDataset dir}"
: "${SPLIT_HASH:?Set SPLIT_HASH}"
: "${WORKDIR:?Set WORKDIR for GEARS artifacts}"
: "${OUT_PREDS:?Set OUT_PREDS path}"

export PYTHONNOUSERSITE=1

micromamba run -p "${ENV_PREFIX}" python scripts/external/run_gears.py \
  --adata "${ADATA}" \
  --artifact "${ARTIFACT}" \
  --split "${SPLIT_HASH}" \
  --workdir "${WORKDIR}" \
  --out-preds "${OUT_PREDS}" \
  --epochs 10 \
  --hidden-size 64 \
  --batch-size 32 \
  --device cuda

