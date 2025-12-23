#!/usr/bin/env bash
#SBATCH --job-name=pfm-dl-scperturb
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=164G
#SBATCH --time=08:00:00

set -euo pipefail

: "${OUT_DIR:?Set OUT_DIR to target download directory}"

MAX_WORKERS="${MAX_WORKERS:-${SLURM_CPUS_PER_TASK}}"

if [[ -n "${VENV_PATH:-}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
elif [[ -n "${CONDA_ENV:-}" ]]; then
  CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
  if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
  else
    echo "conda.sh not found under ${CONDA_BASE}; set CONDA_BASE or use VENV_PATH" >&2
    exit 1
  fi
fi

PYTHON="${PYTHON:-python3}"

"$PYTHON" scripts/download_scperturb.py --out "$OUT_DIR" --max-workers "$MAX_WORKERS"
