#!/usr/bin/env bash
#SBATCH --job-name=pfm-dl-all
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00

set -euo pipefail

: "${PERTURBENCH_DIR:?Set PERTURBENCH_DIR}"
: "${SCPERTURB_DIR:?Set SCPERTURB_DIR}"

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

"$PYTHON" scripts/download_perturbench.py --out "$PERTURBENCH_DIR" --max-workers "$MAX_WORKERS"
"$PYTHON" scripts/download_scperturb.py --out "$SCPERTURB_DIR" --max-workers "$MAX_WORKERS"
