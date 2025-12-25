#!/usr/bin/env bash
#SBATCH --job-name=pfm-sweep
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --array=0-9

set -euo pipefail

: "${CONFIG_LIST:?Set CONFIG_LIST to a file with one config path per line}"
: "${OUT_ROOT:?Set OUT_ROOT for suite outputs}"

CONFIG_PATH=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CONFIG_LIST")
if [[ -z "$CONFIG_PATH" ]]; then
  echo "No config found for task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi

TASK_ROOT="${OUT_ROOT}/task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TASK_ROOT"
RUN_ROOT="${TASK_ROOT}/runs"
OUT_SCORECARD="${TASK_ROOT}/scorecard.json"
cp "$CONFIG_PATH" "${TASK_ROOT}/config.json"

PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${PYTHONPATH:-./src}"

"$PYTHON" scripts/run_perturbench_suite.py \
  --config "$CONFIG_PATH" \
  --out "$OUT_SCORECARD" \
  --run-root "$RUN_ROOT"
