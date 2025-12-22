#!/usr/bin/env bash
set -euo pipefail

echo "[check] import smoke"
python3 -c "import perturbfm; print('ok')"

echo "[check] cli smoke"
perturbfm --help >/dev/null

echo "[check] tests"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
