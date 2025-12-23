# PerturBench Contract (data, splits, metrics)

This document defines the **local contract** for working with the official PerturBench benchmark
without vendoring any external code.

## 1) Expected external layout

We assume you keep the official repository under `third_party/PerturBench` (gitignored).
At minimum, the repo should expose:

- One or more datasets (e.g., `.h5ad` files or dataset-specific folders).
- Official split files (JSON or similar) under a predictable path (recommended: `datasets/<name>/splits/`).
- A reference metric script that prints JSON to stdout (recommended name: `reference_metrics.py`).

If your local layout differs, update `scripts/validate_metrics.py` to point at the correct script.

## 2) Dataset import into PerturbFM artifacts

PerturbFM uses a canonical `PerturbDataset` artifact format (`data.npz` + `meta.json`).

The adapter contract for PerturBench datasets is:

- Required `obs` fields: `pert_id`, `context_id`, `batch_id`, `is_control`
- Optional `obs` fields: `pert_genes`, `covariates`
- Gene order must be consistent across `X_control`, `X_pert`, and `delta`

If you import `.h5ad` directly, install extras:

```
pip install -e ".[bench]"
```

## 3) Official splits → SplitStore (hash-locked)

When importing official splits:

- Preserve `train_idx`, `val_idx`, `test_idx` (and `calib_idx` if provided).
- Record `ood_axes` in split metadata when known.
- Freeze and save splits into `SplitStore` with their hash (no silent regeneration).

The expectation is that `PerturBenchAdapter.load_official_splits()` discovers split JSON
from a known location and returns a dict that can be ingested into SplitStore.

## 4) Metric parity gate

Use the parity harness:

```
python scripts/validate_metrics.py --data <artifact> --preds <predictions.npz>
```

- Always compute internal metrics and print them.
- If `third_party/PerturBench` is present and provides `reference_metrics.py`,
  the script compares globals within tolerance and exits non-zero on mismatch.
- If the external repo is missing, the script prints the expected path and
  next steps, but still exits successfully.

## 5) Provenance requirements

Every run should log:

- data hash
- split hash
- config hash
- git commit
- random seeds
- environment metadata

This is enforced by the evaluator outputs and should remain mandatory for PerturBench runs.
