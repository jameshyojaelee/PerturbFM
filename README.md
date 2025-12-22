# PerturbFM

Universal perturbation foundation model with graph priors and calibrated uncertainty.

This repo enforces immutable OOD splits, full metric panels, and explicit uncertainty evaluation. The architecture plan lives in `project_overview.md`.

## Current status

Implemented:
- Canonical `PerturbDataset` abstraction + artifact I/O.
- Split system with hash‑locked, stored splits.
- Baseline suite + minimal evaluator that writes run artifacts.
- Metrics/uncertainty panels (ambiguous metrics marked TODO).
- PerturbFM v0 (probabilistic) and v1 (graph‑augmented) scaffolds.
- CLI for `data`, `splits`, `train`, `eval`.

Known gaps / TODOs:
- Benchmark adapters: `PerturBenchAdapter` currently loads `PerturbDataset` artifacts only (real AnnData/PerturBench loaders + official splits are TODO). `scPerturBenchAdapter` is intentionally a stub (GPL isolation).
- Metrics: scPerturBench (Energy/Wasserstein/KL/Common‑DEGs) and PerturBench (rank metrics / collapse diagnostics) are wired but return `NaN` until validated against reference scripts.
- Uncertainty: deep ensemble helper exists, but conformal/calibration modules are not implemented yet.
- Training control: no early stopping on OOD validation metrics; HTML report is minimal.

## Repository layout

- `src/perturbfm/`: package code
  - `data/`: dataset abstraction, adapters, splits
  - `models/`: baselines + PerturbFM models
  - `train/`: training helpers
  - `eval/`: metrics, evaluator, report
  - `utils/`: hashing, logging, seeding
- `scripts/`: placeholder helpers for external benchmarks
- `tests/`: unit + smoke tests
- `runs/`: run artifacts (gitignored)
- `third_party/`: external benchmarks (gitignored)
- `splits/`: versioned split artifacts (tracked)

## Dataset artifact schema

Each dataset artifact directory contains:
- `data.npz` with keys:
  - `X_control`, `X_pert`, `delta` (float32, shape `[N, G]`)
  - `obs_idx` (int64, shape `[N]`)
- `meta.json` with keys:
  - `obs` (table-like dict; must include `pert_id`, `context_id`, `batch_id`, `is_control`)
  - `var` (gene identifiers list)
  - `metadata` (freeform dict)

## Split system

Splits are immutable and hash‑locked. They are stored in `splits/` as JSON and referenced by hash in all runs.

You can override the split store location with `PERTURBFM_SPLIT_DIR` (used in tests).

## Run artifacts

Each run writes:
- `config.json`
- `split_hash.txt`
- `predictions.npz` (`mean`, `var`, `idx`)
- `metrics.json`
- `calibration.json`
- `report.html`

## CLI quickstart

```bash
perturbfm data make-synth --out /tmp/pfm_synth
perturbfm splits create --data /tmp/pfm_synth --spec context_ood --holdout-context C0
perturbfm train baseline --data /tmp/pfm_synth --split <HASH> --baseline global_mean
```

Evaluate existing predictions:

```bash
perturbfm eval predictions --data /tmp/pfm_synth --split <HASH> --preds /path/to/predictions.npz --out /tmp/pfm_eval
```

## Testing

```bash
pytest -q
```

If `pytest` crashes in your environment due to SSL/plugin loading issues (e.g., `ImportError: libssl.so.1.1`), try:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

Quick health check:

```bash
scripts/check.sh
```

## Roadmap (next)

1) Validate metric implementations numerically against official benchmark scripts (executed from `third_party/`, without vendoring GPL code).
2) Implement real benchmark adapters + official split import/export (PerturBench first).
3) Implement a novelty candidate model (v2): context‑conditional graph intervention operators with uncertainty‑aware graph trust gating + calibration on OOD splits.
