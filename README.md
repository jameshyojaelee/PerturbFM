# PerturbFM

Predict perturbation-induced gene-expression changes — with **frozen out-of-distribution (OOD) splits**, **complete metric panels**, and **calibrated uncertainty**

Perturbation modeling is one of those areas where it’s easy to get impressive numbers for the wrong reasons: split drift, silent leakage, cherry-picked metrics, or evaluation scripts that don’t match the benchmark’s intent. This repository is designed to prevent those failure modes by making rigor the default.

Why I started this project:
- Wet-lab perturbation experiments are expensive and combinatorial; we can’t measure every (context × perturbation × dose × time) condition.
- Perturbation response models predict **gene expression change** (delta) from a baseline/control state + an intervention description + context/covariates, enabling **in silico screening** and better experimental prioritization.
- The “foundation model” goal is reusable structure + generalization (unseen contexts / perturbations / combinations) with uncertainty that tells you when not to trust a prediction.

## What’s in the repo right now

Core pieces (working):
- Canonical dataset object: `PerturbDataset` (+ artifact I/O).
- Split system: frozen, hash‑locked splits stored under `splits/`.
- CLI: `perturbfm data`, `perturbfm splits`, `perturbfm train`, `perturbfm eval`.
- Models:
  - Baselines (control-only, mean delta variants, ridge, latent shift)
  - PerturbFM v0 (probabilistic)
  - PerturbFM v1 (graph + gating)
  - PerturbFM v2 (CGIO: graph‑propagated intervention + contextual operator)
  - PerturbFM v3 (state‑dependent CGIO)
- Evaluation:
  - scPerturBench-style metric panel (implemented)
  - PerturBench-style metric panel (implemented)
  - Uncertainty outputs + basic conformal intervals (optional)
- Tooling:
  - health check: `scripts/check.sh`
  - metric parity harness: `scripts/validate_metrics.py`
  - suite runners + scorecards: `scripts/run_*_suite.py`
  - sweeps + leaderboard aggregation: `scripts/launch_sweep.py`, `scripts/aggregate_leaderboard.py`
  - pretraining + downloads: `scripts/pretrain_cell_encoder.py`, `scripts/download_*.py`

Known gaps / “not done yet”:
- Official metric parity vs external reference scripts (requires local `third_party/` repos; use `scripts/validate_metrics.py`).
- Training control/reporting: early stopping on OOD validation + richer reports (TODO).

## Big picture (pipeline)

![PerturbFM pipeline diagram](docs/diagrams/pub/pipeline_lane.svg)

If you can’t see the image above for some reason, read it as:

`data artifact -> frozen split -> train -> predictions -> metrics+calibration -> report`

## Models (at a glance)

- **Baselines**
  - `control_only`, `global_mean`, `per_perturbation_mean`, `per_perturbation_context_mean`
  - `ridge` (predict delta from control expression), `latent_shift` (PCA shift)
- **PerturbFM v0**
  - simple probabilistic model in delta space (mean + per-gene variance)
  - inputs: control expression + learned perturbation/context embeddings
- **PerturbFM v1**
  - graph-augmented perturbation encoder + trust gating
  - intended for multi-gene perturbations when you have a gene graph
- **PerturbFM v2 (CGIO)**
  - intervention is a **gene set** (`obs["pert_genes"]`)
  - propagate intervention over one or more graphs (with gating)
  - predict delta via a **context-conditioned low-rank operator**
- **PerturbFM v3**
  - conditions on `x_control` to model state-dependent perturbation effects

CGIO sketch:

![PerturbFM v2 (CGIO) diagram](docs/diagrams/pub/cgio_operator.svg)

If you can’t see the image above for some reason, read it as:

`pert_genes -> pert_mask -> graph propagation (+ context) -> contextual operator -> delta mean/var`

## Repository layout

- `src/perturbfm/`: package code
  - `data/`: dataset abstraction, adapters, splits
  - `models/`: baselines + PerturbFM models
  - `train/`: training helpers
  - `eval/`: metrics, evaluator, report
- `scripts/`: helpers (runs, suites, exports, parity harness)
- `docs/`: tracked documentation (see `docs/`)
- `tests/`: unit + smoke tests
- `splits/`: immutable split artifacts (tracked)
- `runs/`: run artifacts (gitignored)
- `third_party/`: external benchmark clones (gitignored)
- `data/`: downloaded datasets / artifacts (gitignored)

## Quickstart (synthetic smoke run)

Minimal end-to-end run on a toy dataset:

```bash
perturbfm data make-synth --out /tmp/pfm_synth --n-obs 200 --n-genes 50 --n-contexts 3 --n-perts 5
perturbfm splits create --data /tmp/pfm_synth --spec context_ood --holdout-context C0
# capture the printed split_hash
perturbfm train baseline --data /tmp/pfm_synth --split <SPLIT_HASH> --baseline global_mean
```

The `train` command prints `run_dir=...`; metrics are written to `runs/<run_id>/metrics.json`.

## Workflow (real data)

1) **Download processed datasets** (or use your own):
   - PerturBench: `python scripts/download_perturbench.py --out data/perturbench`
   - scPerturb: `python scripts/download_scperturb.py --out data/scperturb`
2) **Import to artifacts**:
   - PerturBench: `perturbfm data import-perturbench --dataset data/perturbench/<file>.h5ad --out data/artifacts/perturbench/<name>`
   - scPerturb: `python scripts/import_scperturbench.py --dataset data/scperturb/<file>.h5ad --out data/artifacts/scperturb/<name>`
3) **Create or import splits**:
   - Create: `perturbfm splits create --data <artifact> --spec context_ood --holdout-context C0`
   - Import official PerturBench splits (if available): `perturbfm splits import-perturbench --dataset <h5ad> --data <artifact> --split-dir <splits_dir>`
4) **Train + evaluate**:
   - `perturbfm train perturbfm-v0|v1|v2|v3 ...`
   - `perturbfm eval predictions --data <artifact> --split <hash> --preds <predictions.npz> --out <out_dir>`
5) **Run suites / scorecards**:
   - `python scripts/run_perturbench_suite.py --config <suite.json> --out <scorecard.json>`
   - `python scripts/run_scperturbench_suite.py --config <suite.json> --out <scorecard.json>`
6) **Sweep + leaderboard**:
   - `python scripts/launch_sweep.py --base <base.json> --grid <grid.json> --out sweeps/run1`
   - `python scripts/aggregate_leaderboard.py --runs runs --out leaderboard.json`

## Dataset artifact schema

Each dataset artifact directory contains:
- `data.npz` with keys:
  - `X_control`, `X_pert`, `delta` (float32, shape `[N, G]`)
  - `obs_idx` (int64, shape `[N]`)
- `meta.json` with keys:
  - `obs` (table-like dict; must include `pert_id`, `context_id`, `batch_id`, `is_control`)
  - `var` (gene identifiers list)
  - `metadata` (freeform dict)

Common `obs` fields you’ll see in practice:
- `pert_id`: `"control"` or a perturbation ID (single or combo string)
- `context_id`: cell type / cell line / condition grouping
- `batch_id`: batch identifier
- `is_control`: boolean
- `pert_genes`: (v2 CGIO) list of perturbed gene IDs per row (empty list for controls)
- `covariates`: optional dict of arrays (dose/time/etc.)

## Graph format + gating modes (v1/v2)

Graph files should be sparse‑friendly and aligned to `var` order:

- `graphs/<name>.npz`
  - `adjacency`: float32 `[G, G]` (dense allowed for small graphs; converted to sparse internally)
  - `genes`: list of length `G` matching `var`

Gating modes (for graph trust):
- `none` — no gating
- `scalar` — single global gate
- `node` — per‑node gate
- `lowrank` — low‑rank edge gating (U,V embeddings)
- `mlp` — edge gating via small MLP over node embeddings

## Split system

Splits are immutable and hash‑locked:
- Stored as JSON files under `splits/` (filename is the split hash).
- Every run records the split hash (`split_hash.txt` + `config.json`).
- Training/eval should never “silently regenerate” a split.

You can override the split store location with `PERTURBFM_SPLIT_DIR` (used in tests).

## Run artifacts

Each run writes:
- `config.json`
- `split_hash.txt`
- `predictions.npz` (`mean`, `var`, `idx`)
- `metrics.json`
- `calibration.json`
- `report.html`

### Metrics and calibration outputs

`metrics.json` is a single combined object that always includes:

- `scperturbench`
  - `global`: metric name → value
  - `per_perturbation`: perturbation → metrics
  - `per_context`: context → metrics
- `perturbench`
  - same structure as above
- `uncertainty`
  - `coverage`: `{0.5, 0.8, 0.9, 0.95}` → empirical coverage
  - `nll`: Gaussian NLL in delta space
  - `risk_coverage`: arrays of risk vs coverage points
  - `ood_auroc`: AUROC if `ood_labels` are provided
  - `conformal`: optional conformal interval stats (when enabled)

`calibration.json` currently mirrors the `uncertainty` panel (so you can load it without parsing the full `metrics.json`).

### Run IDs and hashes

Runs are stored under `runs/<run_id>/` (gitignored). The run id is designed to be sortable and traceable:

- `<UTCYYYYMMDD-HHMMSS>_<splitHash7>_<modelName>_<configHash>`

`config.json` also stores:
- `data_hash` (hash of the dataset artifact metadata)
- `split_hash`
- `config_hash`

## Development and validation

Install (editable):

```bash
python -m pip install -e ".[dev]"
```

Run tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

Validate metrics before making benchmark claims:

```bash
python scripts/validate_metrics.py --data <artifact_dir> --preds <predictions.npz>
```

External benchmark repos (GPL / non-vendored) should live under `third_party/` (gitignored). See `docs/perturbench_contract.md`.

## Related models and benchmarks

External models (for context/positioning):
- [STATE](https://github.com/ArcInstitute/state)
- [GEARS](https://github.com/snap-stanford/GEARS)
- [scGen](https://github.com/theislab/scgen)
- [CPA](https://github.com/theislab/cpa)

Comparison summary (how we position models in this repo):

| Model | Core idea / inputs | Graph prior | Pert genes | Prediction space in this repo | Uncertainty | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Baselines (control_only/global_mean/per_pert/latent_shift/ridge) | Simple statistics or linear baselines | No | No | Delta | None | Fast sanity checks; no calibration. |
| PerturbFM v0 | Probabilistic delta model | No | No | Delta | Native | Default internal reference. |
| PerturbFM v1 | Graph + gating | Yes (adjacency) | Mask per-pert | Delta | Native | Requires graph + pert masks. |
| PerturbFM v2 | CGIO (intervention) | Yes (adjacency) | obs["pert_genes"] | Delta | Native | Requires per-row pert gene lists. |
| PerturbFM v3 | State-dependent CGIO | Yes (adjacency) | obs["pert_genes"] + X_control | Delta | Native | Uses control expression. |
| STATE | External state model | No (in our pipeline) | No | Delta (via conversion) | Fixed (filled) | Run externally; eval via exported predictions. |
| GEARS | Graph-based perturbation model | Yes (GO/coexpr in GEARS) | Uses pert gene list | Delta (via conversion) | Fixed (filled) | External; predictions converted from expression. |
| scGen | Latent arithmetic on VAE | No | No | Delta (via conversion) | Fixed (filled) | External; evaluated from predicted expression. |
| CPA | Compositional perturbation autoencoder | No | No | Delta (via conversion) | Fixed (filled) | External; evaluated from predicted expression. |

Benchmarks:
- [PerturBench](https://github.com/altoslabs/perturbench)
- [scPerturBench](https://github.com/bm2-lab/scPerturBench)

## Docs

- `docs/slurm.md` — SLURM templates and usage.
- `docs/perturbench_contract.md` — contract for working with external PerturBench assets.
- `docs/perturbench_splits.md` — split availability notes.
- `docs/dev_todo.md` — internal TODOs and acceptance criteria.
