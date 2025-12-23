# PerturbFM

Perturbation response prediction with **immutable OOD splits**, **full metric panels**, and **calibrated uncertainty**

Perturbation modeling is one of those areas where it’s easy to get impressive numbers for the wrong reasons: split drift, silent leakage, cherry-picked metrics, or evaluation scripts that don’t match the benchmark’s intent. This repository is designed to prevent those failure modes by making rigor the default.

The goal of this work is not “just another model.” It’s a workflow that turns your experiments into artifacts you can stand behind: frozen splits, complete metric panels, calibrated uncertainty, and provenance-rich run outputs — so when a result looks good, you can trust it (and reproduce it later).

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
- Scripts:
  - `scripts/check.sh` (quick repo health)
  - `scripts/validate_metrics.py` (metric parity harness)
  - `scripts/run_ablations.py` (batch runs + summary)
  - `scripts/run_perturbench_suite.py` (suite runner + scorecard)
  - `scripts/run_scperturbench_suite.py` (scPerturBench wrapper)
  - `scripts/launch_sweep.py` / `scripts/aggregate_leaderboard.py` (sweeps + leaderboard)
  - `scripts/pretrain_cell_encoder.py` (pretraining)
  - `scripts/download_perturbench.py` / `scripts/download_scperturb.py` (dataset downloads)
  - `scripts/generate_pub_diagrams.py` (regenerate README diagrams)

Known gaps / “not done yet”:
- Real benchmarks:
  - `PerturBenchAdapter` supports artifacts and `.h5ad` (requires `anndata`), but importing *official* PerturBench splits depends on you having the benchmark repo/files locally.
  - `scPerturBenchAdapter` is intentionally external-only (GPL isolation).
- Metric definitions still need **numerical parity validation** against official reference scripts in `third_party/` (use `scripts/validate_metrics.py`).
- Training control/reporting:
  - early stopping on an OOD validation metric is still TODO
  - `report.html` is intentionally simple (tables + raw JSON)

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
- `utils/`: hashing, logging, seeding
- `scripts/`: helpers (metrics parity, export predictions, external benchmark stubs)
- `docs/`: planning docs + execution tracker (including `prompts.md`)
- `progress/`: progress log + decision log
- `tests/`: unit + smoke tests
- `runs/`: run artifacts (gitignored)
- `third_party/`: external benchmarks (gitignored)
- `splits/`: versioned split artifacts (tracked)

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

## CLI quickstart (synthetic)

This is the fastest way to sanity-check the full pipeline.

```bash
python -m pip install -e .

perturbfm data make-synth --out /tmp/pfm_synth
perturbfm splits create --data /tmp/pfm_synth --spec context_ood --holdout-context C0
perturbfm train baseline --data /tmp/pfm_synth --split <HASH> --baseline global_mean
```

Train the v2 CGIO model on the same synthetic artifact:

```bash
perturbfm train perturbfm-v2 --data /tmp/pfm_synth --split <HASH> --epochs 5
```

Evaluate existing predictions:

```bash
perturbfm eval predictions --data /tmp/pfm_synth --split <HASH> --preds /path/to/predictions.npz --out /tmp/pfm_eval
```

## Ablations (batch runs)

Create a small JSON list of configs (example):

```json
[
  {"kind": "baseline", "name": "global_mean"},
  {"kind": "v0", "epochs": 2, "hidden_dim": 16},
  {"kind": "v2", "epochs": 2, "hidden_dim": 16, "use_gating": true, "contextual_operator": true}
]
```

Then run:

```bash
python scripts/run_ablations.py --data /tmp/pfm_synth --split <HASH> --configs configs.json --out runs_summary.json
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

## Metric validation (important)

Even though the metric functions are implemented, you should validate them against official reference scripts before making claims:

```bash
python scripts/validate_metrics.py --data /tmp/pfm_synth --preds /path/to/predictions.npz
```

If you have the official benchmark repos cloned under `third_party/`, wire their reference scripts into `scripts/validate_metrics.py` and compare outputs numerically.

## External benchmarks (GPL isolation)

This repo does **not** vendor benchmark code. The intent is:

- Keep PerturbFM core permissively licensed.
- Treat scPerturBench (GPL) as an *external evaluation harness* only.

Typical workflow:
- Clone benchmarks into `third_party/` (gitignored).
- Run their scripts from `scripts/validate_metrics.py` / shell scripts without copying code into `src/perturbfm/`.
- If you need `.h5ad` loaders, install extras: `pip install -e ".[bench]"`.
- Use `scripts/run_scperturbench.sh` / `scripts/run_perturbench.sh` as launch placeholders you can customize locally.

Export predictions for external harnesses:

```bash
python scripts/export_predictions.py --preds runs/<run_id>/predictions.npz --out /tmp/preds --data /path/to/artifact
```

Import scPerturBench data (external-only):

```bash
python scripts/import_scperturbench.py --dataset <name_or_path> --out /tmp/scpb_artifact
```

Run scPerturBench-style suite (wrapper around suite runner):

```bash
python scripts/run_scperturbench_suite.py --config /path/to/suite.json --out /tmp/scpb_scorecard.json
```

## Project overview (detailed)

**A Universal Perturbation Foundation Model with Graph Priors and Calibrated Uncertainty**

### 0. Non-negotiable design constraints (hard rules)

These rules exist to prevent false-positive “wins” that have invalidated many recent perturbation papers.

1. **OOD splits are immutable artifacts**

   * All train/val/test splits are versioned, hash-locked, and stored.
   * Every experiment logs the split hash.
   * No silent regeneration of splits.

2. **No single-metric reporting**

   * scPerturBench metric panel is always reported together.
   * PerturBench rank metrics are always reported alongside RMSE-type metrics.
   * Any result missing part of the panel is invalid.

3. **Uncertainty must be evaluated**

   * Calibration metrics are mandatory.
   * OOD uncertainty separation must be demonstrated empirically.

4. **Licensing isolation**

   * scPerturBench (GPL-3.0) is treated as an external evaluation harness.
   * PerturbFM core repo remains permissively licensed by default.

---

### 1. Repository architecture (PerturbFM)

Baseline skeleton (additional files are OK as long as constraints are upheld).

```
perturbfm/
  README.md
  pyproject.toml
  progress/
    LOG.md
    DECISIONS.md
  docs/
    prompts.md

  src/perturbfm/
    cli.py
    __init__.py

    data/
      registry.py
      canonical.py
      batching.py
      adapters/
        perturbench.py
        scperturbench.py
      splits/
        split_spec.py
        split_store.py
      transforms.py

    models/
      baselines/
        mean_delta.py
        ridge_delta.py
        latent_shift.py
      perturbfm/
        cell_encoder.py
        perturb_encoder.py
        gene_graph.py
        fusion.py
        probabilistic_head.py
        model.py
      pretrain.py
      uncertainty/
        ensembles.py
        conformal.py
        calibration.py

    train/
      trainer.py
      losses.py
      optim.py

    eval/
      metrics_scperturbench.py
      metrics_perturbench.py
      uncertainty_metrics.py
      evaluator.py
      diagnostics.py
      report.py

    utils/
      seeds.py
      logging.py
      hashing.py

  scripts/
    pull_perturbench.sh
    pull_scperturbench.sh
    run_scperturbench.sh
    run_perturbench.sh
    run_perturbench_suite.py
    run_scperturbench_suite.py
    run_context_ood_ablations.py
    validate_metrics.py
    export_predictions.py
    import_scperturbench.py
    convert_to_memmap.py
    pretrain_cell_encoder.py
    launch_sweep.py
    aggregate_leaderboard.py
    slurm/
      train.sbatch
      eval.sbatch
      sweep_array.sbatch
      pretrain.sbatch

  tests/
    test_splits.py
    test_metrics.py
    test_uncertainty.py

  runs/          # gitignored
  third_party/   # external benchmark clones (gitignored)
  splits/        # tracked immutable split artifacts
```

---

### 2. Canonical data abstraction (shared across benchmarks)

#### 2.1 Canonical object: `PerturbDataset`

All adapters must map into this object.

**Fields**

* `X_control`: `[N, G]` tensor
* `X_pert`: `[N, G]` tensor
* `delta`: `[N, G] = X_pert − matched_control`
* `obs` (table):
  * `pert_id`
  * `context_id`
  * `batch_id`
  * `is_control`
  * `covariates` (dose, time, etc.)
* `var`: gene identifiers
* `metadata`: dataset name, preprocessing, gene-space spec

Supports:

* paired datasets
* unpaired distribution-level datasets

---

### 3. Benchmark adapters

#### 3.1 PerturBench adapter (first implementation target)

Responsibilities:

* Download datasets via official accessors.
* Load AnnData / PyTorch datasets.
* Import **official PerturBench splits** when available.
* Map everything into `PerturbDataset`.

Key supported tasks:

* covariate transfer
* combo prediction
* scaling stress tests
* imbalance stress tests

#### 3.2 scPerturBench adapter

Responsibilities:

* Load datasets locally (no vendoring).
* Implement **cellular context generalization**:
  * i.i.d split
  * OOD context split
* Match scPerturBench’s definition of:
  * contexts
  * perturbations
  * evaluation groups

---

### 4. Unified split system (critical to claim validity)

#### 4.1 Split object

Each split stores:

* `train_idx`, `val_idx`, `test_idx`
* `ood_axes`: `{context, perturbation, dataset}`
* `seed`
* `frozen_hash`

#### 4.2 Required split specs

**Core (headline claim)**

* **Context-OOD split**
  * hold out one or more `context_id`s
  * perturbations remain shared across train/test

**Secondary (stress tests)**

* leave-one-context-out CV
* PerturBench covariate transfer
* combo generalization (Norman-style)

---

### 5. Metric implementation (no deviations allowed)

#### 5.1 scPerturBench metrics

All must be implemented with scalable defaults and parity-checked when possible:

* MSE
* PCC-delta
* Energy distance
* Wasserstein distance
* KL divergence
* Common-DEGs

Aggregated:

* per perturbation
* per context
* global weighted mean

#### 5.2 PerturBench metrics

Required:

* RMSE
* rank-based metrics (collapse-sensitive)
* prediction variance diagnostics
* diversity/collapse diagnostics

---

### 6. Uncertainty and calibration (central contribution)

#### 6.1 Outputs

* Per-gene mean and variance
* Aggregated uncertainty per perturbation and per context

#### 6.2 Aleatoric uncertainty

* Diagonal Gaussian in delta space (default)
* Learned variance per gene

#### 6.3 Epistemic uncertainty

* Deep ensembles (K independent models)
* Total variance decomposition:
  * aleatoric
  * epistemic

#### 6.4 Calibration metrics

Mandatory:

* coverage vs nominal (50/80/90/95%)
* negative log-likelihood
* risk–coverage curves
* uncertainty-based OOD detection AUROC

Optional:

* conformal prediction layer

---

### 7. Baseline suite (must be implemented first)

#### 7.1 Internal baselines

* global mean delta
* per-perturbation mean delta
* per-(perturbation, context) mean delta
* ridge regression on control expression
* control-only (predict zero delta)
* latent shift (PCA shift baseline)

#### 7.2 External wrappers (optional but recommended)

* CPA
* GEARS
* CellOT

Wrappers run in isolation and export predictions for evaluation only.

---

### 8. PerturbFM model design

#### 8.1 Target decomposition (anti-systematic-variation)

Explicit additive structure:

```
prediction = basal_state
           + systematic_shift
           + perturbation_effect
```

Each component is separately parameterized and ablated.

---

#### 8.2 v0 model (no graph, probabilistic)

* Cell encoder: MLP or small Transformer
* Perturbation embedding: learned categorical
* Context embedding
* Fusion via MLP
* Output: mean + variance (delta space)

Trained with Gaussian NLL.

This model must already outperform baselines on at least one OOD regime before proceeding.

---

#### 8.3 v1 model (graph-augmented PerturbFM)

**Gene graph**

* external adjacency (STRING / pathway / learned)
* subsetting to dataset gene space

**Perturbation encoder**

* GNN message passing over gene graph
* pooling over perturbed genes
* supports multi-gene perturbations

**Prior trust gating**

* learnable edge gates
* residual path bypassing graph
* required ablation:
  * no graph
  * graph without gating
  * graph with gating

---

#### 8.4 v2 model (novelty candidate): Contextual Graph Intervention Operator (CGIO)

This v2 design is a concrete attempt to be *meaningfully* better than “additive embedding” methods while directly addressing known failure modes:

**Core idea**

*Represent perturbations as gene interventions, and predict effects via a context‑conditional operator with uncertain graph priors.*

**Inputs**

* Control expression (basal state)
* Perturbation as a gene set / mask (supports multi‑gene)
* Context + covariates (dose/time/etc.)
* One or more gene graphs (STRING/pathways/learned)

**Mechanism**

* Learn a perturbation representation by propagating the intervention mask over a (possibly gated) gene graph.
* Predict `delta` via a **low‑rank, context‑conditional operator** applied to the perturbation representation (data efficient, interpretable).
* Make graph usage **uncertainty‑aware**:
  * mixture over multiple graphs
  * per‑edge gating (trust learning)
  * report “graph reliance” diagnostics

**Uncertainty**

* Aleatoric: per‑gene variance head (delta space)
* Epistemic: deep ensembles + explicit evaluation under OOD contexts
* Optional: conformal layer on top of ensembles to guarantee coverage targets on held‑out OOD regimes

**Required ablations (for a credible claim)**

* v0 (no graph, probabilistic)
* v1 (single graph; no gating vs gating)
* v2 operator: operator without graph vs operator + fixed graph vs operator + gated/multi‑graph
* uncertainty: single model vs ensemble; with/without conformal wrapper

##### v2 implementation-ready contract

**Data contract (additive to `PerturbDataset`)**

Required (already core):

* `obs["pert_id"]`: string perturbation identifier (e.g., `"control"`, `"KLF4"`, `"KLF4+POU5F1"`)
* `obs["context_id"]`: string context identifier (cell type / cell line / condition grouping)
* `obs["batch_id"]`: string batch identifier
* `obs["is_control"]`: bool
* `var`: list of gene identifiers defining the gene order for `X_*` and `delta`

Required for v2 CGIO:

* `obs["pert_genes"]`: list-of-lists of gene identifiers (per sample), e.g. `["KLF4"]` or `["KLF4","POU5F1"]`
  * For controls: empty list `[]`
  * For unknown / missing: use empty list and treat as “no intervention signal”

Optional but strongly recommended:

* `obs["covariates"]`: dict of 1D arrays (dose/time/etc.) aligned to rows
* `obs["is_ood"]`: optional bool label used only for OOD-AUROC diagnostics (not required for training)

**Graph contract**

Graphs must be aligned to `var` order.

Recommended on-disk format for each graph:

* `graphs/<name>.npz` with keys:
  * `adjacency`: float32 array `[G, G]` (row-normalized recommended)
  * `genes`: array/list length `G` that matches `var` exactly (for validation)

Multiple-graph support:

* Provide a list of graph names in config; the model learns mixture weights and/or trust gating per graph.

##### v2 architecture (one concrete instantiation)

1) **Intervention propagation**
   * Convert `obs["pert_genes"]` → `pert_mask` `[B, G]` in gene space.
   * For each graph `m`: propagate `pert_mask` over adjacency with gated message passing to obtain `h_m` `[B, d]`.
   * Graph mixture: `h = Σ_m w_m(z,c) · h_m` where `w_m` are context/cell-conditioned mixture weights.

2) **Contextual operator**
   * Learn a small set of global gene bases `{B_k}` each `[G, d]`, with `k=1..K` (K small, e.g. 4–8).
   * Predict mixture weights `α_k(z,c)` and form an effective operator `B(z,c) = Σ_k α_k · B_k`.
   * Predict mean delta: `μ = B(z,c) @ h` (output `[B, G]`).

3) **Variance / uncertainty**
   * Aleatoric variance head: `σ² = softplus(V([z,c,h])) + ε`.
   * Epistemic: deep ensembles (K independently trained models) + evaluate OOD uncertainty separation.
   * Optional conformal: calibrate prediction intervals on held-out OOD validation splits.

This design supports clear ablations:
* turn off graph propagation → `h` from a simple embedding of `pert_id` or direct mask pooling
* turn off gating → fixed adjacency
* turn off mixture → single graph
* turn off operator conditioning → global operator only (no `z,c` dependence)

##### Synthetic regime (fast CI + debugging)

Update the synthetic generator so v2 is testable quickly:

* Create a sparse random gene graph adjacency.
* Assign each `pert_id` a random gene set (1–3 genes) and store it in `obs["pert_genes"]` per row.
* Generate ground-truth deltas by propagating the mask over the graph and applying a context-specific linear operator, so:
  * baselines are non-trivial but imperfect
  * graph-aware operator models have a real signal to learn

##### Exact ablation configs (first pass)

Define a minimal config schema (YAML or CLI flags) so runs are comparable:

* **Baselines**
  * `baseline.global_mean`
  * `baseline.per_perturbation_mean`
  * `baseline.per_perturbation_context_mean`
  * `baseline.ridge(alpha=...)`

* **PerturbFM v0**
  * `v0.full` (basal+context+perturbation embeddings)
  * `v0.no_context`, `v0.no_basal`, `v0.no_perturbation` (sanity ablations)

* **PerturbFM v1**
  * `v1.no_graph`
  * `v1.graph_no_gating`
  * `v1.graph_with_gating`

* **PerturbFM v2 (CGIO)**
  * `v2.no_graph` (mask pooling only)
  * `v2.single_graph_fixed` (use_graph=1, gating=off)
  * `v2.single_graph_gated` (use_graph=1, gating=on)
  * `v2.multi_graph_gated` (multi-graph mixture + gating)
  * `v2.operator_global` (operator not conditioned on z/c)
  * `v2.operator_contextual` (operator conditioned on z/c)

* **Uncertainty**
  * `uncertainty.single`
  * `uncertainty.ensemble(K=5)`
  * `uncertainty.ensemble_conformal(K=5, coverage=[0.5,0.8,0.9,0.95])`

### 9. Training and experiment control

* deterministic seeding
* mixed precision optional
* gradient clipping
* early stopping on **OOD validation metric**, not i.i.d loss
* full config resolution saved per run

Each run produces:

* resolved config
* split hash
* metrics.json
* predictions.npz
* calibration.json
* report.html

---

### 10. Validation checklist (acceptance criteria)

**scPerturBench**

* i.i.d and OOD context results
* full 6-metric panel
* uncertainty calibration plots
* comparison to baselines and PerturbFM ablations

**PerturBench**

* covariate transfer results
* combo generalization
* rank-metric tables
* collapse diagnostics

**Uncertainty**

* higher epistemic uncertainty under OOD
* monotonic risk–coverage curves
* no trivial over-inflation of variance

---

### Claim Check (with tags)

* PerturbFM plan directly targets known benchmark failure modes. **[Inference]**
* OOD context generalization with calibrated uncertainty is not currently solved in a dominant way. **[Inference]**
* This plan is sufficient to produce a publishable negative or positive result. **[Inference]**

---

### 11. Implementation status (as of 2025-12-23)

**Implemented in-repo**

* ✅ Repo scaffold, packaging, CLI, hashing/logging/seeding utilities
* ✅ Canonical `PerturbDataset` + synthetic and memmap artifact formats
* ✅ Batch iterators + minibatch training for v0/v1/v2/v3
* ✅ Context‑OOD split generator + hash‑locked split store (`splits/`)
* ✅ Baselines (including control-only + latent-shift) + evaluator artifacts
* ✅ PerturbFM v0/v1/v2 + v3 (state-dependent CGIO) with tests
* ✅ Metric panels implemented + parity harness (`scripts/validate_metrics.py`)
* ✅ PerturBench/scPerturBench adapters (external-only loaders) + split import
* ✅ Suite runners + scorecards (PerturBench + scPerturBench)
* ✅ SLURM templates + pretraining script and checkpoint loading
* ✅ Sweep launcher + leaderboard aggregator (val-only selection)

**Incomplete / TODO**

* ⚠️ Official metric parity vs external reference scripts (requires local `third_party/` repos)
* ⚠️ Training control: early stopping on OOD validation metric + richer reports
* ⚠️ Large-scale pretraining beyond simple denoising autoencoder (DDP + resume)

### 12. Updated next steps

1. Validate metric implementations numerically against official benchmark scripts (executed from `third_party/`; no GPL code vendored).
2. Use official PerturBench splits and run suite/leaderboard end-to-end on real data.
3. Expand split specs beyond Context‑OOD (LOCO CV, combo generalization, covariate transfer) and import them into SplitStore.
4. Upgrade training/eval control: OOD‑metric early stopping, deterministic run IDs, richer `report.html`.
5. Scale pretraining (DDP + resume) and evaluate v3 vs v2 under strict OOD regimes with ablations.
