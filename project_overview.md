# PerturbFM

**A Universal Perturbation Foundation Model with Graph Priors and Calibrated Uncertainty**

---

## 0. Non-negotiable design constraints (hard rules)

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

## 1. Repository architecture (PerturbFM)

Codex should generate this baseline skeleton (additional files are OK as long as constraints are upheld).

```
perturbfm/
  README.md
  pyproject.toml

  src/perturbfm/
    cli.py
    __init__.py

    config/
      defaults.yaml
      data.yaml
      model.yaml
      eval.yaml

    data/
      registry.py
      canonical.py
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
      perturbfm/
        cell_encoder.py
        perturb_encoder.py
        gene_graph.py
        fusion.py
        probabilistic_head.py
        model.py
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

  tests/
    test_splits.py
    test_metrics.py
    test_uncertainty.py

  runs/          # gitignored
  third_party/   # external benchmark clones (gitignored)
  splits/        # tracked immutable split artifacts
```

---

## 2. Canonical data abstraction (shared across benchmarks)

### 2.1 Canonical object: `PerturbDataset`

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

## 3. Benchmark adapters

### 3.1 PerturBench adapter (first implementation target)

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

### 3.2 scPerturBench adapter

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

## 4. Unified split system (critical to claim validity)

### 4.1 Split object

Each split stores:

* `train_idx`, `val_idx`, `test_idx`
* `ood_axes`: `{context, perturbation, dataset}`
* `seed`
* `frozen_hash`

### 4.2 Required split specs

#### Core (headline claim)

* **Context-OOD split**

  * hold out one or more `context_id`s
  * perturbations remain shared across train/test

#### Secondary (stress tests)

* leave-one-context-out CV
* PerturBench covariate transfer
* combo generalization (Norman-style)

---

## 5. Metric implementation (no deviations allowed)

### 5.1 scPerturBench metrics

All must be implemented exactly:

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

### 5.2 PerturBench metrics

Required:

* RMSE
* rank-based metrics (collapse-sensitive)
* prediction variance diagnostics

---

## 6. Uncertainty and calibration (central contribution)

### 6.1 Outputs

* Per-gene mean and variance
* Aggregated uncertainty per perturbation and per context

### 6.2 Aleatoric uncertainty

* Diagonal Gaussian in delta space (default)
* Learned variance per gene

### 6.3 Epistemic uncertainty

* Deep ensembles (K independent models)
* Total variance decomposition:

  * aleatoric
  * epistemic

### 6.4 Calibration metrics

Mandatory:

* coverage vs nominal (50/80/90/95%)
* negative log-likelihood
* risk–coverage curves
* uncertainty-based OOD detection AUROC

Optional:

* conformal prediction layer

---

## 7. Baseline suite (must be implemented first)

### 7.1 Internal baselines

* global mean delta
* per-perturbation mean delta
* per-(perturbation, context) mean delta
* ridge regression on control expression

### 7.2 External wrappers (optional but recommended)

* CPA
* GEARS
* CellOT

Wrappers run in isolation and export predictions for evaluation only.

---

## 8. PerturbFM model design

### 8.1 Target decomposition (anti-systematic-variation)

Explicit additive structure:

```
prediction = basal_state
           + systematic_shift
           + perturbation_effect
```

Each component is separately parameterized and ablated.

---

### 8.2 v0 model (no graph, probabilistic)

* Cell encoder: MLP or small Transformer
* Perturbation embedding: learned categorical
* Context embedding
* Fusion via MLP
* Output: mean + variance (delta space)

Trained with Gaussian NLL.

This model must already outperform baselines on at least one OOD regime before proceeding.

---

### 8.3 v1 model (graph-augmented PerturbFM)

#### Gene graph

* external adjacency (STRING / pathway / learned)
* subsetting to dataset gene space

#### Perturbation encoder

* GNN message passing over gene graph
* pooling over perturbed genes
* supports multi-gene perturbations

#### Prior trust gating

* learnable edge gates
* residual path bypassing graph
* required ablation:

  * no graph
  * graph without gating
  * graph with gating

---

### 8.4 v2 model (novelty candidate): Contextual Graph Intervention Operator (CGIO)

This v2 design is a concrete attempt to be *meaningfully* better than “additive embedding” methods while directly addressing the failure modes summarized in `current_state.md`:

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

#### v2 implementation-ready contract

This subsection is intentionally concrete so engineering and evaluation don’t drift.

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

#### v2 architecture (one concrete instantiation)

Goal: express richer-than-additive effects while staying data-efficient and ablation-friendly.

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

#### Synthetic regime (fast CI + debugging)

Update the synthetic generator so v2 is testable quickly:

* Create a sparse random gene graph adjacency.
* Assign each `pert_id` a random gene set (1–3 genes) and store it in `obs["pert_genes"]` per row.
* Generate ground-truth deltas by propagating the mask over the graph and applying a context-specific linear operator, so:
  * baselines are non-trivial but imperfect
  * graph-aware operator models have a real signal to learn

#### Exact ablation configs (first pass)

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

## 9. Training and experiment control

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

## 10. Validation checklist (acceptance criteria)

### scPerturBench

* i.i.d and OOD context results
* full 6-metric panel
* uncertainty calibration plots
* comparison to baselines and PerturbFM ablations

### PerturBench

* covariate transfer results
* combo generalization
* rank-metric tables
* collapse diagnostics

### Uncertainty

* higher epistemic uncertainty under OOD
* monotonic risk–coverage curves
* no trivial over-inflation of variance

---

## Claim Check (with tags)

* PerturbFM plan directly targets known benchmark failure modes. **[Inference]**
* OOD context generalization with calibrated uncertainty is not currently solved in a dominant way. **[Inference]**
* This plan is sufficient to produce a publishable negative or positive result. **[Inference]**

---

## 11. Implementation status (as of 2025-12-22)

Implemented in-repo:

* ✅ Repo scaffold, packaging, CLI, hashing/logging/seeding utilities
* ✅ Canonical `PerturbDataset` + synthetic artifact format
* ✅ Context‑OOD split generator + hash‑locked split store (`splits/`)
* ✅ Baselines + evaluator that writes run artifacts
* ✅ PerturbFM v0/v1 scaffolds + basic deep ensemble helper

Incomplete / TODO:

* ⚠️ Real benchmark adapters + official split import (PerturBench/scPerturBench)
* ⚠️ Metric definitions that must be validated (Energy/Wasserstein/KL/Common‑DEGs; rank metrics; collapse diagnostics)
* ⚠️ Training control: early stopping on OOD validation metric, richer reports, and strict numerical validation harnesses

## 12. Updated next steps

1. Validate metric implementations numerically against official benchmark scripts (executed from `third_party/`; no GPL code vendored).
2. Implement real benchmark adapters and import official splits (PerturBench first).
3. Expand split specs beyond Context‑OOD (LOCO CV, combo generalization, covariate transfer).
4. Upgrade training/eval control: OOD‑metric early stopping, config resolution persistence, deterministic run IDs, richer `report.html`.
5. Implement and evaluate the v2 CGIO model against strict OOD regimes with calibrated uncertainty and ablations.
