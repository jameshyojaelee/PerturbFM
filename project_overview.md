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

Codex should generate this exact skeleton.

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

## Gaps & Immediate Next Steps

1. Feed this plan to Codex to:
   * generate repo skeleton
   * implement canonical dataset + adapters
2. Implement **baseline suite before any deep model**
3. Validate scPerturBench metrics numerically against reference scripts
4. Only then begin PerturbFM v0 training

If you want, next I can:

* generate a **Codex-ready mega-prompt** for bootstrapping the repo
* draft a **Methods section** as if for a bioRxiv paper
* design **figures and tables** that reviewers will expect
