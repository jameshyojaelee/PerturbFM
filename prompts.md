# Codex Mega-Prompts — PerturbFM roadmap (post-bootstrap)

Copy/paste these prompts into Codex **in order**. These are designed to follow the updated direction in:

- `project_overview.md` (see v2 CGIO spec in section 8.4)
- `current_state.md` (exec summary + capability matrix)
- `README.md` (current repo status + known gaps)

Assumption: the repo scaffold + baseline v0/v1 code already exists (i.e., the “bootstrap” is done). These prompts focus on what’s still needed for credible benchmarking + a novel model.

---

## Prompt 0 — Preflight: repo health + test runner + invariants

```text
You are an agentic coding assistant working inside the `PerturbFM` repo.

Read first
- `project_overview.md` (hard rules + v2 CGIO spec)
- `current_state.md` (failure modes + what “superior” means)
- `README.md` (known gaps)

Goal
- Make it easy to run a reliable “repo health” check in diverse environments.
- Enforce key invariants early (no silent split regen, no partial metric reporting).

Scope
1) Add a `scripts/check.sh` (or Makefile target) that runs:
   - import smoke (`python -c "import perturbfm; print('ok')"`)
   - CLI smoke (`perturbfm --help`)
   - unit tests (see #2)
2) Some environments crash running `pytest` due to plugin autoload / SSL issues (e.g., `ImportError: libssl.so.1.1` via third-party plugins).
   - Add a safe default test command that sets `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.
   - Document this in `README.md` (short).
3) Add “invariant assertions” tests:
   - Split store refuses overwrite with mismatched content.
   - Evaluator refuses to write `metrics.json` if required metric keys are missing (no partial panels).

Constraints
- Do not add heavy dependencies.
- Keep changes small and targeted.

Verification
- Run `scripts/check.sh` (or your new Makefile target).
```

---

## Prompt 1 — Metrics: implement exact definitions + numerical parity harness

```text
You are an agentic coding assistant working inside the `PerturbFM` repo.

Hard rules (do not violate)
- No single-metric reporting: scPerturBench panel and PerturBench panel must be emitted together; missing metrics must fail the run, not silently write NaNs.
- Licensing isolation: scPerturBench is external-only; do NOT vendor GPL code.

Goal
- Replace TODO/NaN metric placeholders with correct implementations OR (if definition is ambiguous) lock them behind a validation harness that compares to official reference scripts in `third_party/`.
- Add a numerical parity harness that can be run locally when reference repos are present.

Scope
1) Implement scPerturBench metric panel in `src/perturbfm/eval/metrics_scperturbench.py`:
   - MSE (already), PCC-delta (already)
   - Energy distance, Wasserstein distance, KL divergence, Common-DEGs
   - Aggregations: per perturbation, per context, global weighted mean
2) Implement PerturBench metric panel in `src/perturbfm/eval/metrics_perturbench.py`:
   - RMSE (already)
   - Rank-based metrics (collapse-sensitive)
   - Prediction variance diagnostics + collapse diagnostics
3) Implement / extend uncertainty metrics in `src/perturbfm/eval/uncertainty_metrics.py`:
   - coverage vs nominal (50/80/90/95%)
   - NLL
   - risk–coverage curves (already roughly present; verify definition)
   - OOD AUROC (must accept explicit ood_labels)
4) Add `scripts/validate_metrics.py` that:
   - loads a dataset artifact + a predictions artifact
   - computes metrics using PerturbFM code
   - if `third_party/scPerturBench` or `third_party/PerturBench` exists, runs their official metric scripts (as a subprocess) on the same inputs
   - compares results and fails if mismatch > tolerance
   - IMPORTANT: this script may *execute* external code but must not copy/vend it into `src/perturbfm/`.
5) Enforce “no partial panels”:
   - evaluator must refuse to write `metrics.json` unless all required keys exist.
   - If a metric is still unimplemented, it must raise a clear error telling the user to run `scripts/validate_metrics.py` after cloning the benchmark.

Dependencies
- Ask before adding SciPy/POT/scanpy/etc. Prefer numpy/torch implementations where feasible.

Verification
- Run the unit tests.
- Create a synthetic dataset + run a baseline; confirm evaluator now fails loudly if any required metrics are missing.
```

---

## Prompt 2 — Benchmark adapters: PerturBench real loader + official splits (no silent regen)

```text
You are an agentic coding assistant working inside the `PerturbFM` repo.

Hard rules
- OOD splits are immutable artifacts: import official splits where available; log and store split hashes; never silently regenerate.
- Licensing isolation: scPerturBench remains external-only.

Goal
- Implement a real PerturBench adapter that loads canonical data and imports official splits when available.

Scope
1) `src/perturbfm/data/adapters/perturbench.py`
   - Implement `PerturBenchAdapter.load(...)` to load real PerturBench datasets (AnnData / torch datasets).
   - Map into `PerturbDataset` including `obs` fields and `var`.
   - Support covariates (dose/time/etc.) via `obs["covariates"]` dict.
2) Official splits
   - If PerturBench provides official splits, implement `load_official_splits()` to import them.
   - Convert them into `Split` artifacts and store them in the split store.
   - Ensure the split hash is deterministic and logged.
3) CLI
   - Add `perturbfm data import-perturbench --dataset <name> --out <dir>` to write a `PerturbDataset` artifact.
   - Add `perturbfm splits import-perturbench --dataset <name>` to import/store official splits.

Dependencies
- Ask before adding scanpy/anndata. If you do add them, make them optional extras (e.g. `pip install -e .[bench]`) and keep core install lightweight.

Verification
- Add adapter unit tests that run when the benchmark is not present (skip with a clear message).
- Add a smoke path that works with a small local artifact.
```

---

## Prompt 3 — Split specs expansion (LOCO CV, covariate transfer, combo generalization)

```text
You are an agentic coding assistant working inside the `PerturbFM` repo.

Hard rules
- Splits must be hash-locked and stored; do not allow implicit regeneration in training/eval.

Goal
- Expand split specs beyond Context-OOD so claims match benchmarks and stress tests.

Scope
1) Add split generators in `src/perturbfm/data/splits/`:
   - leave-one-context-out CV (generator yielding splits)
   - perturbation-OOD split (hold out perturbations)
   - combo generalization split (hold out perturbation combinations)
   - covariate transfer split (dose/time bucket held-out, if covariates exist)
2) CLI updates
   - Extend `perturbfm splits create` to support these specs and store the resulting frozen split.
   - Ensure `perturbfm train ...` and `perturbfm eval ...` require split hash and fail if not found.
3) Tests
   - Add unit tests verifying each generator produces disjoint indices and consistent hashing.

Verification
- Unit tests.
- Synthetic dataset smoke.
```

---

## Prompt 4 — Uncertainty contract: ensembles + conformal + calibration outputs

```text
You are an agentic coding assistant working inside the `PerturbFM` repo.

Goal
- Turn “uncertainty” into a first-class deliverable, per `current_state.md` and `project_overview.md`.
- Implement a concrete contract and ensure evaluator/report always emits it.

Hard rules
- Calibration must be evaluated; uncertainty must separate OOD from IID empirically (when labels/splits exist).

Scope
1) Implement:
   - `src/perturbfm/models/uncertainty/calibration.py`: calibration utilities (ECE-like bins, coverage tables, reliability plots data)
   - `src/perturbfm/models/uncertainty/conformal.py`: conformal intervals for per-gene regression (start with split conformal on residuals)
   - Extend `src/perturbfm/models/uncertainty/ensembles.py` to support training K models and aggregating predictions.
2) Evaluation outputs
   - Ensure `calibration.json` always contains:
     - coverage@{50,80,90,95}
     - NLL
     - risk–coverage curve points
     - OOD AUROC (when `ood_labels` exist)
3) CLI
   - Add flags for ensemble size and optional conformal calibration.
4) Tests
   - Deterministic unit tests for conformal coverage on synthetic noise.

Dependencies
- Avoid heavy deps; implement with numpy/torch.

Verification
- Unit tests.
```

---

## Prompt 5 — Implement PerturbFM v2 (CGIO): gene-set interventions + uncertain graph priors

```text
You are an agentic coding assistant working inside the `PerturbFM` repo.

Read first
- `project_overview.md` section 8.4 (v2 CGIO implementation-ready contract + ablations)

Goal
- Implement PerturbFM v2 (CGIO): context-conditioned low-rank operator + graph-propagated intervention signal + uncertainty-aware graph trust gating.

Non-negotiables
- Must accept perturbations as gene sets / masks (`obs["pert_genes"]`).
- Must support multi-gene perturbations.
- Must ship with an ablation-friendly config/CLI surface.

Scope
1) Data plumbing
   - Extend the synthetic generator in `src/perturbfm/data/registry.py` to emit:
     - `obs["pert_genes"]` (list-of-lists)
     - a sparse random adjacency stored in metadata or written as a `graphs/*.npz` artifact for tests
   - Add helpers to convert `pert_genes` to `pert_mask [B, G]` aligned to `var`.
2) Model code
   - Add `src/perturbfm/models/perturbfm/cgio.py` (or similar) implementing:
     - graph propagation encoder (single + multi-graph)
     - trust gating (fixed vs gated)
     - contextual operator (global basis + contextual mixture weights)
     - mean + variance outputs in delta space
3) Training
   - Extend `src/perturbfm/train/trainer.py` with `fit_predict_perturbfm_v2(...)`.
   - Add CLI: `perturbfm train perturbfm-v2 ...` with flags to toggle:
     - graph on/off
     - gating on/off
     - single vs multi-graph
     - operator conditioning on/off
4) Evaluation
   - Ensure evaluator writes full metric panels + calibration for v2 runs.
5) Tests
   - Add a fast CPU smoke test showing v2 can learn the synthetic regime.
   - Add a test that multi-gene perturbations do not crash and produce correct shapes.

Dependencies
- Do not add a GNN library; implement message passing directly in torch.

Verification
- Unit tests.
- Quick end-to-end smoke with synthetic dataset + Context-OOD split.
```

---

## Prompt 6 — Ablation runner + report upgrades (make claims reproducible)

```text
You are an agentic coding assistant working inside the `PerturbFM` repo.

Goal
- Make the “ablation grid” in `project_overview.md` executable and comparable across runs.

Scope
1) Config-driven runs
   - Add a simple config loader (YAML) that resolves defaults + overrides and writes a resolved config to the run directory.
   - Ensure every run logs: split hash, config hash, git SHA (if available), and dataset artifact hash.
2) Deterministic run IDs
   - Standardize run_id: `<UTCYYYYMMDD-HHMMSS>_<splitHash7>_<modelName>_<configHash7>`
3) Report upgrades
   - Extend `report.html` to include:
     - metric tables for scPerturBench + PerturBench panels
     - calibration tables (coverage, NLL)
     - risk–coverage curve data (plot optional)
     - graph reliance diagnostics (for v1/v2)
4) Add `scripts/run_ablations.py`
   - Given a dataset artifact + split hash + a list of configs, run the full ablation grid and write a summary table (CSV/JSON).
   - Fail fast if any run produces partial metrics.

Verification
- Run a small ablation set on synthetic data and confirm summary outputs.
```
