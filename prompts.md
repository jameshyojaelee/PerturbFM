# Codex Mega-Prompts — PerturbFM bootstrap

---

## Prompt 1 — Scaffold the repo (exact skeleton + packaging)

```text
You are an agentic coding assistant working inside the `PerturbFM` repo.

Goal
- Bootstrap the repository into the exact skeleton described in `project_overview.md` (section “Repository architecture (PerturbFM)”).
- Make the package pip-installable (`pip install -e .`) with a working CLI entrypoint.

Hard rules (do not violate)
1) OOD splits are immutable artifacts: versioned, hash-locked, stored; experiments must log split hash; no silent regeneration.
2) No single-metric reporting: evaluation code must be structured so “partial metrics” outputs are considered invalid.
3) Uncertainty must be evaluated: calibration + OOD uncertainty separation will be mandatory later; design APIs accordingly.
4) Licensing isolation: scPerturBench (GPL-3.0) is external-only (clone into `third_party/`, gitignored). Do NOT vendor or copy GPL code into core package.

Scope
- Create the full directory/file skeleton exactly as specified.
- Provide minimal-but-clean implementations for:
  - `src/perturbfm/cli.py` (CLI skeleton)
  - `src/perturbfm/utils/{seeds.py,logging.py,hashing.py}`
  - `src/perturbfm/config/*.yaml` (reasonable defaults, but keep minimal)
- Create scripts in `scripts/` as safe placeholders (they can clone benchmarks later, but should not auto-run downloads).
- Add `.gitignore` that ignores `runs/` and `third_party/` (and typical Python junk).
- Create minimal tests that validate core invariants you implement in this prompt (e.g., hashing stability).

Important constraints
- Do not add heavy dependencies without asking first. Prefer stdlib + `numpy` + `torch` + `pyyaml` + `pytest` only if truly needed.
- Keep implementations small, but real (no empty files unless unavoidable).
- Keep the repo permissively licensed by default (do not introduce GPL code or links into the library).

Implementation details
1) Build system / packaging:
   - Use `pyproject.toml` with a `src/` layout (`src/perturbfm/...`).
   - Provide a console script entrypoint: `perturbfm = perturbfm.cli:main`.
   - Make `pytest` runnable.
2) CLI:
   - Use stdlib `argparse` (avoid extra deps).
   - Subcommands (stubs are fine): `doctor`, `config`, `data`, `splits`, `train`, `eval`.
   - Ensure each subcommand has its own `--help`.
3) Hashing utilities:
   - Implement a stable `sha256` hash for “JSON-able” objects with deterministic key ordering.
   - This will be used for split hashes and config hashes later.
4) Logging utilities:
   - Provide a logger helper that writes both to stdout and optionally to a run directory.
5) Seeding utilities:
   - Implement deterministic seeding for `random`, `numpy`, and `torch` (and CUDA if available).

Verification (run these commands)
- `python -m pip install -e .`
- `python -c "import perturbfm; print('ok')"`
- `perturbfm --help`
- `pytest -q`

Deliverables checklist
- All files/dirs in the exact skeleton exist.
- `runs/` and `third_party/` exist and are gitignored.
- CLI works and imports package successfully.
- Tests pass.
```

---

## Prompt 2 — Canonical dataset abstraction + registry + adapters (PerturBench first)

```text
You are an agentic coding assistant working inside the `PerturbFM` repo.

Goal
- Implement the canonical data abstraction described in `project_overview.md` (“Canonical object: PerturbDataset”).
- Implement the dataset registry + first adapter scaffold (PerturBench), plus a safe stub for scPerturBench.

Hard rules to enforce in code structure
- Adapters MUST map into `PerturbDataset` (single canonical representation).
- scPerturBench remains external-only (no vendoring; no importing their code).

Scope (implement in these files)
- `src/perturbfm/data/canonical.py`
  - Define `PerturbDataset` (dataclass or lightweight class).
  - Fields (as in overview): `X_control`, `X_pert`, `delta`, `obs`, `var`, `metadata`.
  - `obs` may be a pandas DataFrame or a dict of 1D arrays; pick one and be consistent throughout.
  - Provide `validate()` that checks shapes, required columns in `obs`, and delta consistency when both X_control/X_pert are present.
  - Provide convenience helpers: `.n_obs`, `.n_genes`, `.to(device)`, and slicing by indices.
- `src/perturbfm/data/registry.py`
  - Implement a registry that can `register(name, loader_fn)` and `load(name, **kwargs)`.
  - Add a built-in “synthetic” dataset generator used for unit tests and for `perturbfm doctor` / smoke tests.
- `src/perturbfm/data/transforms.py`
  - Implement basic transforms used across adapters: matching controls, computing deltas, standardization utilities (minimal).
- `src/perturbfm/data/adapters/perturbench.py`
  - Implement a `PerturBenchAdapter` that can:
    - load from a local path (do not auto-download by default)
    - map to `PerturbDataset`
    - optionally read “official splits” if present (design the API; actual integration can be TODO if benchmark not available locally)
- `src/perturbfm/data/adapters/scperturbench.py`
  - Create a stub adapter with clear errors explaining licensing isolation and how to place external code/data under `third_party/`.

CLI integration
- Add a CLI command: `perturbfm data make-synth --out <path>` that writes a small synthetic dataset artifact (e.g., `.npz` + `.json`), and `perturbfm data info <path>` to inspect it.
- Keep the artifact format simple and documented.
- Specify a minimal schema in docs and CLI output (example):
  - `data.npz` keys: `X_control`, `X_pert`, `delta` (all float32, shape `[N, G]`), and optionally `obs_idx` (int64, shape `[N]`).
  - `meta.json` keys: `obs` (table with required columns), `var` (gene identifiers list), `metadata` (freeform dict).

Tests
- Add unit tests that:
  - construct a small synthetic `PerturbDataset`
  - validate required columns (`pert_id`, `context_id`, `batch_id`, `is_control`, `covariates` optional/allowed)
  - verify slicing preserves alignment
  - verify `delta = X_pert - matched_control` invariant when applicable

Dependencies
- Prefer `numpy` + `torch` + `pandas` (only if needed for `obs`). If you add new deps, ask first.

Verification
- `pytest -q`
- `perturbfm data make-synth --help`
```

---

## Prompt 3 — Unified split system (hash-locked, stored, no silent regen)

```text
You are an agentic coding assistant working inside the `PerturbFM` repo.

Goal
- Implement the unified split system described in `project_overview.md` (“Unified split system (critical to claim validity)”).
- Splits must be immutable, hash-locked artifacts; every run must log the split hash.

Scope
- `src/perturbfm/data/splits/split_spec.py`
  - Implement a `Split` dataclass with:
    - `train_idx`, `val_idx`, `test_idx` (1D integer arrays)
    - `ood_axes` (dict with keys like `context`, `perturbation`, `dataset`)
    - `seed`
    - `frozen_hash`
  - Provide:
    - `compute_hash()` using your stable hashing utility (must depend on indices + ood_axes + seed + split spec metadata)
    - `freeze()` that sets `frozen_hash` and prevents further mutation (enforce immutability defensively)
    - `assert_frozen()` helpers
- `src/perturbfm/data/splits/split_store.py`
  - Implement `SplitStore` that:
    - saves a frozen split to disk in a deterministic JSON format
    - loads by hash
    - refuses to overwrite existing hashes with different contents
  - Decide and document where splits live. Default should be a tracked repo directory (use repo-root `splits/`) so they are versioned; do NOT put them only under `runs/` (gitignored) unless you also provide an explicit “export to tracked store” path.
- `tests/test_splits.py`
  - Add tests that verify:
    - identical splits (different in-memory ordering) hash identically
    - modifying any index changes the hash
    - store refuses overwrites with mismatched content
    - freeze prevents mutation (or at least detects mutation via hash mismatch)

Split generators (minimal but real)
- Add a module-level function (location is your choice, but keep within `src/perturbfm/data/splits/`) that can create:
  - Context-OOD split: hold out one or more `context_id` values for test, with perturbations shared across train/test.
  - Leave-one-context-out CV utility (can be a generator yielding splits).

CLI integration
- Add `perturbfm splits create` to generate and store splits from a dataset artifact (synthetic format from Prompt 2 is OK).
- Add `perturbfm splits show <hash>` to print split summary.
- Ensure the CLI prints the frozen split hash and that it’s easy to copy/paste.

Verification
- `pytest -q`
- Smoke test on synthetic dataset:
  - `perturbfm data make-synth --out /tmp/pfm_synth`
  - `perturbfm splits create --data /tmp/pfm_synth --spec context_ood --holdout-context C1`
```

---

## Prompt 4 — Baseline suite first + run artifact contract

```text
You are an agentic coding assistant working inside the `PerturbFM` repo.

Goal
- Implement the baseline suite BEFORE any deep model, per `project_overview.md`.
- Implement a minimal training/evaluation harness that produces the required run artifacts:
  - resolved config
  - split hash
  - metrics.json
  - predictions.npz
  - calibration.json (even for baselines; can be simple)
  - report.html can be a stub for now (or minimal HTML)

Hard rules
- No single-metric reporting: evaluation must produce the full metric panel outputs (even if some are placeholders for now, it must be explicit and tracked).
- Every run must log the split hash in outputs.

Scope
- Baselines in `src/perturbfm/models/baselines/`:
  - `mean_delta.py`: implement:
    - global mean delta
    - per-perturbation mean delta
    - per-(perturbation, context) mean delta
  - `ridge_delta.py`: implement ridge regression on control expression to predict delta (closed-form or iterative; avoid adding `sklearn` unless you ask first).
- Training harness in `src/perturbfm/train/`:
  - `trainer.py`: unify “fit + predict” for baselines (and later deep models)
  - `losses.py`, `optim.py`: minimal placeholders with correct APIs (more will come later)
- Evaluation harness in `src/perturbfm/eval/`:
  - `evaluator.py`: loads dataset + split, runs model, saves artifacts under `runs/<run_id>/`
  - `diagnostics.py`: include basic collapse diagnostics hooks (can be minimal)
  - `report.py`: generate a minimal HTML report that links/summarizes metrics and calibration

Prediction format contract
- Save `predictions.npz` containing at least:
  - `mean` (N, G)
  - `var` (N, G)  (for baselines, estimate from training residuals or use a small epsilon, but be explicit)
  - `idx` (N,) indices corresponding to dataset rows
- Save `metrics.json` and `calibration.json` with structured keys (document in README if needed).

Tests
- Add tests that run the full baseline pipeline on the synthetic dataset:
  - generate a split
  - fit baseline
  - predict on test
  - ensure artifacts exist and have correct shapes

CLI integration
- Add `perturbfm train baseline ...` and `perturbfm eval ...` commands (or a combined `run` command).
- The command must require a split hash (do not allow implicit split regeneration).
- Fail fast if the split hash is missing or not found in the split store.
- Use a deterministic `run_id` pattern like `<UTCYYYYMMDD-HHMMSS>_<splitHash7>_<modelName>` to keep runs sortable and traceable.

Verification
- `pytest -q`
- End-to-end smoke:
  - `perturbfm data make-synth --out /tmp/pfm_synth`
  - `perturbfm splits create --data /tmp/pfm_synth --spec context_ood --holdout-context C1`
  - `perturbfm train baseline --data /tmp/pfm_synth --split <HASH> --baseline per_perturbation_mean`
```

---

## Prompt 5 — Metrics panel + uncertainty calibration (numerical validation hooks)

```text
You are an agentic coding assistant working inside the `PerturbFM` repo.

Goal
- Implement the metric panel exactly as required by `project_overview.md`:
  - scPerturBench metrics: MSE, PCC-delta, Energy distance, Wasserstein distance, KL divergence, Common-DEGs
  - PerturBench metrics: RMSE, rank-based metrics (collapse-sensitive), prediction variance diagnostics
- Implement uncertainty/calibration metrics:
  - coverage vs nominal (50/80/90/95%)
  - NLL
  - risk–coverage curves
  - uncertainty-based OOD detection AUROC

Important: correctness over completeness
- If any metric definition is ambiguous, implement it behind an interface and add TODO + a numerical validation harness that can be run against the official benchmark scripts.
- Do NOT silently invent definitions without marking them.
- If you need SciPy/POT or other heavy deps for distances, ask before adding them; prefer lightweight implementations.

Scope
- `src/perturbfm/eval/metrics_scperturbench.py`
- `src/perturbfm/eval/metrics_perturbench.py`
- `src/perturbfm/eval/uncertainty_metrics.py`
- `src/perturbfm/eval/report.py` (extend to include metric tables + calibration plots placeholders)
- `tests/test_metrics.py`, `tests/test_uncertainty.py`

Metric API
- Implement metrics as pure functions operating on numpy arrays / torch tensors (choose one; be consistent).
- Provide an aggregation function that returns:
  - per-perturbation
  - per-context
  - global weighted mean
- Ensure evaluator always writes a single combined `metrics.json` that includes all required panels, not partial results.

Numerical validation hooks
- Add `src/perturbfm/eval/diagnostics.py` (or a `scripts/validate_metrics.py`) that can:
  - load a small prediction artifact + ground truth
  - compute metrics via PerturbFM code
  - (optionally) call out to external benchmark reference scripts located under `third_party/` if present
  - compare and print deltas / fail if mismatch beyond tolerance
- This MUST NOT copy benchmark code into the library. It can only *execute* external scripts if the user has cloned them.

Tests
- Add small, deterministic unit tests for each metric where definitions are unambiguous (MSE, RMSE, PCC, NLL, coverage).
- For ambiguous metrics (e.g., “Common-DEGs”, rank metrics), add interface-level tests + TODO markers, and ensure they are still wired into the report and metrics.json output.

Verification
- `pytest -q`
- Run a baseline end-to-end and confirm:
  - `metrics.json` includes the scPerturBench panel + PerturBench panel + uncertainty panel keys.
  - `calibration.json` exists and includes nominal coverages.
```

---

## Optional Prompt 6 — PerturbFM v0 (probabilistic, no graph)

```text
You are an agentic coding assistant working inside the `PerturbFM` repo.

Goal
- Implement PerturbFM v0 as specified:
  - cell encoder (MLP or small Transformer)
  - learned perturbation embedding + context embedding
  - fusion MLP
  - probabilistic output in delta space: per-gene mean + variance
  - train with Gaussian NLL
- Enforce the additive decomposition:
  prediction = basal_state + systematic_shift + perturbation_effect
  with each component separately parameterized and ablatable.

Scope
- `src/perturbfm/models/perturbfm/` modules listed in `project_overview.md`:
  - `cell_encoder.py`, `perturb_encoder.py`, `fusion.py`, `probabilistic_head.py`, `model.py`
- `src/perturbfm/train/trainer.py` enhancements to support torch training loops.
- `src/perturbfm/models/uncertainty/ensembles.py` initial implementation for K independent models and variance decomposition.

Acceptance criteria
- On synthetic data (where the true delta is learnable), v0 improves over at least one baseline on the Context-OOD split (define a simple synthetic regime that makes this meaningful).
- Evaluator produces the full metrics + calibration outputs for v0.

Verification
- `pytest -q`
- A short smoke training run completes on CPU in <1 minute on synthetic data.
```

---

## Optional Prompt 7 — PerturbFM v1 (graph-augmented + trust gating)

```text
You are an agentic coding assistant working inside the `PerturbFM` repo.

Goal
- Implement the graph-augmented PerturbFM v1:
  - gene graph (external adjacency; subsetting to dataset genes)
  - perturbation encoder with GNN message passing
  - prior trust gating (learnable edge gates + residual bypass)
  - required ablations: no-graph, graph-no-gating, graph-with-gating

Scope
- `src/perturbfm/models/perturbfm/gene_graph.py`
- `src/perturbfm/models/perturbfm/perturb_encoder.py` (GNN)
- Update training/eval to support ablations via config.

Constraints
- Do not introduce heavyweight GNN deps without asking first (prefer implementing a small message passing layer in torch).

Verification
- Unit tests for graph subsetting and gating behavior.
```
