# Codex Mega-Prompts — PerturbFM roadmap

## Execution tracker (update this as you go)

Resume rule: start at the **lowest-numbered** prompt whose status is not `DONE` (unless it is explicitly marked `SKIP`).

Legend:
- `DONE`: implemented + tests passing in this repo.
- `PARTIAL`: implemented in spirit, but missing external parity/benchmark proof (or has known gaps).
- `TODO`: not implemented yet.
- `SKIP`: intentionally skipped (must include a reason in the notes).

Last known tests (repo-local): 2025-12-23 — `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q` → `43 passed in 7.61s`.

| # | Prompt (short) | Status | Evidence / next action |
|---:|---|---|---|
| 1 | Ground truth + fix plan | DONE | `docs/dev_todo.md` exists and tracks issues/fixes. |
| 2 | Fix CGIO `residual` bug | DONE | `src/perturbfm/models/perturbfm/cgio.py`; regression test `tests/test_cgio.py`. |
| 3 | Enforce metric completeness for v1 | DONE | Guard in `src/perturbfm/eval/evaluator.py`; tests in `tests/test_invariants.py`. |
| 4 | Remove conformal test leakage | DONE | `calib_idx` + derivation in `src/perturbfm/data/splits/split_spec.py`; tests in `tests/test_uncertainty.py`. |
| 5 | Valid Context-OOD split semantics | DONE | Shared-pert filtering + metadata in `src/perturbfm/data/splits/split_spec.py`; tests in `tests/test_splits.py`. |
| 6 | scPerturBench metric parity + scalability | DONE | Scalable metrics + parity harness in `metrics_scperturbench.py` / `scripts/validate_metrics.py`; tests in `tests/test_metrics_scperturbench.py`. |
| 7 | PerturBench rank/collapse metrics parity | DONE | Rank + collapse metrics implemented in `metrics_perturbench.py`; tests in `tests/test_metrics_perturbench.py`. |
| 8 | Sparse graphs + scalable gating | DONE | Sparse representations in `src/perturbfm/models/perturbfm/*`; “no dense G×G gate params” test in `tests/test_gene_graph.py`. |
| 9 | Packaging + README + harness integration | DONE | README cleaned (no missing files), smoke-run added, export helper documented; adapter errors reference `[bench]` extras. |
| 10 | Guardrail tests / invariants | DONE | `tests/test_invariants.py`, `tests/test_splits.py`, `tests/test_uncertainty.py`. |
| 11 | PerturBench contract + parity gate | DONE | Contract doc in `docs/perturbench_contract.md`; parity gate + tests in `scripts/validate_metrics.py` and `tests/test_validate_metrics.py`. |
| 12 | PerturBench dataset + official split import | DONE | CLI import + split ingestion in `src/perturbfm/cli.py` + adapter; tests in `tests/test_perturbench_import.py`. |
| 13 | PerturBench suite runner + scorecard | DONE | Suite runner in `scripts/run_perturbench_suite.py` + dry-run test in `tests/test_perturbench_suite.py`. |
| 14 | SLURM-first execution layer | DONE | SLURM templates in `scripts/slurm/` + usage doc `docs/slurm.md`. |
| 15 | Scalable data pipeline (memmap/batching) | DONE | Memmap artifacts + minibatch training in `canonical.py`/`trainer.py`; batching utils + tests. |
| 16 | Strong baselines for PerturBench | DONE | Added control-only + latent-shift baselines with tests (`test_baselines_extra.py`). |
| 17 | State-dependent CGIO (“v3”) | DONE | Added PerturbFMv3 model + trainer/evaluator/CLI and smoke test (`tests/test_perturbfm_v3.py`). |
| 18 | Large-scale pretraining + integration | DONE | Pretrain script + encoder load/freeze wiring and tests (`scripts/pretrain_cell_encoder.py`, `tests/test_pretrain.py`). |
| 19 | Full PerturBench sweep + leaderboard | DONE | Sweep launcher `scripts/launch_sweep.py` + val-only aggregator `scripts/aggregate_leaderboard.py`. |
| 20 | scPerturBench adapter (external-only) | DONE | Added adapter + import helper script + tests (`scperturbench.py`, `scripts/import_scperturbench.py`). |
| 21 | scPerturBench context-OOD suite runner | DONE | Wrapper `scripts/run_scperturbench_suite.py` + score-metric override support. |
| 22 | Context-OOD modeling + calibration upgrades | DONE | Added context-OOD ablation runner `scripts/run_context_ood_ablations.py`. |

MEGA-PROMPT 1: Establish ground truth and create a fix plan inside the repo
=========================================================================

You are working in the PerturbFM repository.

Task:
1) Read and summarize the current repository structure and key modules involved in:
   - splits: src/perturbfm/data/splits/
   - evaluator: src/perturbfm/eval/evaluator.py
   - scPerturBench metrics: src/perturbfm/eval/metrics_scperturbench.py
   - PerturBench metrics: src/perturbfm/eval/metrics_perturbench.py
   - CGIO: src/perturbfm/models/perturbfm/cgio.py
   - graph encoder: src/perturbfm/models/perturbfm/perturb_encoder.py and gene_graph.py

2) Confirm the following suspected issues by locating exact lines and writing a brief “issue record” for each:
   A) CGIO GraphPropagator references an undefined variable `residual` in forward()
   B) conformal intervals are computed using test residuals (test leakage) in evaluator.py
   C) context_ood_split does not enforce that pert_id coverage is shared across train/test
   D) run_perturbfm_v1 does not call the same metric completeness guard as other runners
   E) graph adjacency + gating are dense and will not scale to realistic gene counts
   F) scPerturBench metrics contain placeholders and at least one O(n^2) implementation (energy distance)
   G) PerturBench rank metrics are placeholders / not parity with PerturBench definitions
   H) README references a missing file (if true)
   I) pyproject lacks “bench” extras though adapter suggests installing extras

3) Create a repo-internal checklist file:
   - docs/dev_todo.md
   It must include:
   - a numbered list of fixes in P0/P1/P2 order
   - file paths
   - acceptance tests for each item
   - explicit “do not do” constraints (no test leakage, no GPL vendoring)

4) Run tests before changes:
   - PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
   Record results in docs/dev_todo.md.

Output:
- Provide the created docs/dev_todo.md contents.
- Provide a short “next recommended prompt number” suggestion (just a number).


MEGA-PROMPT 2: Fix CGIO undefined variable bug + add a regression test
====================================================================

You are working in PerturbFM.

Goal:
Fix the CGIO GraphPropagator runtime bug where `residual` is referenced but undefined.

Scope:
- src/perturbfm/models/perturbfm/cgio.py
- tests/ (add a targeted unit test)

Requirements:
1) Locate the code path:
   if torch.max(torch.abs(a)) < 1e-6:
       h_list.append(residual)
       continue
   Determine intended behavior and implement a correct replacement variable (likely the incoming h tensor or a residual connection tensor). Document the decision in an inline comment.

2) Add a unit test that deterministically triggers that branch.
   - Construct a GraphPropagator instance with adjacency weights that become ~0 (or force gating to zero).
   - Ensure the forward pass completes and output shape is correct.

3) Ensure you do NOT change public APIs.

Acceptance:
- PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q passes
- The new test fails on old code and passes after the fix.

Output:
- Show the diff for cgio.py and the new test file.
- Show pytest output summary.


MEGA-PROMPT 3: Enforce metric completeness for v1 runs
=====================================================

You are working in PerturbFM.

Goal:
Ensure run_perturbfm_v1 enforces metric completeness the same way as other runners.

Scope:
- src/perturbfm/eval/evaluator.py
- tests/ (add or update tests)

Requirements:
1) Identify how `_require_metrics_complete(metrics)` is used in:
   - run_baseline
   - run_perturbfm_v0
   - run_perturbfm_v2
   Then apply the same enforcement to run_perturbfm_v1.

2) Add a test that:
   - monkeypatches metrics computation to return an incomplete dict
   - confirms v1 raises the same error / exception as v0/v2.

Acceptance:
- pytest passes
- the test fails on previous behavior if v1 previously skipped the guard.

Output:
- Provide diff and pytest summary.


MEGA-PROMPT 4: Remove conformal calibration leakage (introduce calibration split)
================================================================================

You are working in PerturbFM.

Goal:
Eliminate test leakage: conformal calibration residuals must be computed on a held-out calibration set (not test).

Scope:
- src/perturbfm/data/splits/ (extend split object/spec)
- src/perturbfm/eval/evaluator.py
- src/perturbfm/models/uncertainty/conformal.py (if needed)
- tests/test_uncertainty.py (or create)

Design constraints:
- Keep behavior backward-compatible where possible.
- If a split does not define calib_idx, derive it deterministically from val_idx (for example, first 50% of val by stable RNG with seed and split hash).
- Never use test_idx to compute conformal quantiles.

Implementation steps:
1) Extend the split data structure to include calib_idx (optional but preferred).
2) Update split creation utilities to populate calib_idx when making splits.
3) Update evaluator:
   - compute conformal residuals using calib subset only
   - store calibration metadata showing which indices were used
   - apply intervals to test predictions
4) Add tests:
   - ensure residual quantiles are computed from calib_idx only
   - include a “guard” test: if code attempts to use test_idx for calibration, it fails

Acceptance:
- pytest passes
- calibration.json records calib_idx size and provenance
- no test leakage

Output:
- Diff summary
- Explain the calibration split derivation in 5-10 lines (in a comment or docs/dev_todo.md update)


MEGA-PROMPT 5: Make Context-OOD split valid for the headline claim
=================================================================

You are working in PerturbFM.

Goal:
For the headline claim “OOD generalization across cellular contexts,” the Context-OOD split must ensure perturbations are shared across train and test. If not, the split must:
- either fail fast
- or explicitly label the split as “mixed OOD” and optionally filter to shared perturbations

Scope:
- src/perturbfm/data/splits/split_spec.py
- src/perturbfm/data/splits/split_store.py (if required)
- tests/test_splits.py

Requirements:
1) Modify context_ood_split to accept a flag:
   require_shared_perturbations: bool = True (default True for the claim path)
2) Implement checks:
   - pert_id_test ⊆ pert_id_train must hold; otherwise:
     Option A (recommended): filter test rows to shared perturbations, and record a warning field in split metadata
     Option B: raise ValueError
   Choose A or B and justify in a short comment.
3) Record OOD axes in split metadata explicitly:
   - ood_axes should include "context" always
   - include "perturbation" only if mixed OOD happens
4) Add tests:
   - case where held-out context contains a perturbation absent in train
   - ensure behavior matches chosen policy (filter or fail)
   - ensure metadata labels the axes correctly

Acceptance:
- pytest passes
- split JSON artifacts contain enough metadata to interpret the OOD regime

Output:
- Diff and pytest summary
- Example split metadata JSON snippet printed in the prompt output


MEGA-PROMPT 6: scPerturBench metric parity and scalability upgrade
================================================================

You are working in PerturbFM.

Goal:
Replace placeholder scPerturBench metrics with implementations that are:
- scalable by default (no uncontrolled O(n^2) paths)
- testable and documented
- parity-checkable against reference scripts

Scope:
- src/perturbfm/eval/metrics_scperturbench.py
- scripts/validate_metrics.py (create or expand)
- tests/test_metrics_scperturbench.py (create or expand)
- docs/dev_todo.md (update)

Requirements:
1) For each metric listed in metrics_scperturbench.py:
   - write a docstring defining it
   - implement a scalable default strategy:
     - Energy distance: subsampling or unbiased pair sampling
     - Wasserstein: sliced Wasserstein (random projections) or justify 1D approximation
     - KL: define distributional assumption clearly
     - Common-DEGs: explicit, deterministic DEG procedure
2) Add synthetic tests:
   - identity predictor sanity
   - collapsed predictor penalty
3) Add parity validation script:
   - compare against reference metrics JSON or script output within tolerance

Constraints:
- Do not vendor GPL scPerturBench code
- Keep runtime bounded

Acceptance:
- pytest passes
- validate_metrics.py runs and prints pass/fail

Output:
- Diff summary
- Updated docs/dev_todo.md parity notes


MEGA-PROMPT 7: Implement PerturBench rank metrics and collapse diagnostics
=========================================================================

You are working in PerturbFM.

Goal:
Implement PerturBench-style rank and collapse-sensitive metrics.

Scope:
- src/perturbfm/eval/metrics_perturbench.py
- tests/test_metrics_perturbench.py
- docs/dev_todo.md

Requirements:
1) Implement rank-based metric(s) aligned with PerturBench intent.
2) Add collapse diagnostics (prediction variance, etc.).
3) Add synthetic tests:
   - perfect predictor > noisy predictor
   - collapsed predictor heavily penalized
4) Integrate metrics into evaluator output.

Acceptance:
- pytest passes
- sample metric dict printed from synthetic test

Output:
- Diff summary
- Example metrics output


MEGA-PROMPT 8: Graph scalability refactor (dense → sparse) + scalable gating
===========================================================================

You are working in PerturbFM.

Goal:
Refactor graph code to scale beyond toy gene counts.

Scope:
- src/perturbfm/models/perturbfm/gene_graph.py
- src/perturbfm/models/perturbfm/perturb_encoder.py
- src/perturbfm/models/perturbfm/cgio.py
- tests/

Requirements:
1) Implement sparse graph representation (edge_index / CSR).
2) Replace dense matmuls with sparse message passing.
3) Implement scalable gating modes:
   - none
   - scalar
   - node
   - lowrank
   - mlp
4) Add ablation flags to configs.
5) Add tests ensuring no dense GxG allocation.

Acceptance:
- pytest passes
- no dense gate matrices allocated

Output:
- Diff summary
- README snippet documenting graph format and gating modes


MEGA-PROMPT 9: Packaging + README + benchmark harness integration
================================================================

You are working in PerturbFM.

Goal:
Fix dependencies, docs, and external benchmark integration without licensing issues.

Scope:
- pyproject.toml
- README.md
- scripts/
- src/perturbfm/data/adapters/perturbench.py

Requirements:
1) Add optional dependency extras:
   - bench = ["anndata", "h5py", ...]
2) Ensure adapter error messages reference correct extras.
3) Fix README:
   - remove/add missing files
   - add minimal smoke-run instructions
   - document external scPerturBench harness usage
4) Add script to export predictions for external harness.

Acceptance:
- pip install -e ".[bench]" works
- README commands are consistent
- no GPL code vendored

Output:
- Diff summary
- Updated README sections


MEGA-PROMPT 10: Guardrail tests and invariants
=============================================

You are working in PerturbFM.

Goal:
Add guardrail tests to prevent regressions in split validity, calibration, and metrics.

Scope:
- tests/test_splits.py
- tests/test_uncertainty.py
- tests/test_metrics_scperturbench.py
- utils (optional)

Requirements:
1) Tests enforcing:
   - valid context-OOD split semantics
   - no conformal test leakage
   - metric completeness for v0/v1/v2
2) Synthetic regression tests:
   - collapsed predictor penalized
   - identity predictor near-optimal
3) Fail loudly on regression.

Acceptance:
- pytest passes
- tests fail if invariants are violated

Output:
- Test diffs
- pytest summary


===============================================================================
PERTURBENCH-FIRST + SCALE ROADMAP (new)
===============================================================================

This next sequence is optimized for:
1) Winning PerturBench tasks first (primary).
2) Then winning scPerturBench context-OOD (secondary).

Assumptions:
- You have access to SLURM with multi-node GPU partitions and ample CPU/RAM.
- Network is available to clone external benchmark repos into `third_party/`.
- scPerturBench code must remain external (GPL isolation). Do not vendor GPL code.
- For any dependency additions: ask before adding new production deps; prefer optional extras.

Operating principles (do not skip):
- Always use official PerturBench splits (or explicitly label as “approx / non-official”).
- Always maintain “no leakage” (no using test for calibration/selection).
- Always report full metric panels; fail fast if incomplete.
- Always log: data hash, split hash, config hash, git commit, seeds, and environment.

Copy/paste order (recommended):
- 11 → 12 → 13: benchmark contract + data/splits + suite runner
- 14 → 15: scaling infra (data + trainer + SLURM)
- 16 → 17: competitive baselines + model upgrades
- 18 → 19: large-scale pretraining + full sweeps
- 20 → 21 → 22: scPerturBench context-OOD integration + upgrades


MEGA-PROMPT 11: PerturBench “contract” + parity gate (official metrics/splits)
=============================================================================

You are working in PerturbFM.

Goal:
Make “PerturBench results” mean something concrete by implementing a strict contract:
- how datasets are imported
- how official splits are represented and frozen
- how official metrics are computed and validated

Scope:
- scripts/validate_metrics.py (extend into a true parity harness)
- src/perturbfm/eval/metrics_perturbench.py (update as needed for parity)
- src/perturbfm/data/adapters/perturbench.py (split discovery hooks)
- docs/ (create a short contract doc; name it sensibly)

Requirements:
1) Define a PerturBench contract doc:
   - what files/dirs are expected under `third_party/PerturBench` (or user-specified path)
   - how to import datasets into `PerturbDataset` artifacts
   - how to import official splits into `SplitStore` (hash-locked)
   - how to run official metrics and compare against our implementation
2) Upgrade `scripts/validate_metrics.py` into a “parity gate”:
   - run our metrics
   - if `third_party/PerturBench` exists, run its reference metric script(s)
   - compare key globals within tolerance; print a compact diff report
   - if parity check fails, exit non-zero (or clearly mark FAIL)
   - if external repo missing, print exact next steps, but still run internal metrics
3) Add tests that cover the parity harness structure without requiring external repos:
   - test should skip gracefully when third_party is absent
   - ensure script output includes “external missing” message when absent

Constraints:
- Do not vendor any external benchmark code.
- Keep unit tests deterministic and fast.

Acceptance:
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q` passes
- `python scripts/validate_metrics.py --data <artifact> --preds <preds>` works without external repos
- When `third_party/PerturBench` is present locally, parity comparisons run and produce PASS/FAIL

Output:
- Diff summary
- The new contract doc path
- Example parity report output (from a synthetic run)


MEGA-PROMPT 12: PerturBench dataset + official split import (end-to-end)
=======================================================================

You are working in PerturbFM.

Goal:
Make it trivial to go from “I have PerturBench locally” → “I have PerturbDataset artifacts + frozen official splits ready to train/eval”.

Scope:
- src/perturbfm/data/adapters/perturbench.py
- src/perturbfm/data/canonical.py (only if schema needs extension)
- src/perturbfm/data/splits/split_store.py (if import helpers are needed)
- src/perturbfm/cli.py (new subcommands)
- scripts/ (optional helper scripts)
- tests/ (add smoke tests; external repo optional)

Requirements:
1) Implement/import-ready CLI:
   - `perturbfm data import-perturbench ...` (input can be a .h5ad or a dataset directory)
   - `perturbfm splits import-perturbench ...` (discover official split files and store them in SplitStore)
   - `perturbfm splits list` (optional but useful; list available split hashes + metadata)
2) Split import should:
   - preserve train/val/test (and calibration split if provided)
   - populate `ood_axes`/metadata where possible
   - freeze and save split JSON under split hash
3) Adapter should:
   - load large `.h5ad` safely (prefer backed/memory-friendly approach if possible)
   - map required fields into `PerturbDataset.obs` (`pert_id`, `context_id`, `batch_id`, `is_control`)
   - never guess labels silently; fail fast if required columns are missing
4) Tests:
   - smoke test import paths using a synthetic artifact (and/or a minimal tiny .h5ad if feasible)
   - ensure imported splits are frozen and hash-locked

Acceptance:
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q` passes
- Running the import commands produces an artifact with valid schema + at least one stored split

Output:
- Diff summary
- Example CLI commands for the full import flow
- Example `perturbfm splits show <HASH>` output


MEGA-PROMPT 13: PerturBench suite runner + scorecard (reproducible benchmark loop)
=================================================================================

You are working in PerturbFM.

Goal:
Create a reproducible suite runner that can:
- iterate over PerturBench tasks/datasets/splits
- train specified models (baseline/v0/v1/v2/…)
- evaluate with full panels
- write a single “scorecard” JSON and a human-readable summary

Scope:
- scripts/run_perturbench_suite.py (new)
- scripts/run_ablations.py (reuse/extend if appropriate)
- src/perturbfm/eval/report.py (optional improvements)
- docs/ (optional: “how to run the suite”)

Requirements:
1) Task config:
   - define a simple JSON/YAML format listing datasets, split hashes, and model configs
   - include multiple seeds support
2) Suite runner behavior:
   - run each experiment into a unique run_dir
   - aggregate results into `runs_summary.json` (or similar)
   - compute aggregate rankings (global metrics) per task + across tasks
3) Robustness:
   - if a run fails, record failure reason and continue (configurable)
   - always record provenance: data_hash, split_hash, config_hash, git commit, and command line
4) Provide a “dry run” mode that uses the synthetic dataset to validate end-to-end quickly.

Acceptance:
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q` passes
- A dry run on synthetic data produces a scorecard file and at least one run_dir with artifacts

Output:
- Diff summary
- Example suite config file
- Example scorecard snippet


MEGA-PROMPT 14: SLURM-first execution layer (sbatch templates + torchrun/DDP)
============================================================================

You are working in PerturbFM.

Goal:
Make the repo “HPC-native” so large-scale training, pretraining, and sweeps are easy and consistent.

Scope:
- scripts/slurm/ (new folder)
- scripts/ (new launchers as needed)
- docs/ (a short SLURM usage doc)

Requirements:
1) Add SLURM scripts/templates for:
   - single-run training (one dataset/split/model)
   - evaluation-only (compute metrics from predictions)
   - multi-seed sweep (job arrays)
2) DDP support:
   - prefer `torchrun` with env vars compatible with SLURM
   - handle single-node and multi-node cases
   - define clear conventions for output directories and log files
3) Make the scripts configurable:
   - partition, time, gpus, cpus, mem, nodes, array range, etc.
   - minimal required flags; sane defaults
4) Do not hardcode cluster-specific paths; use environment variables and arguments.

Acceptance:
- Scripts are syntactically valid and documented.
- Dry-run instructions exist and are clear.

Output:
- Added script list + brief usage examples
- The doc path for SLURM usage


MEGA-PROMPT 15: Scalable data pipeline (large AnnData, memmap, batching)
======================================================================

You are working in PerturbFM.

Goal:
Make training and evaluation scale to realistic PerturBench sizes without loading everything into RAM.

Scope:
- src/perturbfm/data/ (new dataset utilities)
- src/perturbfm/train/trainer.py (switch to minibatches)
- scripts/ (optional conversion utilities)
- tests/ (new fast unit tests)

Requirements:
1) Implement a scalable dataset backend:
   - support loading from existing artifact (npz) for small/synthetic
   - support a “large mode” (memmap shards or backed h5ad) for real data
2) Provide a batching API:
   - yields `(x_control, delta, pert_id/pert_mask, context_id, ...)` mini-batches
   - deterministic shuffling with seed
3) Implement a conversion utility (optional but recommended):
   - convert a `.h5ad` or big matrix into sharded memmaps aligned with `PerturbDataset.var`
4) Keep schema compatibility:
   - do not break existing artifact format unless you also provide a migration path

Acceptance:
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q` passes
- A synthetic end-to-end training run uses the batching path and matches prior outputs (within tolerance)

Output:
- Diff summary
- Notes on the “small vs large” data paths


MEGA-PROMPT 16: Strong baselines for PerturBench (must-beat references)
======================================================================

You are working in PerturbFM.

Goal:
Add strong baselines so we can quantify when complexity actually helps and avoid chasing fake wins.

Scope:
- src/perturbfm/models/baselines/
- src/perturbfm/train/trainer.py (fit/predict integration)
- src/perturbfm/cli.py (expose baselines)
- tests/ (add lightweight tests)

Candidate baselines (choose a minimal, high-impact set):
- scGen-style latent shift baseline (simple encoder + shift by perturbation)
- per-context ridge / multi-task linear model
- “control-only” predictor and “global perturbation mean” sanity baselines

Requirements:
1) Implement baselines cleanly and modularly.
2) Ensure baselines work with official PerturBench splits and the suite runner.
3) Provide baseline comparisons on synthetic data:
   - show that smarter baselines beat trivial baselines when signal exists
4) Avoid adding new heavy dependencies unless absolutely necessary; ask before adding.

Acceptance:
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q` passes
- Suite runner can train/eval at least 2 baselines end-to-end

Output:
- Diff summary
- Example scorecard showing baseline ordering on synthetic


MEGA-PROMPT 17: Model upgrade for PerturBench (state-dependent CGIO / “v3”)
==========================================================================

You are working in PerturbFM.

Goal:
Remove the biggest current modeling ceiling: v2 CGIO should be able to express state-dependent effects and heterogeneity by conditioning on the control cell state (not only context_id).

Scope:
- src/perturbfm/models/perturbfm/cgio.py (or add a new module for v3)
- src/perturbfm/train/trainer.py
- src/perturbfm/eval/evaluator.py
- src/perturbfm/cli.py
- tests/

Requirements:
1) Add a new model variant (recommended name: perturbfm_v3) that takes:
   - control expression `x_control` (basal state)
   - pert_mask / pert_genes set
   - context_idx (still useful)
   and outputs mean/var in delta space.
2) Architecture guidance (keep it simple but expressive):
   - CellEncoder(x_control) → basal embedding
   - GraphPropagator(pert_mask, ctx_emb) → perturb embedding (or a set encoder)
   - Fuse(basal, perturb, ctx) → contextual operator / head
3) Keep backward compatibility:
   - do not break v0/v1/v2 interfaces; add new commands rather than changing old ones
4) Add tests:
   - smoke training on synthetic data
   - shape checks + determinism under fixed seed

Acceptance:
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q` passes
- `perturbfm train perturbfm-v3` (or equivalent) runs on synthetic and produces predictions + metrics

Output:
- Diff summary
- Short explanation of why v3 removes the “state-independence” ceiling


MEGA-PROMPT 18: Large-scale pretraining (cell foundation encoder) + integration
==============================================================================

You are working in PerturbFM.

Goal:
Leverage large compute: pretrain a strong cell encoder on massive observational scRNA-seq, then fine-tune on PerturBench tasks.

Scope:
- src/perturbfm/models/ (new pretraining model/heads as needed)
- scripts/pretrain_cell_encoder.py (new)
- scripts/slurm/ (new sbatch for pretraining)
- docs/ (short “how to pretrain” guide)

Requirements:
1) Pretraining objective (choose one or support multiple):
   - masked gene modeling (reconstruct masked genes)
   - denoising autoencoder (noise injection → reconstruction)
   - contrastive cell representation learning (optional)
2) Implementation requirements:
   - DDP + mixed precision support
   - checkpointing and resuming
   - logs that include loss curves and key config/provenance
3) Integration:
   - allow PerturbFM models to load a pretrained CellEncoder checkpoint
   - support “freeze encoder” vs “fine-tune encoder” modes
4) Provide a tiny local smoke mode:
   - run on synthetic or small artifact to validate training loop end-to-end without big data

Acceptance:
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q` passes
- Pretraining script runs in smoke mode and writes a checkpoint
- Fine-tuning path can load that checkpoint successfully

Output:
- Diff summary
- Example SLURM command lines for pretrain + fine-tune


MEGA-PROMPT 19: Full PerturBench sweep (multi-seed + HPO + leaderboard artifact)
===============================================================================

You are working in PerturbFM.

Goal:
Turn “we ran a lot of experiments” into a reproducible, queryable leaderboard.

Scope:
- scripts/ (sweep launcher + aggregator)
- scripts/slurm/ (array jobs)
- docs/ (leaderboard/reporting doc)

Requirements:
1) Implement a sweep launcher that can:
   - expand a search space (grid or random)
   - submit SLURM array jobs
   - record every config with a stable hash
2) Implement a results aggregator that:
   - scans `runs/`
   - builds a `leaderboard.json` with best runs per task + overall
   - outputs a short markdown summary table
3) Add guardrails:
   - never select based on test metrics; select on validation metrics only
   - record the selection rule in the leaderboard artifact

Acceptance:
- Dry-run sweep on synthetic works and produces a leaderboard artifact
- No-test-leakage selection is enforced in code

Output:
- Diff summary
- Example leaderboard snippet


MEGA-PROMPT 20: scPerturBench integration (context-OOD next, external-only)
==========================================================================

You are working in PerturbFM.

Goal:
After PerturBench is solid, integrate scPerturBench context-OOD evaluation without GPL contamination:
- load scPerturBench datasets/splits from `third_party/scPerturBench`
- map them into `PerturbDataset`
- run our evaluator and/or their external harness as appropriate

Scope:
- src/perturbfm/data/adapters/scperturbench.py (replace stub with a loader wrapper that calls out to external data)
- scripts/ (import/export helpers)
- docs/ (clear licensing + usage instructions)

Requirements:
1) Implement a loader that:
   - requires user-provided path to scPerturBench checkout/data
   - does not copy GPL code into this repo
   - outputs a `PerturbDataset` artifact (or a compatible view) for our pipeline
2) Provide an evaluation path:
   - run context-OOD split consistent with scPerturBench definitions
   - compute scPerturBench metrics (our implementation) + optionally call reference scripts externally for parity
3) Add tests that:
   - do not require third_party presence (skip when absent)
   - validate the adapter’s error messages and “no vendoring” constraints

Acceptance:
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q` passes
- With a local scPerturBench checkout, the adapter can produce an artifact and run evaluation

Output:
- Diff summary
- Example commands for “with scPerturBench present” and “without scPerturBench present”


MEGA-PROMPT 21: scPerturBench context-OOD suite + scorecard (mirror PerturBench loop)
===================================================================================

You are working in PerturbFM.

Goal:
Build the same kind of suite runner/scorecard for scPerturBench context-OOD as we did for PerturBench, so improvements are measurable and regressions are obvious.

Scope:
- scripts/run_scperturbench_suite.py (new)
- scripts/slurm/ (new sbatch templates)
- docs/ (short “how to run context-OOD suite” guide)

Requirements:
1) Define the suite config format (reuse the PerturBench suite format if possible).
2) Run across:
   - multiple datasets
   - multiple held-out contexts
   - multiple seeds
3) Aggregate results:
   - per-dataset scorecards
   - overall aggregates
4) Provide parity hooks:
   - optional callout to scPerturBench reference scripts if available

Acceptance:
- Dry run works (synthetic or minimal fixture)
- With external scPerturBench present, a real run produces a scorecard artifact

Output:
- Diff summary
- Example suite config + example scorecard snippet


MEGA-PROMPT 22: Context-OOD modeling + calibration upgrades (after harness exists)
===============================================================================

You are working in PerturbFM.

Goal:
Once scPerturBench context-OOD is wired, implement upgrades that specifically target transfer across contexts while remaining honest (calibrated uncertainty, failure detection).

Scope:
- src/perturbfm/models/perturbfm/ (context modules)
- src/perturbfm/eval/uncertainty_metrics.py (if new calibration metrics are added)
- scripts/ (optional ablation helpers)
- tests/ (lightweight tests; no external dependencies)

Candidate upgrades (pick a minimal, testable set):
- context adapters (FiLM/LoRA-style modulation of model layers by context)
- mixture-of-experts by context with calibrated gating
- domain-shift diagnostics (OOD detectors) tied to uncertainty calibration

Requirements:
1) Implement at least one context-OOD-targeted mechanism behind flags.
2) Add ablations so we can measure benefit vs baselines and ensure no regressions.
3) Keep evaluation honest:
   - selection on validation only
   - maintain no leakage
4) Add small synthetic tests that at least verify:
   - the feature toggles work
   - uncertainty responds sensibly to distribution shift in toy settings

Acceptance:
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q` passes
- Suite runners can compare baseline vs upgraded models with a single config change

Output:
- Diff summary
- Suggested next prompt number based on findings
