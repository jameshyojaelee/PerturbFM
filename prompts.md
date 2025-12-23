# Codex Mega-Prompts — PerturbFM roadmap

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
