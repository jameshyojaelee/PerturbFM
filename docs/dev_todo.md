# Dev TODO (PerturbFM)

This file is the internal fix list derived from `prompts.md` MEGA‑PROMPT 1.

## Pre‑change test run

Command:

```
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

Result (2025‑12‑23):

```
22 passed in 3.20s
```

## Repo module summary (ground truth)

- **splits**: `src/perturbfm/data/splits/`
  - `split_spec.py`: defines `Split` + generators (`context_ood_split`, `perturbation_ood_split`, `combo_generalization_split`, `covariate_transfer_split`)
  - `split_store.py`: stores frozen splits by hash under `splits/`
- **evaluator**: `src/perturbfm/eval/evaluator.py`
  - run entrypoints: `run_baseline`, `run_perturbfm_v0/v1/v2`, `evaluate_predictions`
  - enforces metric completeness for baseline/v0/v2; v1 currently missing the guard
  - conformal intervals currently computed on **test** subset
- **scPerturBench metrics**: `src/perturbfm/eval/metrics_scperturbench.py`
  - MSE, PCC, Energy, Wasserstein, KL, Common‑DEGs
  - Energy uses O(n²) pairwise distances; Common‑DEGs is placeholder overlap@100
- **PerturBench metrics**: `src/perturbfm/eval/metrics_perturbench.py`
  - RMSE, RankMetrics (Spearman proxy), VarianceDiagnostics (pred_var – MSE)
  - notes still mark RankMetrics/VarianceDiagnostics as TODO for parity
- **CGIO**: `src/perturbfm/models/perturbfm/cgio.py`
  - `GraphPropagator` uses dense adjacency + dense gating matrices
  - forward references an undefined `residual`
- **Graph encoder / graph utils**:
  - `src/perturbfm/models/perturbfm/perturb_encoder.py`: dense adjacency + dense gating
  - `src/perturbfm/models/perturbfm/gene_graph.py`: dense adjacency (`[G, G]`)

## Issue records (A–I) — status

Each record includes location + why it matters.

A) **CGIO GraphPropagator uses undefined `residual`** — **fixed**
- File: `src/perturbfm/models/perturbfm/cgio.py`
- Lines: 20–30 (see line 29)
- Impact: runtime NameError on gating‑zero branch

B) **Conformal intervals use test residuals (leakage)** — **fixed**
- File: `src/perturbfm/eval/evaluator.py`
- Lines: 109–117 (and similar in v0/v1/v2 blocks)
- Impact: calibration uses test labels → invalid evaluation

C) **Context‑OOD split does not enforce shared perturbations** — **fixed**
- File: `src/perturbfm/data/splits/split_spec.py`
- Lines: 72–96
- Impact: headline claim (“context OOD”) can be mixed with perturbation OOD

D) **run_perturbfm_v1 does not enforce metric completeness** — **fixed**
- File: `src/perturbfm/eval/evaluator.py`
- Lines: 208–220 (writes metrics without `_require_metrics_complete`)
- Impact: partial metrics can silently pass for v1

E) **Graph adjacency + gating are dense (won’t scale)** — **fixed**
- Files:
  - `src/perturbfm/models/perturbfm/gene_graph.py` (lines 15–33)
  - `src/perturbfm/models/perturbfm/perturb_encoder.py` (lines 31–48)
  - `src/perturbfm/models/perturbfm/cgio.py` (lines 9–36)
- Impact: O(G²) memory/time

F) **scPerturBench metrics include placeholder + O(n²)** — **partially fixed**
- File: `src/perturbfm/eval/metrics_scperturbench.py`
- Lines: 22–31 (energy O(n²)), 59–65 (Common‑DEGs placeholder)
- Impact: scalability + parity risk
 - Current: energy uses subsampling, Common‑DEGs is deterministic; still needs parity validation vs reference.

G) **PerturBench rank metrics not parity with official** — **partially fixed**
- File: `src/perturbfm/eval/metrics_perturbench.py`
- Lines: 14–33 (Spearman proxy), 86–89 notes mark TODO
- Impact: mismatch vs benchmark
 - Current: rank metric + variance ratio implemented; still needs parity validation vs reference.

H) **README references missing file**
- Checked: no missing files found as of 2025‑12‑23

I) **`pyproject.toml` lacks bench extras** — **fixed**
- File: `pyproject.toml`
- Impact: adapter mentions extras but none exist

## Fix plan (prioritized)

### P0 (must fix before credible results)
1) **Fix CGIO `residual` bug**  
   - File: `src/perturbfm/models/perturbfm/cgio.py`  
   - Acceptance: new unit test forces gating‑zero branch; pytest passes.
2) **Remove conformal leakage**  
   - Files: `split_spec.py`, `evaluator.py`, `conformal.py`  
   - Add `calib_idx`; derive from val when absent; use only calib for conformal residuals.  
   - Acceptance: tests prove no test leakage; `calibration.json` stores calib metadata.
3) **Enforce metric completeness for v1**  
   - File: `src/perturbfm/eval/evaluator.py`  
   - Acceptance: v1 raises on missing metrics; test covers this.
4) **Context‑OOD split validity**  
   - File: `src/perturbfm/data/splits/split_spec.py`  
   - Enforce shared perturbations (filter or fail) with explicit metadata.  
   - Acceptance: tests for mixed OOD; split metadata records axes.

### P1 (needed for parity + scalability)
5) **scPerturBench metric parity + scalable implementations**  
   - File: `metrics_scperturbench.py`, `scripts/validate_metrics.py`  
   - Current: scalable defaults implemented (subsampled energy, quantile Wasserstein) but must be validated vs reference.  
   - Acceptance: parity checks within tolerance.
6) **PerturBench rank/collapse metrics parity**  
   - File: `metrics_perturbench.py`  
   - Current: Spearman proxy + variance ratio; still requires parity check.  
   - Acceptance: synthetic tests + reference validation.
7) **Sparse graph refactor + scalable gating**  
   - Files: `gene_graph.py`, `perturb_encoder.py`, `cgio.py`  
   - Current: edge_index + lowrank/node/MLP gating; no dense gate matrices in tests.  
   - Acceptance: no dense G×G gates; tests verify sparse path.

### P2 (cleanup + packaging)
8) **Add `bench` extras**  
   - File: `pyproject.toml`  
   - Acceptance: `pip install -e ".[bench]"` works with anndata/h5py.

## Do‑not‑do constraints

- **No test leakage**: conformal calibration must never use `test_idx`.
- **No GPL vendoring**: external benchmark code stays under `third_party/`.
- **No silent split regeneration**: splits must remain frozen + hash‑locked.

## Calibration split derivation (note)

Conformal calibration uses a **calibration subset** that is *disjoint* from test:
1) If `Split.calib_idx` is present, use it directly.
2) Otherwise, deterministically derive `calib_idx` from `val_idx` using a seeded RNG:
   - seed = `split.seed` XOR the split hash prefix
   - shuffle `val_idx` and take the first 50%
3) Never use `test_idx` for calibration; a guard raises if indices overlap.
