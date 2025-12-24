# Dev TODO (PerturbFM)

This file is the internal fix list from the initial buildout phase (kept for historical record).
For the **current** execution roadmap and next experiments, see `docs/prompts.md`.
For repository layout and current architecture, see `README.md`.
For local agent behavior rules, see `docs/SYSTEM_PROMPT.md` / `docs/AGENT_GUIDELINES.md` (gitignored by design).

## Pre‑change test run

Command:

```
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

Result (2025‑12‑23):

```
22 passed in 3.20s
```

## Repo module summary (pointer)

This section intentionally stays lightweight to avoid drifting from the code.
Use the current sources of truth:
- `README.md` → “Repository layout”
- `docs/prompts.md` → “Active roadmap prompts”

## Issue records (A–I) — status (archived)

These were the original MEGA‑PROMPT 1 findings. They are retained as historical record.

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

F) **scPerturBench metrics include placeholder + O(n²)** — **fixed**
- File: `src/perturbfm/eval/metrics_scperturbench.py`
- Lines: 22–50 (energy sampled), 66–78 (Common‑DEGs effect size)
- Impact: scalability + parity risk
 - Current: scalable defaults (pair-sampled energy, sliced Wasserstein, effect-size DEGs) + parity harness in `scripts/validate_metrics.py`; still needs external reference to confirm official parity.

G) **PerturBench rank metrics not parity with official** — **fixed**
- File: `src/perturbfm/eval/metrics_perturbench.py`
- Lines: 14–78 (mean-delta rank + top-k overlap + diversity ratio)
- Impact: mismatch vs benchmark
 - Current: rank and collapse diagnostics implemented with scalable defaults; still needs parity validation vs official reference when available.

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
   - Current: pair-sampled energy + sliced Wasserstein + effect-size Common‑DEGs; parity harness compares against reference JSON or external scripts.  
   - Acceptance: parity checks within tolerance when reference available.
6) **PerturBench rank/collapse metrics parity**  
   - File: `metrics_perturbench.py`  
   - Current: mean-delta rank + top-k overlap + variance/diversity diagnostics; parity harness pending external reference.  
   - Acceptance: synthetic tests + reference validation.
7) **Sparse graph refactor + scalable gating**  
   - Files: `gene_graph.py`, `perturb_encoder.py`, `cgio.py`  
   - Current: edge_index + lowrank/node/MLP gating; no dense gate matrices in tests.  
   - Acceptance: no dense G×G gates; tests verify sparse path.

### P2 (cleanup + packaging)
8) **Add `bench` extras**  
   - File: `pyproject.toml`  
   - Acceptance: `pip install -e ".[bench]"` works with anndata/h5py.

## Do‑not‑do constraints (moved)

See `docs/SYSTEM_PROMPT.md` (local, gitignored) for the current hard‑rule list.

## Calibration split derivation (note)

Conformal calibration uses a **calibration subset** that is *disjoint* from test:
1) If `Split.calib_idx` is present, use it directly.
2) Otherwise, deterministically derive `calib_idx` from `val_idx` using a seeded RNG:
   - seed = `split.seed` XOR the split hash prefix
   - shuffle `val_idx` and take the first 50%
3) Never use `test_idx` for calibration; a guard raises if indices overlap.
