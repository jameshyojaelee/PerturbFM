# PerturbFM — Session Context (auto-generated)

This file is a quick-start context snapshot so a new Codex session can resume without re-explaining.

## Mission + validity rails (non-negotiable)
- Frozen OOD splits (SplitStore, hash-locked). Never regenerate silently.
- Complete metric panels (scPerturBench + PerturBench + uncertainty) or fail.
- No leakage: never use `test_idx` for calibration, preprocessing, or selection.
- Graph priors are external; no GPL code in `src/`.

## Current plan (high level)
1) Lock validity rails + regression tests.
2) Prove Transformer value without diffusion (new `perturbfm-v3a`).
3) Diffusion head only if Transformer passes go/no-go.
4) Tahoe-100M pretraining + immutable eval subsets.
5) Cross-regime “god-like” validation suite with negative controls.

See `docs/prompts.md` for the full roadmap and mega-prompts.

## Execution tracker snapshot
DONE: prompts 23–35 (double-pert baselines, residuals, reports).
PARTIAL: prompts 36–39 (sweeps launched; some tasks running).
NEW: prompts 40–44 (anti-self-deception, v3a, diffusion, Tahoe, validation suite).

### Ongoing sweeps / SLURM
- Grid sweep (Prompt 36): job `12833547`, tasks 0–9 running; pending tasks canceled.  
  Resume list: `runs/scorecards/sweeps/norman19_doubles_grid/config_list_resume.txt`
- Graph sweep (Prompt 37): pending tasks canceled earlier.
- Combo-weight sweep (Prompt 38): pending tasks canceled earlier.
- Split sweep (Prompt 39): pending tasks canceled earlier.

To resume later:
```
sbatch --partition=gpu --gres=gpu:l40s:1 --cpus-per-task=8 --mem=120G --time=24:00:00 \
  --array=0-25 \
  --export=ALL,CONFIG_LIST=runs/scorecards/sweeps/norman19_doubles_grid/config_list_resume.txt,OUT_ROOT=runs/scorecards/sweeps/norman19_doubles_grid \
  scripts/slurm/sweep_array.sbatch
```

## Key artifacts and reports
- Latest doubles report with v3/v3_residual:  
  `runs/reports/norman19_doubles_report_v3.md` (+ `.json`)
- Baseline doubles report:  
  `runs/reports/norman19_doubles_report.md`

## New tooling added recently
- Transformer v3a path (model + trainer + evaluator + CLI).
- Eval split support (`--eval-split val|test`) in suite runner.
- Combo-weight residual loss (v2/v3 residual) + tests.
- Sweep generators:
  - `scripts/generate_norman19_doubles_sweep.py`
  - `scripts/generate_norman19_graph_sweep.py`
  - `scripts/generate_norman19_combo_weight_sweep.py`
  - `scripts/generate_norman19_split_sweep.py`
- Sweep aggregation: `scripts/aggregate_sweep_results.py`

## Tahoe-100M (local data)
Local path (already downloaded):
`/gpfs/commons/home/jameslee/CellJEPA/data/raw/tahoe-100m/`
Layout:
- `data/train-*.parquet` shards
- `metadata/*.parquet` tables

Plan (Prompt 43):
- Use local parquet shards for streaming pretraining.
- Materialize immutable eval subsets as `PerturbDataset` artifacts.
- Define OOD splits: cell line / drug / MoA holdouts.

## Decisions (current defaults)
- Tahoe ingestion: local parquet shards (library choice to be finalized; likely `pyarrow` as optional dep).
- Transformer gene set: prioritize best results; start with HVGs for iteration, confirm full-gene runs for final claims.
- Foundation claim scope: **both** Tahoe OOD + transfer to PerturBench/scPerturBench.
- Pretraining objective: start with denoising AE; consider adding masked-gene modeling if it improves.

## Reminders
- For SLURM, ensure `PYTHONPATH=./src` or install editable (`pip install -e .`).
- Avoid test-peeking in sweeps; Prompt 40 adds dev/test gates.
