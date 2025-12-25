# v3a Ablation Report (Template)

This report summarizes Transformer v3a results with mandatory sections.

## Metadata
- dataset:
- split hash:
- git commit:
- eval split (val/test):
- configuration(s):

## Primary metrics
- scPerturBench (MSE, PCC, Energy, Wasserstein, KL, Common_DEGs)
- PerturBench (RMSE, RankMetrics, VarianceDiagnostics)

## Uncertainty + calibration
- coverage@0.90 / coverage@0.95
- NLL
- risk–coverage curve notes
- OOD AUROC (if labels available)

## Variance attribution
- aleatoric vs epistemic (if ensemble used)

## Data efficiency
- 10% / 30% / 100% comparisons (fixed steps)

## Decision
- Go / no‑go vs v3 baseline
