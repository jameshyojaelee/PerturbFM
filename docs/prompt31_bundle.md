# Prompt 31 — Reproducible Results Bundle

This file summarizes the minimal artifacts, hashes, and commands needed to reproduce
Prompt 24–30 outputs and assemble a shareable results bundle.

## Dataset manifest

| Dataset | Artifact path | Data hash | Split hash | Split JSON |
| --- | --- | --- | --- | --- |
| Norman19 | `data/artifacts/perturbench/norman19` | `bb87cf02` | `da58b88d99c4133b173cf75d4998649a36974d7ec4ebd831930a10328cfe0b1b` | `splits/da58b88d99c4133b173cf75d4998649a36974d7ec4ebd831930a10328cfe0b1b.json` |
| XieHon2017 | `data/artifacts/scperturb/xiehon2017` | `cadf5f4b` | `b8613d05c135cab50c2986a67689235cd5bc6857b3e3128273fd2dc0c6f8ae46` | `splits/b8613d05c135cab50c2986a67689235cd5bc6857b3e3128273fd2dc0c6f8ae46.json` |

Data hashes are recorded in run `config.json` files (e.g., `runs/20251224-001350_da58b88_perturbfm_v0_f5161df3/config.json`).

## Graphs used

- Identity graphs:
  - `graphs/norman19_identity.npz`
  - `graphs/xiehon2017_identity.npz`
- Coexpression prior (train-only):
  - `graphs/norman19_coexpr_train_top20.npz`
  - `graphs/norman19_coexpr_train_top20.notes.txt`

## Core scorecards and summaries

- PerturBench (Norman19): `runs/scorecards/perturbench_norman19_scorecard.json`
- scPerturb (XieHon2017): `runs/scorecards/scperturb_xiehon2017_scorecard.json`
- Fine-tuning summary: `runs/finetune_norman19_summary.md`
- Multi-seed summary: `runs/finetune_multiseed_summary.md`
- Sweep leaderboard: `sweeps/norman19_v0/leaderboard.json` (+ `leaderboard.md`)

## External competitor evaluation

- Manifest: `configs/external_predictions_manifest.example.json`
- Predictions (examples):
  - `runs/external_predictions/state/state_norman19/da58b88d99c4133b173cf75d4998649a36974d7ec4ebd831930a10328cfe0b1b/predictions.npz`
  - `runs/external_predictions/gears/norman19/da58b88d99c4133b173cf75d4998649a36974d7ec4ebd831930a10328cfe0b1b/predictions.npz`
  - `runs/external_predictions/state/state_xiehon2017/b8613d05c135cab50c2986a67689235cd5bc6857b3e3128273fd2dc0c6f8ae46/predictions.npz`
  - `runs/external_predictions/scgen/xiehon2017/b8613d05c135cab50c2986a67689235cd5bc6857b3e3128273fd2dc0c6f8ae46/predictions.npz`
  - `runs/external_predictions/cpa/xiehon2017/b8613d05c135cab50c2986a67689235cd5bc6857b3e3128273fd2dc0c6f8ae46/predictions.npz`
- Eval outputs: `runs/external_eval/*/...` and `runs/external_eval/manifest_results.json`

## Repro commands (minimal)

### Internal suites

```bash
# PerturBench suite (Norman19)
PYTHONPATH=src python scripts/run_perturbench_suite.py \
  --config configs/suites/perturbench_norman19.json \
  --out runs/scorecards/perturbench_norman19_scorecard.json

# scPerturb suite (XieHon2017)
PYTHONPATH=src python scripts/run_scperturbench_suite.py \
  --config configs/suites/scperturb_xiehon2017.json \
  --out runs/scorecards/scperturb_xiehon2017_scorecard.json
```

### External predictions evaluation

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
  micromamba run -p third_party/mamba/env_scgen \
  python scripts/eval_external_predictions.py \
  --manifest configs/external_predictions_manifest.example.json
```

### Leaderboard aggregation

```bash
PYTHONPATH=src python scripts/aggregate_leaderboard.py \
  --runs sweeps/norman19_v0 \
  --out sweeps/norman19_v0/leaderboard.json
```

## Notes

- External model predictions are evaluated in delta space using the same splits
  and metric panels; uncertainty for external models is filled with a small
  constant variance when missing.
- If you refresh external repos under `third_party/`, reapply the compatibility
  patches listed in `docs/third_party_compat.md`.
