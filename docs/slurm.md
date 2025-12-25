# SLURM usage (templates)

This repo includes **SLURM-first templates** under `scripts/slurm/`.
They are intentionally generic; override resources at submit time (command-line flags take precedence over `#SBATCH` defaults).

## Environment note (important)

These templates assume the `perturbfm` package is importable on the compute node.
Two safe options:
1) install the repo in your environment (recommended): `python -m pip install -e .`
2) run directly from a checkout by exporting `PYTHONPATH=./src` (the array template does this automatically)

## 1) Single-run training

```bash
sbatch \
  --partition=<partition> \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --time=02:00:00 \
  --export=ALL,PYTHONPATH=./src,DATA_PATH=/path/to/artifact,SPLIT_HASH=<HASH>,MODEL_KIND=baseline,BASELINE_NAME=global_mean \
  scripts/slurm/train.sbatch
```

For v1/v2, set `MODEL_KIND=perturbfm-v1` or `MODEL_KIND=perturbfm-v2` and pass:

```bash
--export=ALL,DATA_PATH=...,SPLIT_HASH=...,MODEL_KIND=perturbfm-v1,ADJACENCY_PATH=/path/to/graph.npz,PERT_MASKS_PATH=/path/to/pert_masks.json
```

## 2) Evaluation-only

```bash
sbatch \
  --partition=<partition> \
  --cpus-per-task=4 \
  --time=00:30:00 \
  --export=ALL,PYTHONPATH=./src,DATA_PATH=/path/to/artifact,SPLIT_HASH=<HASH>,PREDS_PATH=/path/to/predictions.npz \
  scripts/slurm/eval.sbatch
```

## 3) Multi-seed sweep (array)

Create a file with one suite config path per line:

```bash
printf "%s\n" configs/suite_seed0.json configs/suite_seed1.json > /tmp/suite_list.txt
```

Then submit:

```bash
sbatch \
  --partition=<partition> \
  --array=0-1 \
  --cpus-per-task=4 \
  --time=01:00:00 \
  --export=ALL,CONFIG_LIST=/tmp/suite_list.txt,OUT_ROOT=/path/to/sweep_outputs \
  scripts/slurm/sweep_array.sbatch
```

## 4) DDP / torchrun

If your training code supports DDP, set `USE_TORCHRUN=1` and `GPUS_PER_NODE`:

```bash
--export=ALL,USE_TORCHRUN=1,GPUS_PER_NODE=4,DATA_PATH=...,SPLIT_HASH=...,MODEL_KIND=perturbfm-v0
```

The template uses `torchrun` with SLURM-provided node info (`SLURM_NNODES`, `SLURM_NODEID`).

## 5) Pretraining the CellEncoder

```bash
sbatch \
  --partition=<partition> \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --time=04:00:00 \
  --export=ALL,DATA_PATH=/path/to/artifact,OUT_PATH=/path/to/checkpoints/cell_encoder.pt \
  scripts/slurm/pretrain.sbatch
```
