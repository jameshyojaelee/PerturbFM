#!/usr/bin/env python3
"""
Pretrain a CellEncoder using a simple denoising autoencoder objective.

Usage:
  python scripts/pretrain_cell_encoder.py --data /path/to/artifact --out /tmp/cell_encoder.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from perturbfm.data.batching import iter_index_batches
from perturbfm.data.canonical import PerturbDataset
from perturbfm.models.pretrain import CellAutoencoder


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Dataset artifact path.")
    ap.add_argument("--out", required=True, help="Checkpoint path.")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--noise-std", type=float, default=0.1)
    args = ap.parse_args()

    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torch is required for pretraining") from exc

    ds = PerturbDataset.load_artifact(args.data)
    X = ds.X_control if ds.X_control is not None else ds.X_pert
    if X is None:
        raise ValueError("Dataset must include X_control or X_pert for pretraining.")

    model = CellAutoencoder(n_genes=ds.n_genes, hidden_dim=args.hidden_dim).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    for epoch in range(args.epochs):
        for batch_idx in iter_index_batches(np.arange(ds.n_obs), batch_size=args.batch_size, seed=epoch, shuffle=True):
            x = torch.as_tensor(X[batch_idx], dtype=torch.float32, device=args.device)
            noise = torch.randn_like(x) * args.noise_std
            x_noisy = x + noise
            recon, _ = model(x_noisy)
            loss = torch.nn.functional.mse_loss(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": model.encoder.state_dict(), "hidden_dim": args.hidden_dim}, out_path)
    print(f"Wrote encoder checkpoint to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
