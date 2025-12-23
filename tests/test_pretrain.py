import os
import subprocess
import sys
from pathlib import Path

import pytest

from perturbfm.data.registry import make_synthetic_dataset
from perturbfm.data.splits.split_spec import context_ood_split
from perturbfm.train.trainer import fit_predict_perturbfm_v0


def test_pretrain_cell_encoder_smoke(tmp_path):
    torch = pytest.importorskip("torch")
    ds = make_synthetic_dataset(n_obs=30, n_genes=6, n_contexts=3, seed=0)
    data_dir = tmp_path / "data"
    ds.save_artifact(data_dir)
    ckpt_path = tmp_path / "encoder.pt"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/pretrain_cell_encoder.py",
            "--data",
            str(data_dir),
            "--out",
            str(ckpt_path),
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--hidden-dim",
            "16",
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0
    assert ckpt_path.exists()

    split = context_ood_split(ds.obs["context_id"], ["C0"], obs_perts=ds.obs["pert_id"], seed=0, val_fraction=0.2).freeze()
    out = fit_predict_perturbfm_v0(
        ds,
        split,
        hidden_dim=16,
        lr=1e-2,
        epochs=1,
        device="cpu",
        pretrained_encoder=str(ckpt_path),
        freeze_encoder=True,
    )
    assert out["mean"].shape == out["var"].shape
