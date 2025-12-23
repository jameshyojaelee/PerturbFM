import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from perturbfm.data.registry import make_synthetic_dataset


def test_validate_metrics_missing_external(tmp_path):
    data_dir = tmp_path / "data"
    ds = make_synthetic_dataset(n_obs=25, n_genes=12, seed=0)
    ds.save_artifact(data_dir)

    preds_path = tmp_path / "preds.npz"
    idx = np.arange(ds.n_obs)
    np.savez_compressed(preds_path, mean=ds.delta, var=np.zeros_like(ds.delta), idx=idx)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run(
        [sys.executable, "scripts/validate_metrics.py", "--data", str(data_dir), "--preds", str(preds_path)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0
    assert "scPerturBench reference repo not found" in result.stdout
    assert "PerturBench reference repo not found" in result.stdout
