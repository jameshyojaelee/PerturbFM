import os
import subprocess
import sys
from pathlib import Path


def test_perturbench_suite_dry_run(tmp_path):
    out_path = tmp_path / "scorecard.json"
    run_root = tmp_path / "runs"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_perturbench_suite.py",
            "--dry-run",
            "--out",
            str(out_path),
            "--run-root",
            str(run_root),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0
    assert out_path.exists()
    assert (tmp_path / "runs_summary.json").exists()
    assert (tmp_path / "scorecard.txt").exists()
    # Ensure at least one run directory with metrics exists.
    run_dirs = [p for p in run_root.glob("*") if p.is_dir()]
    assert run_dirs
    assert any((p / "metrics.json").exists() for p in run_dirs)
