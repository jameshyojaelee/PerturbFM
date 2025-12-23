"""Shared download utilities."""

from __future__ import annotations

import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable


def _download_one(url: str, out_path: Path, retries: int = 3, timeout: int = 60) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                with open(out_path, "wb") as f:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
            return
        except Exception as exc:  # pragma: no cover - network errors
            last_err = exc
            time.sleep(min(5 * attempt, 30))
    raise RuntimeError(f"Failed to download {url}: {last_err}")


def download_many(items: Iterable[tuple[str, Path]], max_workers: int, retries: int = 3) -> None:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_download_one, url, path, retries) for url, path in items]
        for fut in as_completed(futures):
            fut.result()


def cpu_workers(default: int = 8) -> int:
    try:
        return max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "")) or os.cpu_count() or default)
    except Exception:
        return default
