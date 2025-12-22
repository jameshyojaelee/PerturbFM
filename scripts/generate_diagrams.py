#!/usr/bin/env python3
"""
Generate static diagrams for GitHub README rendering.

Outputs:
  docs/diagrams/pipeline.svg
  docs/diagrams/pipeline.png
  docs/diagrams/cgio.svg
  docs/diagrams/cgio.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def _box(ax, xy, w, h, text, fontsize=11, fc="#f8fafc"):
    x, y = xy
    shadow = FancyBboxPatch(
        (x + 0.006, y - 0.006),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=0,
        edgecolor="none",
        facecolor="#0f172a",
        alpha=0.08,
    )
    ax.add_patch(shadow)
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        edgecolor="#111827",
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color="#0f172a")


def _arrow(ax, start, end):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", lw=1.8, color="#0f172a"),
    )


def pipeline(out_path: Path, fmt: str) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 3.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    w, h = 0.16, 0.36
    y = 0.32
    xs = [0.03, 0.22, 0.41, 0.60, 0.79, 0.96]

    labels = [
        "Dataset artifact\n(data.npz + meta.json)",
        "SplitStore\n(hash-locked)",
        "Train\n(baseline / v0 / v1 / v2)",
        "predictions.npz\n(mean, var, idx)",
        "metrics.json\n+ calibration.json",
        "report.html",
    ]

    colors = ["#eef2ff", "#ecfeff", "#f0fdf4", "#fff7ed", "#f5f3ff", "#f8fafc"]
    for x, label, fc in zip(xs, labels, colors):
        _box(ax, (x - w / 2, y), w, h, label, fontsize=10, fc=fc)

    for i in range(len(xs) - 1):
        _arrow(ax, (xs[i] + w / 2, y + h / 2), (xs[i + 1] - w / 2, y + h / 2))

    ax.text(0.03, 0.9, "PerturbFM pipeline", fontsize=14, color="#0f172a", weight="bold")
    ax.text(0.03, 0.84, "artifact → split → train → eval → report", fontsize=11, color="#334155")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=260 if fmt == "png" else None, bbox_inches="tight")
    plt.close(fig)


def cgio(out_path: Path, fmt: str) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 6.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    w, h = 0.42, 0.1
    x = 0.3
    y0 = 0.82

    _box(ax, (x, y0), w, h, "pert_genes\n(list per sample)", fc="#eef2ff")
    _box(ax, (x, y0 - 0.14), w, h, "pert_mask\n(B × G)", fc="#ecfeff")
    _box(ax, (x, y0 - 0.28), w, h, "Graph propagation\n(single/multi‑graph, gated)", fc="#f0fdf4")
    _box(ax, (x, y0 - 0.42), w, h, "h embedding\n(B × d)", fc="#fff7ed")
    _box(ax, (x, y0 - 0.56), w, h, "Contextual low‑rank operator", fc="#f5f3ff")

    _box(ax, (0.08, y0 - 0.72), 0.35, h, "delta mean\n(B × G)", fc="#f8fafc")
    _box(ax, (0.57, y0 - 0.72), 0.35, h, "delta variance\n(B × G)", fc="#f8fafc")

    _box(ax, (0.05, y0 - 0.28), 0.2, h, "context_id", fc="#e2e8f0")
    _box(ax, (0.05, y0 - 0.42), 0.2, h, "context\nembedding", fc="#e2e8f0")

    _arrow(ax, (x + w / 2, y0), (x + w / 2, y0 - 0.04))
    _arrow(ax, (x + w / 2, y0 - 0.14), (x + w / 2, y0 - 0.18))
    _arrow(ax, (x + w / 2, y0 - 0.28), (x + w / 2, y0 - 0.32))
    _arrow(ax, (x + w / 2, y0 - 0.42), (x + w / 2, y0 - 0.46))
    _arrow(ax, (x + w / 2, y0 - 0.56), (0.255, y0 - 0.62))
    _arrow(ax, (x + w / 2, y0 - 0.56), (0.745, y0 - 0.62))

    _arrow(ax, (0.25, y0 - 0.23), (0.25, y0 - 0.28))
    _arrow(ax, (0.25, y0 - 0.37), (0.3, y0 - 0.23))
    _arrow(ax, (0.25, y0 - 0.42), (0.3, y0 - 0.56))

    ax.text(0.03, 0.95, "PerturbFM v2 (CGIO) sketch", fontsize=14, color="#0f172a", weight="bold")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=260 if fmt == "png" else None, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "docs" / "diagrams"
    pipeline(out_dir / "pipeline.svg", "svg")
    pipeline(out_dir / "pipeline.png", "png")
    cgio(out_dir / "cgio.svg", "svg")
    cgio(out_dir / "cgio.png", "png")


if __name__ == "__main__":
    main()
