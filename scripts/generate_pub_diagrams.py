#!/usr/bin/env python3
"""
Generate publication-style diagrams (SVG + PNG) for README/docs.

Outputs to:
  docs/diagrams/pub/
    pipeline_lane.svg/.png
    pipeline_compact.svg/.png
    cgio_modular.svg/.png
    cgio_operator.svg/.png
    landscape_matrix.svg/.png
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


def _configure_matplotlib() -> None:
    matplotlib.use("Agg")
    mpl.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 12,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",  # keep text as text in SVG
        }
    )


@dataclass(frozen=True)
class Style:
    bg: str = "#ffffff"
    ink: str = "#111827"
    muted: str = "#475569"
    panel: str = "#f8fafc"
    panel2: str = "#f1f5f9"
    accent: str = "#2563eb"
    accent2: str = "#7c3aed"
    ok: str = "#16a34a"
    warn: str = "#d97706"
    bad: str = "#dc2626"


STYLE = Style()


def _rounded(
    ax,
    x,
    y,
    w,
    h,
    text,
    fc,
    ec=None,
    lw=1.0,
    fontsize=12,
    weight="normal",
    align="center",
    fit_items: list | None = None,
    pad: float = 0.02,
):
    if ec is None:
        ec = STYLE.ink
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    text_obj = ax.text(
        x + (w / 2 if align == "center" else 0.03),
        y + h / 2,
        text,
        ha=("center" if align == "center" else "left"),
        va="center",
        fontsize=fontsize,
        color=STYLE.ink,
        weight=weight,
        linespacing=1.05,
    )
    if fit_items is not None:
        fit_items.append((text_obj, (x, y, w, h, pad)))
    return patch, text_obj


def _fit_text_items(ax, items: list, min_fontsize: int = 8) -> None:
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text_obj, (x, y, w, h, pad) in items:
        # available box in display coordinates (minus padding)
        (x0, y0) = ax.transData.transform((x + pad, y + pad))
        (x1, y1) = ax.transData.transform((x + w - pad, y + h - pad))
        box_w = abs(x1 - x0)
        box_h = abs(y1 - y0)
        fs = int(text_obj.get_fontsize())
        for new_fs in range(fs, min_fontsize - 1, -1):
            text_obj.set_fontsize(new_fs)
            fig.canvas.draw()
            bb = text_obj.get_window_extent(renderer=renderer)
            if bb.width <= box_w and bb.height <= box_h:
                break


def _arrow(ax, x1, y1, x2, y2, color=None, lw=1.2, rad=0.0):
    if color is None:
        color = STYLE.ink
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            lw=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
            connectionstyle=f"arc3,rad={rad}",
        ),
    )


def _title(ax, text):
    ax.text(0.02, 0.97, text, transform=ax.transAxes, ha="left", va="top", fontsize=16, weight="bold", color=STYLE.ink)


def _subtitle(ax, text):
    ax.text(0.02, 0.91, text, transform=ax.transAxes, ha="left", va="top", fontsize=12, color=STYLE.muted)


def diagram_pipeline_lane(out: Path, fmt: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.0))
    fig.patch.set_facecolor(STYLE.bg)
    ax.set_facecolor(STYLE.bg)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _title(ax, "PerturbFM pipeline (reproducible by design)")
    _subtitle(ax, "Immutable splits + full metric panels + uncertainty outputs; all runs log hashes.")

    # swimlanes
    lanes = [
        (0.05, 0.63, 0.9, 0.25, "Inputs & artifacts"),
        (0.05, 0.34, 0.9, 0.25, "Training / evaluation run"),
        (0.05, 0.05, 0.9, 0.25, "Outputs (run directory)"),
    ]
    for x, y, w, h, label in lanes:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=STYLE.panel2, edgecolor="#e2e8f0", linewidth=1.0))
        ax.text(x + 0.01, y + h - 0.04, label, ha="left", va="top", fontsize=12, color=STYLE.muted, weight="bold")

    fit_items = []
    # boxes
    _rounded(ax, 0.10, 0.68, 0.20, 0.14, "Dataset artifact\n(data.npz + meta.json)", STYLE.panel, fit_items=fit_items, fontsize=11)
    _rounded(ax, 0.34, 0.68, 0.20, 0.14, "SplitStore\n(splits/*.json)\nhash-locked", STYLE.panel, fit_items=fit_items, fontsize=11)
    _rounded(ax, 0.58, 0.68, 0.30, 0.14, "External benchmarks\n(third_party/)\nGPL-isolated harness", STYLE.panel, fit_items=fit_items, fontsize=11)

    _rounded(
        ax,
        0.12,
        0.39,
        0.76,
        0.14,
        "Train / Eval (CLI)\nbaseline | v0 | v1 | v2 (CGIO)   +   uncertainty (ensemble / conformal)",
        "#eef2ff",
        ec=STYLE.accent,
        lw=1.2,
        fit_items=fit_items,
        fontsize=11,
    )

    _rounded(ax, 0.10, 0.10, 0.24, 0.14, "predictions.npz\n(mean, var, idx)", STYLE.panel, fit_items=fit_items, fontsize=11)
    _rounded(ax, 0.38, 0.10, 0.24, 0.14, "metrics.json\n(scPerturBench + PerturBench)", STYLE.panel, fit_items=fit_items, fontsize=11)
    _rounded(ax, 0.66, 0.10, 0.22, 0.14, "calibration.json\n+ report.html", STYLE.panel, fit_items=fit_items, fontsize=11)

    # arrows
    _arrow(ax, 0.20, 0.68, 0.50, 0.53, color=STYLE.ink, lw=1.3, rad=-0.05)
    _arrow(ax, 0.44, 0.68, 0.50, 0.53, color=STYLE.ink, lw=1.3, rad=0.05)
    _arrow(ax, 0.72, 0.68, 0.62, 0.53, color=STYLE.ink, lw=1.1, rad=0.15)

    _arrow(ax, 0.50, 0.39, 0.21, 0.24, color=STYLE.ink, lw=1.3, rad=0.0)
    _arrow(ax, 0.50, 0.39, 0.48, 0.24, color=STYLE.ink, lw=1.3, rad=0.0)
    _arrow(ax, 0.50, 0.39, 0.75, 0.24, color=STYLE.ink, lw=1.3, rad=0.0)
    _arrow(ax, 0.32, 0.17, 0.37, 0.17, color=STYLE.ink, lw=1.0)
    _arrow(ax, 0.59, 0.17, 0.64, 0.17, color=STYLE.ink, lw=1.0)

    _fit_text_items(ax, fit_items, min_fontsize=8)
    fig.savefig(out, dpi=300 if fmt == "png" else None, bbox_inches="tight")
    plt.close(fig)


def diagram_pipeline_compact(out: Path, fmt: str) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 3.4))
    fig.patch.set_facecolor(STYLE.bg)
    ax.set_facecolor(STYLE.bg)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _title(ax, "PerturbFM: artifact-driven experimentation")
    _subtitle(ax, "Everything is an artifact: dataset, split, predictions, metrics.")

    fit_items = []
    _rounded(ax, 0.06, 0.62, 0.26, 0.18, "Dataset artifact", "#ecfeff", ec=STYLE.accent, lw=1.2, fontsize=13, weight="bold", fit_items=fit_items)
    _rounded(ax, 0.06, 0.36, 0.26, 0.18, "Split artifact\n(hash-locked)", "#f0fdf4", ec=STYLE.ok, lw=1.2, fontsize=12, fit_items=fit_items)
    _rounded(ax, 0.40, 0.48, 0.22, 0.20, "Run\n(train+eval)", "#eef2ff", ec=STYLE.accent2, lw=1.2, fontsize=13, weight="bold", fit_items=fit_items)
    _rounded(ax, 0.70, 0.62, 0.24, 0.18, "predictions.npz", STYLE.panel, fit_items=fit_items)
    _rounded(ax, 0.70, 0.36, 0.24, 0.18, "metrics + calibration", STYLE.panel, fit_items=fit_items)

    _arrow(ax, 0.32, 0.71, 0.40, 0.61, lw=1.4)
    _arrow(ax, 0.32, 0.45, 0.40, 0.55, lw=1.4)
    _arrow(ax, 0.62, 0.61, 0.70, 0.71, lw=1.4)
    _arrow(ax, 0.62, 0.55, 0.70, 0.45, lw=1.4)

    ax.text(0.70, 0.28, "report.html", fontsize=12, color=STYLE.muted)

    _fit_text_items(ax, fit_items, min_fontsize=8)
    fig.savefig(out, dpi=300 if fmt == "png" else None, bbox_inches="tight")
    plt.close(fig)


def diagram_cgio_modular(out: Path, fmt: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.2))
    fig.patch.set_facecolor(STYLE.bg)
    ax.set_facecolor(STYLE.bg)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _title(ax, "PerturbFM v2 (CGIO): Contextual Graph Intervention Operator")
    _subtitle(ax, "Inspired by prior art (CPA/scGen, GEARS/TxPert, GPerturb) but designed for strict context-OOD + calibration.")

    fit_items = []
    # Inputs column
    _rounded(ax, 0.05, 0.70, 0.22, 0.18, "Perturbation\n(pert_genes)\n→ gene-set intervention", STYLE.panel, fit_items=fit_items, fontsize=11)
    _rounded(ax, 0.05, 0.48, 0.22, 0.16, "Context\n(context_id)\n+ covariates", STYLE.panel, fit_items=fit_items, fontsize=11)
    _rounded(ax, 0.05, 0.26, 0.22, 0.16, "Gene graph(s)\n(STRING/pathways)\n+ learned trust", STYLE.panel, fit_items=fit_items, fontsize=11)

    # Encoders
    _rounded(ax, 0.33, 0.67, 0.24, 0.18, "Intervention encoder\nmask → propagation\n(+ gating + mixture)", "#f0fdf4", ec=STYLE.ok, lw=1.2, fit_items=fit_items, fontsize=11)
    _rounded(ax, 0.33, 0.47, 0.24, 0.12, "Context encoder", "#ecfeff", ec=STYLE.accent, lw=1.2, fit_items=fit_items, fontsize=11)

    # Operator + heads
    _rounded(ax, 0.62, 0.63, 0.30, 0.22, "Contextual low-rank operator\n$\\mu_\\Delta = B(c)\\,h$", "#eef2ff", ec=STYLE.accent2, lw=1.2, fit_items=fit_items, fontsize=11)
    _rounded(ax, 0.62, 0.38, 0.30, 0.18, "Uncertainty heads\naleatoric: $\\sigma^2_{a}(c,h)$\nepistemic: ensembles", STYLE.panel, fit_items=fit_items, fontsize=11)

    # Outputs
    _rounded(ax, 0.78, 0.14, 0.18, 0.18, "Outputs\n$\\mu_\\Delta$, $\\sigma^2$\n(calibration eval)", STYLE.panel2, ec="#cbd5e1", fit_items=fit_items, fontsize=11)

    # Arrows (clean L->R)
    _arrow(ax, 0.27, 0.79, 0.33, 0.76, lw=1.4)
    _arrow(ax, 0.27, 0.56, 0.33, 0.53, lw=1.2)
    _arrow(ax, 0.27, 0.34, 0.33, 0.73, lw=1.2, rad=0.15)

    _arrow(ax, 0.57, 0.76, 0.62, 0.74, lw=1.5)
    _arrow(ax, 0.57, 0.53, 0.62, 0.72, lw=1.2, rad=-0.15)
    _arrow(ax, 0.77, 0.63, 0.87, 0.32, lw=1.3, rad=0.0)
    _arrow(ax, 0.77, 0.45, 0.87, 0.32, lw=1.1, rad=0.15)

    # Prior art callout
    _rounded(
        ax,
        0.33,
        0.18,
        0.59,
        0.14,
        "Prior art touchpoints:\n"
        "CPA/scGen: decomposition + perturbation representation   |   "
        "GEARS/TxPert: graph priors   |   "
        "GPerturb: uncertainty emphasis",
        "#ffffff",
        ec="#e2e8f0",
        lw=1.0,
        fontsize=11,
        align="left",
    )

    _fit_text_items(ax, fit_items, min_fontsize=8)
    fig.savefig(out, dpi=300 if fmt == "png" else None, bbox_inches="tight")
    plt.close(fig)


def diagram_cgio_operator(out: Path, fmt: str) -> None:
    fig, ax = plt.subplots(figsize=(12.2, 5.4))
    fig.patch.set_facecolor(STYLE.bg)
    ax.set_facecolor(STYLE.bg)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _title(ax, "PerturbFM v2 (CGIO): decomposition + operator view")
    _subtitle(ax, "CPA/scGen-style additive structure, with CGIO providing the perturbation-effect branch.")

    fit_items = []

    # ---------------------------------------------------------------------
    # Panel A: decomposition (top)
    ax.add_patch(Rectangle((0.04, 0.63), 0.92, 0.27, facecolor=STYLE.panel2, edgecolor="#e2e8f0", linewidth=1.0))
    ax.text(0.06, 0.88, "A) Decomposition (CPA/scGen style)", fontsize=12, color=STYLE.muted, weight="bold")

    _rounded(ax, 0.07, 0.70, 0.22, 0.14, "basal_state\n= X_control", STYLE.panel, fit_items=fit_items, fontsize=12)
    _rounded(
        ax,
        0.33,
        0.70,
        0.22,
        0.14,
        "systematic_shift\n= s(context, covariates)",
        STYLE.panel,
        fit_items=fit_items,
        fontsize=11,
    )
    _rounded(
        ax,
        0.59,
        0.70,
        0.22,
        0.14,
        "perturbation_effect\n= Δ̂(pert_genes, graph, context)",
        STYLE.panel,
        fit_items=fit_items,
        fontsize=10,
    )
    ax.text(0.30, 0.77, "+", fontsize=18, color=STYLE.ink, weight="bold", ha="center", va="center")
    ax.text(0.56, 0.77, "+", fontsize=18, color=STYLE.ink, weight="bold", ha="center", va="center")
    ax.text(0.07, 0.655, "X̂_pert = X_control + s(context) + Δ̂", fontsize=12, color=STYLE.ink)

    # ---------------------------------------------------------------------
    # Panel B: how CGIO computes the perturbation_effect (bottom)
    ax.add_patch(Rectangle((0.04, 0.08), 0.92, 0.50, facecolor="#ffffff", edgecolor="#e2e8f0", linewidth=1.0))
    ax.text(0.06, 0.55, "B) CGIO branch (perturbation_effect)", fontsize=12, color=STYLE.muted, weight="bold")

    _rounded(ax, 0.07, 0.38, 0.22, 0.16, "Intervention\npert_genes → mask g", STYLE.panel, fit_items=fit_items, fontsize=11)
    _rounded(ax, 0.33, 0.38, 0.22, 0.16, "Graph propagation\nh = f_graph(g; A, τ)", "#f0fdf4", ec=STYLE.ok, lw=1.2, fit_items=fit_items, fontsize=11)
    _rounded(ax, 0.59, 0.38, 0.28, 0.16, "Operator\nΔ̂ = B(c) h", "#eef2ff", ec=STYLE.accent2, lw=1.2, fit_items=fit_items, fontsize=12)

    _rounded(ax, 0.33, 0.18, 0.22, 0.14, "Context\nc = embed(context)\n(+ covariates)", "#ecfeff", ec=STYLE.accent, lw=1.2, fit_items=fit_items, fontsize=10)
    _rounded(ax, 0.59, 0.20, 0.28, 0.12, "Aleatoric variance\nσ_a^2 = v(c,h)", STYLE.panel, fit_items=fit_items, fontsize=11)
    _rounded(ax, 0.59, 0.08, 0.28, 0.10, "Epistemic\nensembles → σ_e^2", STYLE.panel, fit_items=fit_items, fontsize=11)

    # arrows (clean left-to-right)
    _arrow(ax, 0.29, 0.46, 0.33, 0.46, lw=1.4)
    _arrow(ax, 0.55, 0.46, 0.59, 0.46, lw=1.4)
    _arrow(ax, 0.44, 0.32, 0.62, 0.38, lw=1.2, rad=0.12)  # context -> operator
    _arrow(ax, 0.73, 0.38, 0.73, 0.32, lw=1.1)  # operator -> aleatoric
    _arrow(ax, 0.73, 0.20, 0.73, 0.18, lw=1.1)  # aleatoric -> epistemic (visual stack)

    # dashed linkage from panel B output to decomposition term
    ax.annotate(
        "",
        xy=(0.70, 0.70),
        xytext=(0.73, 0.54),
        arrowprops=dict(arrowstyle="-|>", lw=1.2, color=STYLE.muted, linestyle="--"),
    )
    ax.text(0.74, 0.62, "provides Δ̂", fontsize=10, color=STYLE.muted)
    ax.annotate(
        "",
        xy=(0.44, 0.70),
        xytext=(0.44, 0.32),
        arrowprops=dict(arrowstyle="-|>", lw=1.0, color=STYLE.muted, linestyle="--"),
    )
    ax.text(0.46, 0.52, "feeds s(·)", fontsize=10, color=STYLE.muted, rotation=90, va="center")

    _fit_text_items(ax, fit_items, min_fontsize=8)
    fig.savefig(out, dpi=300 if fmt == "png" else None, bbox_inches="tight")
    plt.close(fig)


def diagram_landscape_matrix(out: Path, fmt: str) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    fig.patch.set_facecolor(STYLE.bg)
    ax.set_facecolor(STYLE.bg)
    ax.axis("off")

    _title(ax, "Landscape (high-level): how CGIO combines the strongest ideas")
    _subtitle(ax, "This is a qualitative summary of themes from current_state.md (not a benchmark result).")

    rows = ["scGen", "CPA", "CellOT", "GEARS", "TxPert", "PerturbNet", "GPerturb", "PerturbFM v2 (CGIO)"]
    cols = ["Context OOD", "Pert OOD", "Distributional", "Calibrated uncertainty", "Graph prior"]
    # states: strong, partial, absent, mixed
    M = [
        ["partial", "partial", "partial", "absent", "absent"],
        ["partial", "partial", "partial", "partial", "absent"],
        ["partial", "absent", "strong", "absent", "absent"],
        ["absent", "partial", "absent", "absent", "strong"],
        ["partial", "partial", "absent", "absent", "strong"],
        ["partial", "partial", "strong", "partial", "absent"],
        ["partial", "partial", "absent", "strong", "absent"],
        ["strong", "partial", "mixed", "strong", "strong"],
    ]

    def draw_mark(xc: float, yc: float, state: str) -> None:
        if state == "strong":
            ax.add_patch(plt.Circle((xc, yc), 0.015, color=STYLE.ok, ec="none"))
            return
        if state == "partial":
            ax.add_patch(plt.Circle((xc, yc), 0.015, color=STYLE.warn, alpha=0.25, ec=STYLE.warn, lw=1.2))
            return
        if state == "mixed":
            ax.add_patch(plt.Circle((xc, yc), 0.015, color=STYLE.ok, alpha=0.25, ec=STYLE.warn, lw=1.4))
            ax.add_patch(plt.Circle((xc, yc), 0.006, color=STYLE.ok, ec="none"))
            return
        # absent
        ax.plot([xc - 0.012, xc + 0.012], [yc - 0.012, yc + 0.012], color=STYLE.bad, lw=2.0)
        ax.plot([xc - 0.012, xc + 0.012], [yc + 0.012, yc - 0.012], color=STYLE.bad, lw=2.0)

    # draw table
    x0, y0 = 0.06, 0.18
    cell_w, cell_h = 0.16, 0.085
    # headers
    ax.text(x0, y0 + cell_h * (len(rows) + 1) + 0.02, "Model", fontsize=11, color=STYLE.muted, weight="bold")
    for j, col in enumerate(cols):
        ax.text(x0 + 0.24 + j * cell_w, y0 + cell_h * (len(rows) + 1) + 0.02, col, fontsize=11, color=STYLE.muted, weight="bold")

    # body
    for i, r in enumerate(rows):
        y = y0 + cell_h * (len(rows) - 1 - i)
        ax.text(x0, y + cell_h / 2, r, ha="left", va="center", fontsize=11, color=STYLE.ink)
        for j, state in enumerate(M[i]):
            x = x0 + 0.24 + j * cell_w
            ax.add_patch(Rectangle((x, y), cell_w, cell_h, facecolor="#ffffff", edgecolor="#e2e8f0", linewidth=1.0))
            draw_mark(x + cell_w / 2, y + cell_h / 2, state)

    ax.text(
        0.06,
        0.10,
        "Legend: green dot = strong/explicit   orange circle = partial/implicit   red X = absent/unclear   green+orange = mixed",
        fontsize=11,
        color=STYLE.muted,
    )

    fig.savefig(out, dpi=300 if fmt == "png" else None, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _configure_matplotlib()
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "docs" / "diagrams" / "pub"
    out_dir.mkdir(parents=True, exist_ok=True)

    for fmt in ["svg", "png"]:
        diagram_pipeline_lane(out_dir / f"pipeline_lane.{fmt}", fmt)
        diagram_pipeline_compact(out_dir / f"pipeline_compact.{fmt}", fmt)
        diagram_cgio_modular(out_dir / f"cgio_modular.{fmt}", fmt)
        diagram_cgio_operator(out_dir / f"cgio_operator.{fmt}", fmt)


if __name__ == "__main__":
    main()
