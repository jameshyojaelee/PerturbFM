"""HTML report generation."""

from __future__ import annotations

import json
from typing import Dict


def _table(panel: Dict[str, float], title: str) -> str:
    rows = "".join(f"<tr><td>{k}</td><td>{panel[k]}</td></tr>" for k in panel)
    return f"<h3>{title}</h3><table border='1' cellspacing='0' cellpadding='4'><tr><th>Metric</th><th>Value</th></tr>{rows}</table>"


def render_report(metrics: Dict[str, object]) -> str:
    sc = metrics.get("scperturbench", {}).get("global", {})
    pb = metrics.get("perturbench", {}).get("global", {})
    unc = metrics.get("uncertainty", {})
    coverage = unc.get("coverage", {}) if isinstance(unc, dict) else {}
    tables = []
    if sc:
        tables.append(_table(sc, "scPerturBench (global)"))
    if pb:
        tables.append(_table(pb, "PerturBench (global)"))
    unc_summary = {}
    if isinstance(unc, dict):
        if "nll" in unc:
            unc_summary["NLL"] = unc["nll"]
        if "ood_auroc" in unc:
            unc_summary["OOD_AUROC"] = unc["ood_auroc"]
    if unc_summary:
        tables.append(_table(unc_summary, "Uncertainty (summary)"))
    if coverage:
        tables.append(_table(coverage, "Coverage"))

    return f"""<!doctype html>
<html>
  <head><meta charset="utf-8"><title>PerturbFM Report</title></head>
  <body>
    <h1>PerturbFM Report</h1>
    {''.join(tables)}
    <h3>Raw JSON</h3>
    <pre>{json.dumps(metrics, indent=2)}</pre>
  </body>
</html>
"""
