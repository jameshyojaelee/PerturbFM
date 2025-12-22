"""HTML report generation."""

from __future__ import annotations

import json
from typing import Dict


def render_report(metrics: Dict[str, object]) -> str:
    payload = json.dumps(metrics, indent=2)
    return f\"\"\"<!doctype html>
<html>
  <head><meta charset=\\"utf-8\\"><title>PerturbFM Report</title></head>
  <body>
    <h1>PerturbFM Report</h1>
    <pre>{payload}</pre>
  </body>
</html>
\"\"\"
