"""Model, strategies, paths."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.getenv("SOEVAL_ROOT", Path(__file__).resolve().parents[2]))
TASKS_PATH = ROOT / "tasks" / "tasks.yaml"
REPORTS = ROOT / "reports"

MODEL = os.getenv("SOEVAL_MODEL", "claude-opus-4-8")

# Prompting strategies, weakest → strongest constraint:
#   plain            : "return JSON" with no schema
#   schema_in_prompt : the JSON Schema pasted into the prompt
#   native           : output_config.format json_schema (constrained decoding)
#   strict_tool      : a strict tool whose input_schema is the target schema
STRATEGIES = ["plain", "schema_in_prompt", "native", "strict_tool"]
