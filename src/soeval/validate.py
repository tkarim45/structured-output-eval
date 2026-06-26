"""Parse + validate model output: is it valid JSON? does it satisfy the schema? how many
fields match the gold object? These three are the axes a structured-output system is judged on.
"""
from __future__ import annotations

import json
import re

from jsonschema import Draft202012Validator

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json(text: str):
    """Best-effort parse: strip ``` fences, else grab the first {...} block.
    Returns (obj, valid_json: bool)."""
    if text is None:
        return None, False
    candidates = []
    m = _FENCE.search(text)
    if m:
        candidates.append(m.group(1))
    candidates.append(text.strip())
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        candidates.append(brace.group(0))
    for c in candidates:
        try:
            return json.loads(c), True
        except (json.JSONDecodeError, TypeError):
            continue
    return None, False


def schema_adherent(obj, schema: dict) -> bool:
    if obj is None:
        return False
    return not list(Draft202012Validator(schema).iter_errors(obj))


def field_accuracy(obj, gold: dict) -> float:
    """Fraction of gold fields the object reproduces exactly (case/space-normalized strings)."""
    if not isinstance(obj, dict) or not gold:
        return 0.0
    correct = 0
    for k, v in gold.items():
        got = obj.get(k)
        if _match(got, v):
            correct += 1
    return correct / len(gold)


def _match(a, b) -> bool:
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) < 1e-6
    if isinstance(a, list) and isinstance(b, list):
        return [str(x).strip().lower() for x in a] == [str(x).strip().lower() for x in b]
    return a == b
