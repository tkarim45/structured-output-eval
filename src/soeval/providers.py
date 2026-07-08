"""Providers run a task under one strategy and return the model's raw text.

ClaudeProvider implements the four strategies against the real API. MockProvider simulates
each strategy's characteristic failure modes deterministically, so the benchmark — and the
"constrained decoding fixes validity" finding — runs offline without a key.
"""
from __future__ import annotations

import json
import os
import random

from .config import MODEL

SYSTEM = ("Extract the requested fields from the input and return a single JSON object. "
          "Respond with ONLY the JSON — no preamble, no explanation, no markdown fences.")


def _strict_schema(schema):
    """Structured outputs / strict tools require every object schema to explicitly set
    ``additionalProperties: false`` (Bedrock enforces this; a 400 otherwise). Deep-copy the schema
    with that added on every object node, without mutating the task's original schema."""
    if isinstance(schema, dict):
        out = {k: _strict_schema(v) for k, v in schema.items()}
        if out.get("type") == "object" or "properties" in out:
            out.setdefault("additionalProperties", False)
        return out
    if isinstance(schema, list):
        return [_strict_schema(v) for v in schema]
    return schema


def _dummy(prop: dict):
    t = prop.get("type")
    t = t[0] if isinstance(t, list) else t
    return {"string": "unknown", "number": 0.0, "integer": 0, "boolean": False,
            "array": [], "object": {}, "null": None}.get(t, "unknown")


def ideal_object(task: dict) -> dict:
    """A schema-valid object that's correct on the gold fields (the target output)."""
    obj = dict(task["gold"])
    props = task["schema"].get("properties", {})
    for k in task["schema"].get("required", []):
        if k not in obj:
            obj[k] = _dummy(props.get(k, {}))
    return obj


# -----------------------------------------------------------------------------
class MockProvider:
    name = "mock"

    # per-strategy (p_malformed_json, p_drop_field, p_type_break, p_value_wrong)
    PROFILES = {
        "plain":            (0.30, 0.25, 0.20, 0.10),
        "schema_in_prompt": (0.10, 0.10, 0.08, 0.08),
        "native":           (0.00, 0.00, 0.00, 0.05),  # constrained: always valid+schema
        "strict_tool":      (0.00, 0.00, 0.00, 0.04),
    }

    def generate(self, task: dict, strategy: str) -> str:
        rng = random.Random(hash((task["id"], strategy)) & 0xFFFFFFFF)
        obj = ideal_object(task)
        p_bad, p_drop, p_type, p_val = self.PROFILES[strategy]
        req = task["schema"].get("required", [])

        if rng.random() < p_drop and len(req) > 1:        # drop a required field -> schema fail
            obj.pop(rng.choice(req), None)
        elif rng.random() < p_type:                       # wrong type on a numeric field -> schema fail
            for k, v in list(obj.items()):
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    obj[k] = str(v)
                    break
        elif rng.random() < p_val and task["gold"]:       # wrong VALUE (schema-valid, accuracy hit)
            k = rng.choice(list(task["gold"]))
            obj[k] = obj[k] + "_x" if isinstance(obj[k], str) else obj[k]

        text = json.dumps(obj)
        if rng.random() < p_bad:                          # trailing comma -> genuinely invalid JSON
            return text[:-1].rstrip() + ",}\nHere you go!"
        return text


# -----------------------------------------------------------------------------
class ClaudeProvider:
    def __init__(self, model: str = MODEL):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model

    @property
    def name(self):
        return self.model

    def generate(self, task: dict, strategy: str) -> str:
        if strategy == "native":
            return self._native(task)
        if strategy == "strict_tool":
            return self._strict_tool(task)
        return self._prompted(task, with_schema=(strategy == "schema_in_prompt"))

    def _prompted(self, task, with_schema: bool) -> str:
        prompt = f"Input:\n{task['input']}"
        if with_schema:
            prompt += f"\n\nReturn JSON matching this schema:\n{json.dumps(task['schema'])}"
        resp = self.client.messages.create(model=self.model, max_tokens=512, system=SYSTEM,
                                           messages=[{"role": "user", "content": prompt}])
        return next((b.text for b in resp.content if b.type == "text"), "")

    def _native(self, task) -> str:
        resp = self.client.messages.create(
            model=self.model, max_tokens=512,
            messages=[{"role": "user", "content": f"Extract fields. Input:\n{task['input']}"}],
            output_config={"format": {"type": "json_schema", "schema": _strict_schema(task["schema"])}})
        return next((b.text for b in resp.content if b.type == "text"), "")

    def _strict_tool(self, task) -> str:
        resp = self.client.messages.create(
            model=self.model, max_tokens=512,
            messages=[{"role": "user", "content": f"Extract fields. Input:\n{task['input']}"}],
            tools=[{"name": "record", "description": "Record the extracted fields.",
                    "strict": True, "input_schema": _strict_schema(task["schema"])}],
            tool_choice={"type": "tool", "name": "record"})
        for b in resp.content:
            if b.type == "tool_use":
                return json.dumps(b.input)
        return ""


class BedrockProvider(ClaudeProvider):
    """Same four strategies as ClaudeProvider, but Claude on AWS Bedrock — for machines that have AWS
    credentials but no direct ANTHROPIC_API_KEY. output_config.format and strict tool use are both
    supported on Bedrock, so the native/strict_tool strategies work unchanged."""

    def __init__(self, model: str | None = None):
        from anthropic import AnthropicBedrock

        self.client = AnthropicBedrock(aws_region=os.getenv("AWS_REGION", "us-east-1"))
        self.model = model or os.getenv(
            "SOEVAL_BEDROCK_MODEL", "global.anthropic.claude-haiku-4-5-20251001-v1:0")


def _has_aws() -> bool:
    return bool(os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("AWS_PROFILE"))


def get_provider():
    """Prefer a real Claude if any credential is available (Bedrock or direct API), else the
    calibrated mock. Force a choice with SOEVAL_PROVIDER=bedrock|anthropic|mock."""
    forced = os.getenv("SOEVAL_PROVIDER", "").lower()
    if forced == "mock":
        return MockProvider()
    if forced == "bedrock" or (not forced and _has_aws()):
        return BedrockProvider()
    if forced == "anthropic" or os.getenv("ANTHROPIC_API_KEY"):
        return ClaudeProvider()
    return MockProvider()
