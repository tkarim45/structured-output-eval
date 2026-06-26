"""Validator + harness tests (key-free, deterministic mock)."""
import yaml

from soeval.config import TASKS_PATH
from soeval.harness import run
from soeval.providers import MockProvider
from soeval.validate import extract_json, field_accuracy, schema_adherent

SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}, "n": {"type": "integer"}},
          "required": ["name", "n"], "additionalProperties": False}


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == ({"a": 1}, True)
    assert extract_json('```json\n{"a": 1}\n```')[1] is True            # fenced
    assert extract_json('Here is it: {"a": 1}')[1] is True              # recover from preamble
    assert extract_json('{"a": 1,}\nHere you go!')[1] is False          # trailing comma -> invalid
    assert extract_json("not json at all") == (None, False)


def test_schema_adherent():
    assert schema_adherent({"name": "x", "n": 2}, SCHEMA) is True
    assert schema_adherent({"name": "x"}, SCHEMA) is False              # missing required
    assert schema_adherent({"name": "x", "n": "2"}, SCHEMA) is False    # wrong type


def test_field_accuracy():
    assert field_accuracy({"name": "Jane Doe", "n": 3}, {"name": "jane doe"}) == 1.0
    assert field_accuracy({"name": "Bob"}, {"name": "Alice", "n": 1}) == 0.0
    assert field_accuracy({"n": 5}, {"name": "x", "n": 5}) == 0.5


def _tasks():
    return yaml.safe_load(open(TASKS_PATH))["tasks"]


def test_constrained_strategies_are_perfectly_valid():
    res = run(_tasks(), MockProvider())["summary"]
    for strat in ("native", "strict_tool"):
        assert res[strat]["valid_json_rate"] == 1.0
        assert res[strat]["schema_rate"] == 1.0


def test_constrained_beats_plain_prompting():
    res = run(_tasks(), MockProvider())["summary"]
    assert res["plain"]["valid_json_rate"] < 1.0                        # plain emits malformed JSON
    assert res["plain"]["schema_rate"] < res["native"]["schema_rate"]   # and breaks the schema
    assert res["schema_in_prompt"]["schema_rate"] > res["plain"]["schema_rate"]  # schema-in-prompt helps
