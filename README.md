# 🔢 Structured-Output Reliability Eval

> Agents break when the model emits malformed JSON or the wrong tool arguments. This
> benchmarks **structured-output reliability** across prompting strategies — **plain prompt
> → schema-in-prompt → native structured outputs → strict tool use** — scoring **valid-JSON
> rate, JSON-Schema adherence, and field-level accuracy**, including under **adversarial
> inputs** (missing fields, prompt injection, preamble bait). Runs offline (calibrated mock);
> plug in a key for live Claude numbers.

"Just parse the JSON" is where production LLM pipelines silently fail. The question isn't
*can* the model return JSON — it's *how often, how schema-correct, and how much does
constrained decoding help*. This measures exactly that, per strategy.

---

## The strategies (weakest → strongest constraint)

| Strategy | What it does |
|---|---|
| `plain` | "return JSON" with no schema — most prone to preamble, fences, malformed output |
| `schema_in_prompt` | the JSON Schema pasted into the prompt — better, still unconstrained |
| `native` | `output_config.format` JSON-Schema (**constrained decoding**) — valid by construction |
| `strict_tool` | a `strict: true` tool whose `input_schema` is the target schema |

---

## Measured (`soeval`)

```
$ soeval
provider: mock · 8 tasks
------------------------------------------------------------------------
strategy            valid_json   schema  field_acc  adv_schema
plain                    0.500    0.375      0.438       0.667
schema_in_prompt         0.875    0.750      0.771       1.000
native                   1.000    1.000      1.000       1.000
strict_tool              1.000    1.000      0.938       1.000
```

The headline: **constrained decoding (`native` / `strict_tool`) takes valid-schema rate to
100%**, while `plain` leaks malformed JSON and schema violations — and the gap is *widest on
adversarial inputs*. The mock's failure rates are calibrated to real model behavior so the
finding holds offline; run `--provider claude` for live numbers on `claude-opus-4-8`.

---

## Quickstart

> Uses the conda **`personal`** env (per environment conventions — never `base`).

```bash
PY=~/miniconda3/envs/personal/bin/python
$PY -m pip install -e ".[all]"

soeval                          # offline reliability benchmark (mock provider)

export ANTHROPIC_API_KEY=sk-ant-...
soeval --provider claude        # live: plain vs schema vs native vs strict tool on Claude
```

Each run writes `reports/report_<provider>.json` (per-strategy summary + per-task rows).

---

## What's scored

| Metric | Question |
|---|---|
| **valid_json_rate** | did the output parse as JSON at all? (lenient: fences/preamble recovered) |
| **schema_rate** | does the parsed object satisfy the JSON Schema? (`jsonschema` Draft 2020-12) |
| **field_accuracy** | fraction of gold fields extracted correctly (case/number-normalized) |
| **adversarial_schema_rate** | schema adherence on the adversarial subset only |

---

## Repo layout

```
structured-output-eval/
├── src/soeval/
│   ├── validate.py   JSON extraction · schema validation · field accuracy
│   ├── providers.py  Claude strategies (plain/schema/native/strict_tool) + calibrated mock
│   ├── harness.py    run strategies × tasks → leaderboard  (CLI: soeval)
│   └── config.py     model, strategies, paths
├── tasks/tasks.yaml  labeled extraction/classification tasks (clean + adversarial)
├── tests/            validator + harness tests (key-free) — 5 cases
└── pyproject.toml · Dockerfile · Makefile · .github/workflows/ci.yml
```

---

## Résumé framing

> *Built a structured-output reliability benchmark for LLMs — valid-JSON rate, JSON-Schema
> adherence, and field accuracy across plain / schema-in-prompt / native structured-output /
> strict-tool strategies under adversarial inputs; showed constrained decoding raises
> valid-schema rate to 100% vs prompting, with the gap widest on adversarial cases.*

## License
MIT (`LICENSE`).
