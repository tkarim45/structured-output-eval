"""Run every strategy over every task, score validity / schema-adherence / field accuracy,
and print a leaderboard. Shows how constrained decoding (native / strict tool) lifts
valid-schema rate over plain prompting.

    python -m soeval.harness                 # offline mock (no key)
    python -m soeval.harness --provider claude   # live Claude (needs ANTHROPIC_API_KEY)
"""
from __future__ import annotations

import argparse
import json

import yaml

from .config import REPORTS, STRATEGIES, TASKS_PATH
from .providers import ClaudeProvider, MockProvider
from .validate import extract_json, field_accuracy, schema_adherent


def run(tasks: list[dict], provider, strategies=STRATEGIES) -> dict:
    out = {}
    rows = []
    for strat in strategies:
        n = vj = sa = 0
        facc = 0.0
        adv_sa = adv_n = 0
        for t in tasks:
            raw = provider.generate(t, strat)
            obj, valid = extract_json(raw)
            ok_schema = schema_adherent(obj, t["schema"])
            fa = field_accuracy(obj, t["gold"])
            n += 1; vj += valid; sa += ok_schema; facc += fa
            if t.get("category") == "adversarial":
                adv_n += 1; adv_sa += ok_schema
            rows.append({"strategy": strat, "task": t["id"], "category": t.get("category"),
                         "valid_json": valid, "schema_ok": ok_schema, "field_acc": round(fa, 3)})
        out[strat] = {
            "valid_json_rate": round(vj / n, 3),
            "schema_rate": round(sa / n, 3),
            "field_accuracy": round(facc / n, 3),
            "adversarial_schema_rate": round(adv_sa / adv_n, 3) if adv_n else None,
            "n": n,
        }
    return {"summary": out, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description="structured-output reliability benchmark")
    ap.add_argument("--provider", choices=["mock", "claude"], default="mock")
    args = ap.parse_args()

    tasks = yaml.safe_load(open(TASKS_PATH))["tasks"]
    provider = ClaudeProvider() if args.provider == "claude" else MockProvider()
    res = run(tasks, provider)

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"report_{args.provider}.json").write_text(json.dumps(res, indent=2, default=str))

    print(f"\nprovider: {provider.name} · {res['summary'][STRATEGIES[0]]['n']} tasks")
    print("-" * 72)
    print(f"{'strategy':18} {'valid_json':>11} {'schema':>8} {'field_acc':>10} {'adv_schema':>11}")
    for strat, m in res["summary"].items():
        adv = "—" if m["adversarial_schema_rate"] is None else f"{m['adversarial_schema_rate']:.3f}"
        print(f"{strat:18} {m['valid_json_rate']:>11.3f} {m['schema_rate']:>8.3f} "
              f"{m['field_accuracy']:>10.3f} {adv:>11}")
    print("-" * 72)
    best = max(res["summary"], key=lambda s: res["summary"][s]["schema_rate"])
    print(f"highest schema-adherence: {best} ({res['summary'][best]['schema_rate']:.0%})")


if __name__ == "__main__":
    main()
