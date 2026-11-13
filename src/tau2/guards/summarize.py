"""Summarize guard decisions in a tau2 results file.

    python -m tau2.guards.summarize data/simulations/<run>/results.json

Works for both the single-file and the directory results formats. Reads the
``raw_data["guard"]`` log that ``GuardedAgent`` attaches to assistant messages.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


def load_simulations(path: Path) -> list[dict]:
    path = Path(path)
    meta_path = path / "results.json" if path.is_dir() else path
    with open(meta_path) as f:
        data = json.load(f)
    sims = data.get("simulations") or []
    sims_dir = meta_path.parent / "simulations"
    if not sims and sims_dir.is_dir():
        for p in sorted(sims_dir.glob("*.json")):
            with open(p) as f:
                sims.append(json.load(f))
    return sims


def summarize_simulation(sim: dict) -> dict:
    decisions, blocked_attempts = [], 0
    guard_cost, guard_latency = 0.0, 0.0
    for m in sim.get("messages") or []:
        log = (
            (m.get("raw_data") or {}).get("guard")
            if m.get("role") == "assistant"
            else None
        )
        if not log:
            continue
        decisions.extend(log.get("decisions", []))
        blocked_attempts += len(log.get("blocked_attempts", []))
        guard_cost += log.get("guard_cost") or 0.0
        guard_latency += log.get("guard_latency_seconds") or 0.0
    blocked = [d for d in decisions if not d.get("allow")]
    return {
        "task_id": sim.get("task_id"),
        "trial": sim.get("trial"),
        "reward": (sim.get("reward_info") or {}).get("reward"),
        "termination": sim.get("termination_reason"),
        "guard_checks": len(decisions),
        "blocked_calls": len(blocked),
        "blocked_attempts": blocked_attempts,
        "blocked_tools": dict(Counter(d["tool_call"]["name"] for d in blocked)),
        "guard_errors": sum(1 for d in decisions if d.get("error")),
        "cached_checks": sum(1 for d in decisions if d.get("cached")),
        "guard_cost": guard_cost,
        "guard_latency_seconds": guard_latency,
    }


def summarize(sims: Iterable[dict]) -> dict:
    rows = [summarize_simulation(s) for s in sims]
    checks = sum(r["guard_checks"] for r in rows)
    blocked = sum(r["blocked_calls"] for r in rows)
    return {
        "simulations": len(rows),
        "guard_checks": checks,
        "blocked_calls": blocked,
        "block_rate": blocked / checks if checks else None,
        "sims_with_a_block": sum(1 for r in rows if r["blocked_calls"]),
        "guard_errors": sum(r["guard_errors"] for r in rows),
        "cached_checks": sum(r["cached_checks"] for r in rows),
        "guard_cost": sum(r["guard_cost"] for r in rows),
        "mean_guard_latency_per_check": (
            sum(r["guard_latency_seconds"] for r in rows) / checks if checks else None
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path)
    parser.add_argument(
        "--json", action="store_true", help="print the full summary as JSON"
    )
    args = parser.parse_args()
    summary = summarize(load_simulations(args.results))
    if args.json:
        print(json.dumps(summary, indent=2))
        return
    for r in summary["rows"]:
        print(
            f"task {r['task_id']!s:>4} trial {r['trial']!s:>2}  reward {r['reward']!s:>4}  "
            f"checks {r['guard_checks']:>2}  blocked {r['blocked_calls']:>2} "
            f"{r['blocked_tools'] or ''}  errors {r['guard_errors']}"
        )
    agg = {k: v for k, v in summary.items() if k != "rows"}
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
