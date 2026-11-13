"""Print a task definition or one simulation, with every guard decision.

    python src/experiments/guards/show_sim.py task <task_id> [--domain airline]
    python src/experiments/guards/show_sim.py sim <results_dir_or_json> <task_id> <trial>

Used for the audits in ``audits/``. Trials are 0-indexed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tau2.guards.summarize import load_simulations


def short(x, n: int = 600) -> str:
    s = x if isinstance(x, str) else json.dumps(x)
    return s if len(s) <= n else s[:n] + f" ...[{len(s) - n} more chars]"


def show_task(domain: str, task_id: str) -> None:
    tasks = json.loads(Path(f"data/tau2/domains/{domain}/tasks.json").read_text())
    t = next(t for t in tasks if str(t["id"]) == str(task_id))
    print(f"TASK {task_id}")
    print("User scenario:", json.dumps(t.get("user_scenario"), indent=1))
    ec = t.get("evaluation_criteria") or {}
    print("\nGold actions (scoring compares the final DB with replaying these):")
    for a in ec.get("actions") or []:
        print(f"  - {a['name']}({json.dumps(a['arguments'])})")
    if ec.get("communicate_info"):
        print("Must communicate:", ec["communicate_info"])
    if ec.get("nl_assertions"):
        print("NL assertions:", ec["nl_assertions"])


def show_sim(results: Path, task_id: str, trial: int) -> None:
    s = next(
        s
        for s in load_simulations(results)
        if str(s["task_id"]) == str(task_id)
        and s["trial"] == trial
        and s["termination_reason"] != "infrastructure_error"
    )
    ri = s["reward_info"]
    print(
        f"task {task_id} trial {trial}: reward={ri['reward']} "
        f"db_match={ri['db_check']['db_match']} termination={s['termination_reason']}"
    )
    for i, m in enumerate(s["messages"]):
        role = m["role"]
        if role == "tool":
            err = " ERROR" if m.get("error") else ""
            print(
                f"[{i}] TOOL RESULT{err} (id={m.get('id')}): {short(m.get('content') or '')}"
            )
            continue
        if m.get("content"):
            print(f"[{i}] {role.upper()}: {m['content']}")
        for tc in m.get("tool_calls") or []:
            print(
                f"[{i}] {role.upper()} CALLS {tc['name']}({json.dumps(tc['arguments'])}) id={tc['id']}"
            )
        log = (m.get("raw_data") or {}).get("guard") if role == "assistant" else None
        if not log:
            continue
        for b in log.get("blocked_attempts", []):
            calls = ", ".join(
                f"{c['name']}({json.dumps(c['arguments'])})" for c in b["tool_calls"]
            )
            print(
                f"[{i}]   GUARD-WITHHELD ATTEMPT {b['attempt']}: text={short(b.get('content') or '', 300)!r} calls={calls}"
            )
        for d in log.get("decisions", []):
            tc = d["tool_call"]
            verdict = "ALLOW" if d["allow"] else "DENY"
            extra = (" (cached)" if d.get("cached") else "") + (
                f" ERROR {d['error']}" if d.get("error") else ""
            )
            print(
                f"[{i}]   GUARD {verdict} {tc['name']}({json.dumps(tc['arguments'])}){extra}"
            )
            if d.get("violated_conditions"):
                print(f"[{i}]     violated: {d['violated_conditions']}")
            if d.get("rationale"):
                print(f"[{i}]     rationale: {short(d['rationale'], 800)}")
            checks = (d.get("details") or {}).get("checks")
            if checks:
                print(f"[{i}]     checks: {short(checks, 1500)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("task")
    t.add_argument("task_id")
    t.add_argument("--domain", default="airline")
    s = sub.add_parser("sim")
    s.add_argument("results", type=Path)
    s.add_argument("task_id")
    s.add_argument("trial", type=int)
    a = p.parse_args()
    if a.cmd == "task":
        show_task(a.domain, a.task_id)
    else:
        show_sim(a.results, a.task_id, a.trial)


if __name__ == "__main__":
    main()
