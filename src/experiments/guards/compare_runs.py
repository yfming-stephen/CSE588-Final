"""Compare a plain-agent run (G0) with a guarded run (G1).

    python src/experiments/guards/compare_runs.py \
        data/simulations/g0_test_k4 data/simulations/g1_replan_test_k4 --matched

Reports pass^k, DB match, writes on tasks whose gold has no write, guard
checks and blocks, cost, a paired comparison over tasks (exact bootstrap CI
and an exact sign-flip permutation test), McNemar over (task, trial) pairs,
and a per-task table. Infrastructure-error sims are dropped. With
``--matched`` only (task, trial) pairs that both runs finished are kept, so
both arms cover the same tasks and trial counts.

Same-seed sims of the two arms are not coupled (the OpenAI API is not
deterministic at temperature 0), so pairing trial i with trial i is arbitrary;
the task-level test is the one to report.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from tau2.guards.base import is_mutating_tool
from tau2.guards.summarize import load_simulations, summarize_simulation
from tau2.runner.build import build_environment

# Test tasks whose guard decisions in the first k=1 G1 run prompted the v2/v3
# judge prompts.
TUNING_TASKS = {"22", "24", "29", "37", "48"}


def load(path: Path) -> tuple[dict, list[dict], int]:
    meta = path / "results.json" if path.is_dir() else path
    data = json.loads(meta.read_text())
    sims = load_simulations(meta)
    ok = [s for s in sims if s.get("termination_reason") != "infrastructure_error"]
    return data, ok, len(sims) - len(ok)


def key(sim: dict) -> tuple[str, int]:
    return str(sim["task_id"]), sim["trial"]


def success(sim: dict) -> int:
    reward = (sim.get("reward_info") or {}).get("reward") or 0.0
    return int(reward >= 1 - 1e-6)


def pass_hat_k(counts: dict[str, tuple[int, int]], k: int) -> tuple[float, int]:
    """tau2's pass^k, averaged over tasks with at least k trials."""
    vals = [math.comb(c, k) / math.comb(n, k) for c, n in counts.values() if n >= k]
    return (sum(vals) / len(vals) if vals else float("nan")), len(vals)


def executed_writes(sim: dict, mutating: set[str]) -> list[tuple[str, dict]]:
    """Calls to DB-writing tools that reached the trajectory and succeeded."""
    ok = {
        m.get("id"): not m.get("error") for m in sim["messages"] if m["role"] == "tool"
    }
    return [
        (tc["name"], tc["arguments"])
        for m in sim["messages"]
        if m["role"] == "assistant"
        for tc in m.get("tool_calls") or []
        if tc["name"] in mutating and ok.get(tc["id"])
    ]


def guard_decisions(sim: dict) -> list[dict]:
    return [
        d
        for m in sim["messages"]
        if m["role"] == "assistant"
        for d in ((m.get("raw_data") or {}).get("guard") or {}).get("decisions", [])
    ]


def arm_stats(sims: list[dict], mutating: set[str], gold: dict) -> dict:
    by_task = defaultdict(list)
    for s in sims:
        by_task[str(s["task_id"])].append(success(s))
    counts = {t: (sum(v), len(v)) for t, v in by_task.items()}
    max_n = max(n for _, n in counts.values())
    full = {t: c for t, c in counts.items() if c[1] == max_n}
    n = len(sims)
    breakdown = [
        (s.get("reward_info") or {}).get("reward_breakdown") or {} for s in sims
    ]
    guard = [summarize_simulation(s) for s in sims]
    no_write = [s for s in sims if not gold.get(str(s["task_id"]))]
    gold_blocks = sum(
        1
        for s in sims
        for d in guard_decisions(s)
        if not d["allow"]
        and (d["tool_call"]["name"], d["tool_call"]["arguments"])
        in gold.get(str(s["task_id"]), [])
    )
    checks = sum(g["guard_checks"] for g in guard)
    blocks = sum(g["blocked_calls"] for g in guard)
    cached = sum(g["cached_checks"] for g in guard)
    cached_blocks = sum(
        1
        for s in sims
        for d in guard_decisions(s)
        if d.get("cached") and not d["allow"]
    )
    return {
        "sims": n,
        "success_rate_over_sims": sum(success(s) for s in sims) / n,
        "pass^k": {k: pass_hat_k(counts, k) for k in range(1, 5)},
        # tasks with fewer trials drop out of pass^k for larger k, so also
        # report every k on the tasks that have the most trials
        "pass^k_full_tasks": {k: pass_hat_k(full, k) for k in range(1, max_n + 1)},
        "db_match": sum(b.get("DB", 0) >= 1 for b in breakdown) / n,
        "no_write_task_sims": len(no_write),
        "no_write_task_sims_with_write": sum(
            bool(executed_writes(s, mutating)) for s in no_write
        ),
        "sims_with_non_gold_write": sum(
            any(
                w not in gold.get(str(s["task_id"]), [])
                for w in executed_writes(s, mutating)
            )
            for s in sims
        ),
        "agent_cost_per_sim": sum(s.get("agent_cost") or 0 for s in sims) / n,
        "user_cost_per_sim": sum(s.get("user_cost") or 0 for s in sims) / n,
        "judge_cost_per_sim": sum(g["guard_cost"] for g in guard) / n,
        # cached checks are logged with 0 s latency, so average over real calls
        "judge_latency_per_call": (
            sum(g["guard_latency_seconds"] for g in guard) / (checks - cached)
            if checks > cached
            else None
        ),
        "guard_checks": checks,
        "guard_cached_checks": cached,
        "guard_blocks": blocks,
        "guard_cached_blocks": cached_blocks,
        "sims_with_block": sum(g["blocked_calls"] > 0 for g in guard),
        "guard_errors": sum(g["guard_errors"] for g in guard),
        "blocks_of_exact_gold_calls": gold_blocks,
        "termination": dict(Counter(s["termination_reason"] for s in sims)),
        "counts": counts,
    }


def sign_flip_p(diffs: list[float]) -> float:
    """Exact two-sided paired permutation test on per-task differences."""
    obs = abs(sum(diffs))
    nz = [d for d in diffs if d != 0]
    if not nz:
        return 1.0
    hits = sum(
        abs(sum(s * d for s, d in zip(signs, nz))) >= obs - 1e-12
        for signs in itertools.product((1, -1), repeat=len(nz))
    )
    return hits / 2 ** len(nz)


def bootstrap_ci(diffs: list[Fraction]) -> tuple[float, float]:
    """Exact 95% percentile-bootstrap CI of the mean.

    The per-task differences are fractions, so the distribution of the
    resampled sum can be enumerated exactly instead of sampled. Monte Carlo
    is unreliable here because the sum takes few values and a quantile can sit
    right at a step.
    """
    m = len(diffs)
    dist = {Fraction(0): Fraction(1)}
    for _ in range(m):
        step = defaultdict(Fraction)
        for total, p in dist.items():
            for d in diffs:
                step[total + d] += p / m
        dist = step
    lo = hi = None
    cum = Fraction(0)
    for total, p in sorted(dist.items()):
        cum += p
        if lo is None and cum >= Fraction(1, 40):
            lo = total / m
        if hi is None and cum >= Fraction(39, 40):
            hi = total / m
    return float(lo), float(hi)


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def paired_over_tasks(s0: dict, s1: dict, tasks: list[str]) -> dict:
    rate = lambda s, t: Fraction(*s["counts"][t])  # noqa: E731
    diffs = [rate(s1, t) - rate(s0, t) for t in tasks]
    return {
        "tasks": len(tasks),
        "g0": float(sum(rate(s0, t) for t in tasks) / len(tasks)),
        "g1": float(sum(rate(s1, t) for t in tasks) / len(tasks)),
        "diff": float(sum(diffs) / len(diffs)),
        "ci95": bootstrap_ci(diffs),
        "sign_flip_p": sign_flip_p(diffs),
        # with n differing tasks the exact test cannot go below 2 / 2**n
        "sign_flip_min_p": min(1.0, 2 / 2 ** sum(d != 0 for d in diffs)),
        "tasks_g1_better": sum(d > 0 for d in diffs),
        "tasks_g1_worse": sum(d < 0 for d in diffs),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("g0", type=Path)
    p.add_argument("g1", type=Path)
    p.add_argument("--domain", default="airline")
    p.add_argument(
        "--matched",
        action="store_true",
        help="keep only (task, trial) pairs both runs finished",
    )
    p.add_argument("--json", type=Path, help="write every number to this file")
    a = p.parse_args()

    tools = build_environment(a.domain).get_tools()
    mutating = {t.name for t in tools if is_mutating_tool(t)}
    (d0, sims0, infra0), (d1, sims1, infra1) = load(a.g0), load(a.g1)
    gold = {
        str(t["id"]): [
            (x["name"], x["arguments"])
            for x in ((t.get("evaluation_criteria") or {}).get("actions") or [])
            if x["name"] in mutating
        ]
        for t in d0["tasks"]
    }
    for name, d, sims, infra in (("G0", d0, sims0, infra0), ("G1", d1, sims1, infra1)):
        info = d["info"]
        print(
            f"{name}: {info['agent_info']['implementation']} @ {str(info.get('git_commit'))[:7]}, "
            f"{len(sims)} scored sims, {infra} infrastructure errors dropped"
        )
    if a.matched:
        both = {key(s) for s in sims0} & {key(s) for s in sims1}
        print(f"Matched: keeping {len(both)} (task, trial) pairs")
        sims0 = [s for s in sims0 if key(s) in both]
        sims1 = [s for s in sims1 if key(s) in both]

    s0, s1 = arm_stats(sims0, mutating, gold), arm_stats(sims1, mutating, gold)
    print(f"\n{'metric':36s} {'G0':>10s} {'G1':>10s}")
    line = lambda label, v0, v1: print(f"{label:36s} {v0:>10s} {v1:>10s}")  # noqa: E731
    line("sims", str(s0["sims"]), str(s1["sims"]))
    line(
        "success rate over sims",
        f"{s0['success_rate_over_sims']:.3f}",
        f"{s1['success_rate_over_sims']:.3f}",
    )
    for k in range(1, 5):
        (v0, n0), (v1, _) = s0["pass^k"][k], s1["pass^k"][k]
        if n0:
            line(f"pass^{k} ({n0} tasks)", f"{v0:.3f}", f"{v1:.3f}")
    full0, full1 = s0["pass^k_full_tasks"], s1["pass^k_full_tasks"]
    if full0[1][1] != s0["pass^k"][1][1]:
        for k in full0:
            (v0, n0), (v1, _) = full0[k], full1.get(k, (float("nan"), 0))
            line(f"pass^{k} (same {n0} tasks)", f"{v0:.3f}", f"{v1:.3f}")
    line("DB match", f"{s0['db_match']:.3f}", f"{s1['db_match']:.3f}")
    line(
        "no-write-task sims with a write",
        f"{s0['no_write_task_sims_with_write']}/{s0['no_write_task_sims']}",
        f"{s1['no_write_task_sims_with_write']}/{s1['no_write_task_sims']}",
    )
    line(
        "sims with a non-gold write",
        str(s0["sims_with_non_gold_write"]),
        str(s1["sims_with_non_gold_write"]),
    )
    line(
        "guard checks / blocks",
        f"{s0['guard_checks']}/{s0['guard_blocks']}",
        f"{s1['guard_checks']}/{s1['guard_blocks']}",
    )
    line(
        "  of which cached (checks / blocks)",
        f"{s0['guard_cached_checks']}/{s0['guard_cached_blocks']}",
        f"{s1['guard_cached_checks']}/{s1['guard_cached_blocks']}",
    )
    line("sims with a block", str(s0["sims_with_block"]), str(s1["sims_with_block"]))
    line(
        "blocks of exact gold calls",
        str(s0["blocks_of_exact_gold_calls"]),
        str(s1["blocks_of_exact_gold_calls"]),
    )
    line("guard errors", str(s0["guard_errors"]), str(s1["guard_errors"]))
    for label, field in (
        ("agent $/sim", "agent_cost_per_sim"),
        ("user $/sim", "user_cost_per_sim"),
        ("judge $/sim", "judge_cost_per_sim"),
    ):
        line(label, f"{s0[field]:.4f}", f"{s1[field]:.4f}")
    if s1["judge_latency_per_call"] is not None:
        line(
            "judge seconds/call (uncached)", "-", f"{s1['judge_latency_per_call']:.1f}"
        )
    print(f"termination: G0 {s0['termination']} | G1 {s1['termination']}")

    tasks = sorted(set(s0["counts"]) & set(s1["counts"]), key=int)
    paired = {
        "all_tasks": paired_over_tasks(s0, s1, tasks),
        "without_tuning_tasks": paired_over_tasks(
            s0, s1, [t for t in tasks if t not in TUNING_TASKS]
        ),
    }
    o0 = {key(s): success(s) for s in sims0}
    o1 = {key(s): s for s in sims1}
    common = sorted(set(o0) & set(o1), key=lambda k: (int(k[0]), k[1]))
    diff_pairs = [
        {
            "task": t,
            "trial": tr,
            "g0": o0[(t, tr)],
            "g1": success(o1[(t, tr)]),
            "g1_blocks": summarize_simulation(o1[(t, tr)])["blocked_calls"],
        }
        for t, tr in common
        if o0[(t, tr)] != success(o1[(t, tr)])
    ]
    b = sum(d["g0"] for d in diff_pairs)
    c = sum(d["g1"] for d in diff_pairs)
    paired["mcnemar_by_task_trial"] = {
        "pairs": len(common),
        "g0_only": b,
        "g1_only": c,
        "p": mcnemar_exact(b, c),
    }
    paired["differing_pairs"] = diff_pairs

    print("\nPaired over tasks (per-task success rate):")
    for name in ("all_tasks", "without_tuning_tasks"):
        r = paired[name]
        print(
            f"  {name:22s} n={r['tasks']:2d}  G0 {r['g0']:.3f}  G1 {r['g1']:.3f}  diff {r['diff']:+.3f}  "
            f"95% CI [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]  sign-flip p={r['sign_flip_p']:.3f} (min {r['sign_flip_min_p']:.3f})  "
            f"G1 better/worse on {r['tasks_g1_better']}/{r['tasks_g1_worse']} tasks"
        )
    m = paired["mcnemar_by_task_trial"]
    print(
        f"  McNemar over {m['pairs']} (task, trial) pairs: G0-only {m['g0_only']}, G1-only {m['g1_only']}, "
        f"p={m['p']:.3f} (trials of one task are not independent; prefer the task-level test)"
    )
    print(
        "\nPairs with different outcomes (trial i of G0 vs trial i of G1; the API is not"
        "\ndeterministic, so the pairing is arbitrary; a G1 sim with no block is a fresh"
        "\ndraw of the G0 process):"
    )
    for d in diff_pairs:
        print(
            f"  task {d['task']:>2} trial {d['trial']}: G0 {d['g0']} G1 {d['g1']}  G1 blocks={d['g1_blocks']}"
        )

    print(
        f"\n{'task':>5s} {'gold writes':>11s} {'G0':>5s} {'G1':>5s} {'G1 blocks':>9s}"
    )
    blocks_by_task = Counter()
    for s in sims1:
        blocks_by_task[str(s["task_id"])] += summarize_simulation(s)["blocked_calls"]
    for t in tasks:
        (c0, n0), (c1, n1) = s0["counts"][t], s1["counts"][t]
        mark = " *" if t in TUNING_TASKS else ""
        print(
            f"{t:>5s} {len(gold.get(t, [])):>11d} {c0}/{n0:<3d} {c1}/{n1:<3d} {blocks_by_task[t]:>9d}{mark}"
        )
    print("  * = task that informed the v2/v3 judge prompts")

    if a.json:
        a.json.write_text(
            json.dumps({"G0": s0, "G1": s1, "paired": paired}, indent=2, default=str)
        )


if __name__ == "__main__":
    main()
