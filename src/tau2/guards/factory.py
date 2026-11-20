"""CLI factories for guarded agents.

Each block mode gets its own agent name (``g1_replan``, ``g1_feedback``,
``g1_notify``, ``g1_terminate``). Guard options travel inside
``--agent-llm-args`` under a ``guard_`` prefix, so they are stripped before
the agent's LLM call and still recorded verbatim in results.json:

    --agent-llm-args '{"temperature": 0.0,
                       "guard_llm": "gpt-4.1-2025-04-14",
                       "guard_llm_args": {"temperature": 0.0},
                       "guard_evidence_rule": true,
                       "guard_max_replans": 3,
                       "guard_on_error": "block",
                       "guard_tools": ["cancel_reservation", "send_certificate"]}'

Note that ``--agent-llm-args`` replaces tau2's default ``{"temperature": 0.0}``
for the agent, so keep ``temperature`` in the dict.
"""

from __future__ import annotations

from functools import partial
from typing import Callable

from tau2.agent.llm_agent import LLMAgent
from tau2.guards.base import BlockMode, GuardedAgent
from tau2.guards.llm_judge import (
    DEFAULT_GUARD_LLM,
    DEFAULT_PROMPT_VERSION,
    LLMJudgeGuard,
)

GUARD_ARG_PREFIX = "guard_"


def split_guard_args(llm_args: dict | None) -> tuple[dict, dict]:
    """Separate ``guard_*`` options from the agent's own LLM arguments."""
    agent_args, guard_args = {}, {}
    for key, value in (llm_args or {}).items():
        if key.startswith(GUARD_ARG_PREFIX):
            guard_args[key[len(GUARD_ARG_PREFIX) :]] = value
        else:
            agent_args[key] = value
    return agent_args, guard_args


def create_g1_agent(
    tools, domain_policy, *, mode: BlockMode | str, **kwargs
) -> GuardedAgent:
    """G0's LLMAgent, unchanged, behind an LLM-judge guard."""
    agent_args, opts = split_guard_args(kwargs.get("llm_args"))
    inner = LLMAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=agent_args,
    )
    evidence_rule = opts.pop("evidence_rule", True)
    if not isinstance(evidence_rule, bool):
        # A quoted "false" would otherwise silently select the wrong arm.
        raise ValueError("guard_evidence_rule must be a JSON true or false")
    guard = LLMJudgeGuard(
        llm=opts.pop("llm", DEFAULT_GUARD_LLM),
        llm_args=opts.pop("llm_args", None),
        evidence_rule=evidence_rule,
        prompt_version=opts.pop("prompt_version", DEFAULT_PROMPT_VERSION),
    )
    guarded_tools = opts.pop("tools", None)
    agent = GuardedAgent(
        inner=inner,
        guard=guard,
        mode=mode,
        guarded_tools=set(guarded_tools) if guarded_tools is not None else None,
        max_replans=int(opts.pop("max_replans", 3)),
        on_guard_error=opts.pop("on_error", "block"),
    )
    if opts:
        raise ValueError(
            f"Unknown guard options: {sorted(GUARD_ARG_PREFIX + k for k in opts)}"
        )
    return agent


G1_AGENT_FACTORIES: dict[str, Callable] = {
    f"g1_{mode.value}": partial(create_g1_agent, mode=mode) for mode in BlockMode
}
