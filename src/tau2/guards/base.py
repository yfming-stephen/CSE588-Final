"""Pre-execution guards for tool-calling agents.

A guard sits between the agent and the environment. Before a *guarded* tool
call (by default: every tool that mutates the DB) is executed, the guard
decides whether the call is allowed. Every guard (G1 LLM judge, G2 per-call
rules, G3 stateful provenance) plugs into the same ``GuardedAgent`` wrapper, so
the block-handling strategies are identical across guards.

Why the wrapper lives on the agent side
---------------------------------------
tau2 v1.0.1 rebuilds the final DB for scoring by *replaying every mutating
tool call found in the trajectory* (``Environment.set_state``). It does not
look at ``ToolMessage.error`` and has no notion of a rejected call (that only
exists upstream in tau2 >= 2.0.0). If a blocked call were left in the
trajectory, the evaluator would execute it for real. ``GuardedAgent``
therefore never returns a blocked call to the orchestrator: the blocked call
and the guard's reply live only in the inner agent's private history, and the
decision is logged in ``AssistantMessage.raw_data["guard"]``.
"""

from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from tau2.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    ToolCall,
    ToolMessage,
)
from tau2.environment.tool import Tool
from tau2.environment.toolkit import MUTATES_STATE_ATTR

STOP_TOKEN = "###STOP###"


class BlockMode(str, Enum):
    """What happens after the guard blocks a call."""

    TERMINATE = "terminate"
    """End the conversation right away (AGENT_STOP, so the run is still scored)."""

    REPLAN = "replan"
    """The agent gets a generic 'blocked' tool error, with no reason, and tries again."""

    FEEDBACK = "feedback"
    """The agent gets the unmet policy conditions as a tool error and tries again."""

    NOTIFY = "notify"
    """The user is told which conditions are unmet; the agent's turn ends."""


class GuardDecision(BaseModel):
    """The guard's verdict on one proposed tool call."""

    allow: bool
    violated_conditions: list[str] = Field(default_factory=list)
    rationale: str = ""
    guard: str = ""
    latency_seconds: Optional[float] = None
    cost: Optional[float] = None
    usage: Optional[dict] = None
    error: Optional[str] = Field(
        default=None,
        description="Set when the guard itself failed and the fallback decision was used.",
    )
    cached: bool = Field(
        default=False,
        description="Reused from an identical earlier check (same call, same transcript).",
    )
    details: Optional[dict] = Field(
        default=None,
        description="Guard-specific extras, e.g. the judge's per-condition checks.",
    )


@dataclass
class GuardContext:
    """Static information every guard may use."""

    domain_policy: str
    tools: dict[str, Tool] = field(default_factory=dict)


class Guard(ABC):
    """Interface shared by G1/G2/G3."""

    name: str = "guard"

    @abstractmethod
    def check(
        self,
        tool_call: ToolCall,
        transcript: list[Message],
        context: GuardContext,
    ) -> GuardDecision:
        """Decide whether ``tool_call`` may run now.

        Args:
            tool_call: The proposed call.
            transcript: The real conversation so far, as it appears in the
                trajectory: user messages, assistant messages and tool results.
                Calls blocked earlier are not included. If the agent issued
                other calls earlier in the same step, the transcript ends with
                an assistant message holding those pending calls (no results
                yet); they will run before ``tool_call`` if it is allowed.
            context: Policy and tool definitions.
        """

    def set_seed(self, seed: int) -> None:  # noqa: B027 - optional hook
        """Optional: make the guard deterministic for a given simulation seed."""

    def describe(self) -> dict:
        """The guard's resolved configuration, logged with every decision."""
        return {"name": self.name}


def is_mutating_tool(tool: Tool) -> bool:
    """True if the tool writes to the environment DB (tau2's ``mutates_state``)."""
    func = getattr(tool, "_func", None)
    return bool(getattr(func, MUTATES_STATE_ATTR, False))


def decision_key(tool_call: ToolCall, transcript: list[Message]) -> str:
    """Identify a guard question: the call (minus its id) plus the exact transcript.

    Two checks with the same key are the same question, so they get the same
    answer instead of a fresh sample from an LLM judge. This only covers an
    unchanged retry with nothing new in between. If the agent first runs other
    tools (even read-only ones), the transcript differs and the guard is asked
    again, so a stateless judge can still flip; that is a property of G1, not
    something this cache hides.
    """

    def sig(m: Any) -> Any:
        if isinstance(m, MultiToolMessage):
            return [sig(t) for t in m.tool_messages]
        if isinstance(m, ToolMessage):
            return ["tool", m.id, m.content, m.error]
        calls = [
            [tc.name, tc.arguments] for tc in (getattr(m, "tool_calls", None) or [])
        ]
        return [getattr(m, "role", None), getattr(m, "content", None), calls]

    payload = [tool_call.name, tool_call.arguments, [sig(m) for m in transcript]]
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


class GuardedAgentState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    inner_state: Any
    transcript: list[Message] = Field(default_factory=list)
    decision_cache: dict[str, GuardDecision] = Field(default_factory=dict)


class GuardedAgent(HalfDuplexAgent[GuardedAgentState]):
    """Wraps a half-duplex agent and checks guarded tool calls before they run.

    The inner agent must keep its history in ``state.messages`` (true for
    ``LLMAgent``), because the terminal block modes write the guard's reply
    into that history so the next turn stays a valid tool-call sequence.
    """

    REPLAN_TOOL_ERROR = "Error: this action was blocked and was not executed."
    SIBLING_TOOL_ERROR = (
        "Error: not executed, because another action in the same step was blocked."
    )
    GIVE_UP_MESSAGE = "I'm sorry, but I'm unable to complete that request."
    TERMINATE_MESSAGE = (
        "I'm sorry, but I can't complete this request, so I'm ending this conversation."
    )
    PENDING_NOTE = (
        "(Guard note: the agent issued these actions in the same step as the "
        "proposed action. If allowed, they run before it.)"
    )

    def __init__(
        self,
        inner: HalfDuplexAgent,
        guard: Guard,
        mode: BlockMode | str = BlockMode.REPLAN,
        guarded_tools: Optional[set[str]] = None,
        max_replans: int = 3,
        on_guard_error: str = "block",
    ):
        super().__init__(tools=inner.tools, domain_policy=inner.domain_policy)
        if on_guard_error not in ("block", "allow"):
            raise ValueError("on_guard_error must be 'block' or 'allow'")
        if not isinstance(max_replans, int) or max_replans < 0:
            raise ValueError("max_replans must be an integer >= 0")
        self.inner = inner
        self.guard = guard
        self.mode = BlockMode(mode)
        self.max_replans = max_replans
        self.on_guard_error = on_guard_error
        tool_map = {t.name: t for t in inner.tools}
        if guarded_tools is None:
            guarded_tools = {n for n, t in tool_map.items() if is_mutating_tool(t)}
        unknown = set(guarded_tools) - set(tool_map)
        if unknown:
            raise ValueError(
                f"Guarded tools not offered to the agent: {sorted(unknown)}"
            )
        self.guarded_tools = set(guarded_tools)
        self.context = GuardContext(domain_policy=inner.domain_policy, tools=tool_map)

    # ------------------------------------------------------------------ setup

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> GuardedAgentState:
        history = list(message_history or [])
        return GuardedAgentState(
            inner_state=self.inner.get_init_state(message_history=list(history)),
            transcript=history,
        )

    def set_seed(self, seed: int):
        if hasattr(self.inner, "set_seed"):
            self.inner.set_seed(seed)
        self.guard.set_seed(seed)

    def is_stop(self, message: AssistantMessage) -> bool:
        """Only the guard's own terminate message ends the run.

        G0's LLMAgent never stops on its own, so neither does the inner agent
        here: agent text containing the stop token does not end a G1 run.
        """
        guard_log = (message.raw_data or {}).get("guard") or {}
        return bool(guard_log.get("terminated"))

    def config(self) -> dict:
        """Resolved wrapper + guard settings, logged with every decision."""
        return {
            "mode": self.mode.value,
            "max_replans": self.max_replans,
            "on_guard_error": self.on_guard_error,
            "guarded_tools": sorted(self.guarded_tools),
            "guard": self.guard.describe(),
        }

    # --------------------------------------------------------------- the loop

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: GuardedAgentState
    ) -> tuple[AssistantMessage, GuardedAgentState]:
        if isinstance(message, MultiToolMessage):
            state.transcript.extend(message.tool_messages)
        else:
            state.transcript.append(message)

        incoming: ValidAgentInputMessage = message
        decisions_log: list[dict] = []
        blocked_attempts: list[dict] = []
        costs: list[Optional[float]] = []
        usages: list[Optional[dict]] = []
        final: Optional[AssistantMessage] = None
        terminated = False

        for attempt in range(self.max_replans + 1):
            candidate, state.inner_state = self.inner.generate_next_message(
                incoming, state.inner_state
            )
            costs.append(candidate.cost)
            usages.append(candidate.usage)

            guarded_calls = [
                tc
                for tc in (candidate.tool_calls or [])
                if tc.name in self.guarded_tools
            ]
            if not guarded_calls:
                final = candidate
                break

            # Calls in one step run in order, so each guarded call is judged
            # with the calls before it in the step marked as pending.
            verdicts = []
            for j, tc in enumerate(candidate.tool_calls):
                if tc.name in self.guarded_tools:
                    pending = candidate.tool_calls[:j]
                    verdicts.append((tc, self._check(tc, state, pending)))
            for tc, d in verdicts:
                decisions_log.append(
                    {
                        "attempt": attempt,
                        "tool_call": tc.model_dump(mode="json"),
                        **d.model_dump(mode="json"),
                    }
                )
            blocked = {tc.id: d for tc, d in verdicts if not d.allow}
            if not blocked:
                final = candidate
                break

            blocked_attempts.append(
                {
                    "attempt": attempt,
                    "content": candidate.content,
                    "tool_calls": [
                        tc.model_dump(mode="json") for tc in candidate.tool_calls
                    ],
                }
            )
            tool_errors = self._tool_errors(candidate, blocked)

            can_replan = self.mode in (BlockMode.REPLAN, BlockMode.FEEDBACK)
            if can_replan and attempt < self.max_replans:
                incoming = (
                    tool_errors[0]
                    if len(tool_errors) == 1
                    else MultiToolMessage(role="tool", tool_messages=tool_errors)
                )
                continue

            # Terminal outcome: notify, terminate, or replan budget used up.
            final = AssistantMessage(
                role="assistant",
                content=self._closing_text(list(blocked.values()), can_replan),
                tool_calls=None,
                cost=0.0,
                usage=None,
            )
            self._append_inner_history(state.inner_state, [*tool_errors, final])
            terminated = self.mode == BlockMode.TERMINATE
            break

        assert final is not None
        if decisions_log:
            final = self._annotate(
                final, decisions_log, blocked_attempts, costs, usages, terminated
            )
        state.transcript.append(final)
        return final, state

    # ---------------------------------------------------------------- helpers

    def _check(
        self,
        tool_call: ToolCall,
        state: GuardedAgentState,
        pending: Optional[list[ToolCall]] = None,
    ) -> GuardDecision:
        transcript = list(state.transcript)
        if pending:
            transcript.append(
                AssistantMessage(
                    role="assistant", content=self.PENDING_NOTE, tool_calls=pending
                )
            )
        key = decision_key(tool_call, transcript)
        cached = state.decision_cache.get(key)
        if cached is not None:
            return cached.model_copy(
                update={
                    "cached": True,
                    "cost": None,
                    "usage": None,
                    "latency_seconds": 0.0,
                }
            )
        decision = self._ask_guard(tool_call, transcript)
        if decision.error is None:  # a failed check may succeed if asked again
            state.decision_cache[key] = decision
        return decision

    def _ask_guard(
        self, tool_call: ToolCall, transcript: list[Message]
    ) -> GuardDecision:
        start = time.perf_counter()
        try:
            decision = self.guard.check(tool_call, list(transcript), self.context)
        except Exception as e:  # the guard must never crash the simulation
            logger.warning(f"Guard {self.guard.name} failed on {tool_call.name}: {e}")
            decision = GuardDecision(
                allow=self.on_guard_error == "allow",
                rationale="Guard error; fallback decision applied.",
                error=f"{type(e).__name__}: {e}",
                cost=getattr(e, "cost", None),
                usage=getattr(e, "usage", None),
            )
        if decision.latency_seconds is None:
            decision.latency_seconds = time.perf_counter() - start
        if not decision.guard:
            decision.guard = self.guard.name
        return decision

    def _tool_errors(
        self, candidate: AssistantMessage, blocked: dict[str, GuardDecision]
    ) -> list[ToolMessage]:
        """One ToolMessage per call in the blocked step, in call order."""
        errors = []
        for tc in candidate.tool_calls or []:
            if tc.id in blocked:
                content = self._blocked_tool_text(blocked[tc.id])
            else:
                content = self.SIBLING_TOOL_ERROR
            errors.append(
                ToolMessage(
                    id=tc.id,
                    role="tool",
                    content=content,
                    requestor="assistant",
                    error=True,
                )
            )
        return errors

    def _blocked_tool_text(self, decision: GuardDecision) -> str:
        if self.mode == BlockMode.REPLAN:
            return self.REPLAN_TOOL_ERROR
        conditions = _format_conditions(decision.violated_conditions)
        return (
            "Error: this action was blocked by the policy guard and was not executed."
            + (f" Unmet policy conditions:\n{conditions}" if conditions else "")
            + "\nDo not retry this action unless these conditions are satisfied."
        )

    def _closing_text(
        self, decisions: list[GuardDecision], replan_exhausted: bool
    ) -> str:
        if self.mode == BlockMode.TERMINATE:
            return f"{self.TERMINATE_MESSAGE} {STOP_TOKEN}"
        if self.mode == BlockMode.NOTIFY:
            reasons = [c for d in decisions for c in d.violated_conditions]
            conditions = _format_conditions(list(dict.fromkeys(reasons)))
            if conditions:
                return (
                    "I'm sorry, but I can't complete that request because these "
                    f"policy conditions are not met:\n{conditions}"
                )
            return "I'm sorry, but I can't complete that request because it does not comply with our policy."
        assert replan_exhausted
        return self.GIVE_UP_MESSAGE

    @staticmethod
    def _append_inner_history(inner_state: Any, messages: list[Message]) -> None:
        history = getattr(inner_state, "messages", None)
        if not isinstance(history, list):
            raise TypeError(
                "GuardedAgent needs an inner agent whose state keeps a 'messages' list "
                f"(e.g. LLMAgent); got {type(inner_state).__name__}."
            )
        history.extend(messages)

    def _annotate(
        self,
        final: AssistantMessage,
        decisions_log: list[dict],
        blocked_attempts: list[dict],
        costs: list[Optional[float]],
        usages: list[Optional[dict]],
        terminated: bool = False,
    ) -> AssistantMessage:
        """Attach the guard log and charge every agent attempt to the returned message."""
        raw = dict(final.raw_data or {})
        raw["guard"] = {
            "guard": self.guard.name,
            "mode": self.mode.value,
            "config": self.config(),
            "terminated": terminated,
            "decisions": decisions_log,
            "blocked_attempts": blocked_attempts,
            "guard_cost": _sum_or_none([d.get("cost") for d in decisions_log]),
            "guard_latency_seconds": sum(
                d.get("latency_seconds") or 0.0 for d in decisions_log
            ),
        }
        agent_cost = _sum_or_none(costs)
        if agent_cost is None:
            agent_cost = final.cost
        return final.model_copy(
            update={
                "raw_data": raw,
                "cost": agent_cost,
                "usage": _sum_usage(usages) or final.usage,
            }
        )


def _format_conditions(conditions: list[str]) -> str:
    return "\n".join(f"- {c.strip()}" for c in conditions if c and c.strip())


def _sum_or_none(values: list[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) if vals else None


def _sum_usage(usages: list[Optional[dict]]) -> Optional[dict]:
    present = [u for u in usages if u]
    if not present:
        return None
    # tau2's get_token_usage() reads both keys, so always emit both.
    total = {"prompt_tokens": 0, "completion_tokens": 0}
    for u in present:
        for key in total:
            if isinstance(u.get(key), (int, float)):
                total[key] += int(u[key])
    return total
