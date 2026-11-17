"""G2: deterministic, per-call airline business-policy guard."""

from __future__ import annotations

import re
from datetime import datetime

from pydantic import ValidationError

from tau2.data_model.message import Message, ToolCall
from tau2.guards.airline_evidence import collect_evidence
from tau2.guards.airline_rules import ARGUMENT_MODELS, AirlineChecks
from tau2.guards.base import Guard, GuardContext, GuardDecision


class AirlineRuleGuard(Guard):
    """Check all six airline database-writing tools with explicit predicates.

    Inputs are rebuilt from the transcript for each call. No LLM, solver, or
    cross-call fact/provenance store is used. Dialogue parsing is deterministic
    and intentionally fails closed when it cannot establish user approval.
    """

    name = "g2_airline_rules_v8"

    def describe(self) -> dict:
        """Expose coverage and parsing version in the experiment logs."""
        return {
            "name": self.name,
            "supported_tools": sorted(ARGUMENT_MODELS),
            "dialogue_parser": "deterministic_english_v8",
            "backend_evidence": "matched_tool_results",
        }

    def check(
        self,
        tool_call: ToolCall,
        transcript: list[Message],
        context: GuardContext,
    ) -> GuardDecision:
        """Validate inputs, evaluate applicable predicates, and log every check."""
        match = re.search(
            r"The current time is (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) EST\.",
            context.domain_policy,
        )
        try:
            now = datetime.fromisoformat(match[1]) if match else None
        except ValueError:
            now = None
        evidence = collect_evidence(transcript)
        checks = AirlineChecks(tool_call, transcript, evidence, now)
        checks.add(
            "assistant tool request",
            tool_call.requestor == "assistant",
            tool_call.requestor,
            "schema",
        )
        checks.add(
            "airline policy",
            context.domain_policy.startswith("# Airline Agent Policy"),
            "policy heading",
        )
        checks.add(
            "policy current time available",
            now is not None,
            match[1] if match else None,
            "evidence",
        )
        model = ARGUMENT_MODELS.get(tool_call.name)
        checks.add(
            "supported airline write tool", model is not None, tool_call.name, "schema"
        )
        checks.add(
            "no earlier unresolved call in this step",
            not evidence.pending,
            [c.name for c in evidence.pending],
            "evidence",
        )
        if model is not None:
            try:
                arguments = model.model_validate(tool_call.arguments, strict=True)
            except ValidationError as exc:
                checks.add(
                    "valid tool arguments",
                    False,
                    exc.errors(include_input=False, include_url=False),
                    "schema",
                )
            else:
                checks.args = arguments.model_dump()
                checks.add("valid tool arguments", True, checks.args, "schema")
                if now is not None and context.domain_policy.startswith(
                    "# Airline Agent Policy"
                ):
                    dispatch = {
                        "cancel_reservation": checks.cancel,
                        "book_reservation": checks.book,
                        "update_reservation_flights": checks.change_flights,
                        "update_reservation_baggages": checks.change_baggage,
                        "update_reservation_passengers": checks.change_passengers,
                        "send_certificate": checks.compensate,
                    }
                    dispatch[tool_call.name]()
        unmet = [
            condition.condition for condition in checks.conditions if not condition.met
        ]
        return GuardDecision(
            allow=not unmet,
            violated_conditions=unmet,
            guard=self.name,
            rationale="All conditions met" if not unmet else "; ".join(unmet),
            details={
                "checks": [
                    condition.model_dump(mode="json") for condition in checks.conditions
                ]
            },
            cost=0.0,
        )
