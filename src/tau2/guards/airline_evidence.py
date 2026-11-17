"""Rebuild G2's per-call inputs from actual tool exchanges, never user claims."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    ToolCall,
    ToolMessage,
)
from tau2.domains.airline.data_model import DirectFlight, Reservation, User

WRITE_TOOLS = frozenset(
    {
        "book_reservation",
        "cancel_reservation",
        "update_reservation_flights",
        "update_reservation_baggages",
        "update_reservation_passengers",
        "send_certificate",
    }
)
FLIGHT_STATUSES = frozenset(
    {"available", "on time", "delayed", "cancelled", "flying", "landed"}
)


class Change(BaseModel):
    """A completed write, retained locally to check compensation prerequisites."""

    call: ToolCall
    before: Reservation | None = None
    statuses: list[str | None] = Field(default_factory=list)
    index: int


class AirlineEvidence(BaseModel):
    """Ephemeral input snapshot, constructed anew for every guard check."""

    reservations: dict[str, Reservation] = Field(default_factory=dict)
    users: dict[str, User] = Field(default_factory=dict)
    statuses: dict[tuple[str, str], str] = Field(default_factory=dict)
    flights: dict[tuple[str, str], DirectFlight] = Field(default_factory=dict)
    searches: dict[str, list[tuple[str, str]]] = Field(default_factory=dict)
    changes: list[Change] = Field(default_factory=list)
    pending: list[ToolCall] = Field(default_factory=list)


def flight_key(flight: Any) -> tuple[str, str]:
    """Identify a dated flight from either a model or call argument."""
    if isinstance(flight, dict):
        return flight["flight_number"], flight["date"]
    return flight.flight_number, flight.date


def _decode(content: str | None) -> Any:
    try:
        return json.loads(content or "null")
    except (ValueError, TypeError):
        return content


def _search_results(evidence: AirlineEvidence, call: ToolCall, value: Any) -> None:
    query = call.name + json.dumps(call.arguments, sort_keys=True)
    for key in evidence.searches.pop(query, []):
        evidence.flights.pop(key, None)
        evidence.statuses.pop(key, None)
    evidence.searches[query] = []
    if not isinstance(value, list):
        return
    # Invalidate the queried route/date even if a later search returns no seats.
    args = call.arguments
    if call.name == "search_direct_flight":
        for key, flight in list(evidence.flights.items()):
            if (
                key[1] == args.get("date")
                and flight.origin == args.get("origin")
                and flight.destination == args.get("destination")
            ):
                evidence.flights.pop(key)
                evidence.statuses.pop(key, None)
        routes = [[item] for item in value]
    else:
        routes = value
    for route in routes:
        if not isinstance(route, list) or not route:
            continue
        try:
            flights = [DirectFlight.model_validate(item) for item in route]
        except (ValidationError, TypeError):
            continue
        if flights[0].origin != args.get("origin") or flights[
            -1
        ].destination != args.get("destination"):
            continue
        if call.name == "search_direct_flight":
            flights[0].date = args.get("date")
        if flights[0].date != args.get("date"):
            continue
        for flight in flights:
            if not flight.date:
                continue
            key = flight_key(flight)
            evidence.flights[key] = flight
            evidence.statuses[key] = flight.status
            evidence.searches[query].append(key)


def collect_evidence(transcript: list[Message]) -> AirlineEvidence:
    """Associate results with unique assistant calls and validate record identities.

    Successful writes replace the affected reservation snapshot and invalidate
    payment/seat snapshots. Completed transitions are used only within this call,
    e.g. proving a delayed booking was changed before issuing compensation.
    """
    evidence = AirlineEvidence()
    pending: dict[str, ToolCall] = {}
    seen: set[str] = set()
    for index, message in enumerate(transcript):
        if isinstance(message, AssistantMessage):
            for call in message.tool_calls or []:
                if call.requestor == "assistant" and call.id and call.id not in seen:
                    pending[call.id] = call
                    seen.add(call.id)
            continue
        results = (
            message.tool_messages
            if isinstance(message, MultiToolMessage)
            else [message]
            if isinstance(message, ToolMessage)
            else []
        )
        for result in results:
            call = pending.pop(result.id, None)
            if call is None or result.requestor != "assistant":
                continue
            args, value = call.arguments, _decode(result.content)
            if result.error:
                # Some backend functions can mutate before raising. Never retain
                # snapshots after a failed write as if nothing could have changed.
                if call.name in WRITE_TOOLS:
                    evidence.reservations.clear()
                    evidence.users.clear()
                    evidence.flights.clear()
                    evidence.statuses.clear()
                continue
            if call.name in WRITE_TOOLS:
                rid = args.get("reservation_id")
                before = (
                    evidence.reservations.get(rid) if isinstance(rid, str) else None
                )
                evidence.changes.append(
                    Change(
                        call=call,
                        before=before,
                        statuses=[
                            evidence.statuses.get(flight_key(f)) for f in before.flights
                        ]
                        if before
                        else [],
                        index=index,
                    )
                )
                uid = args.get("user_id") or (before.user_id if before else None)
                if uid:
                    evidence.users.pop(uid, None)
                if call.name != "send_certificate":
                    if isinstance(rid, str):
                        evidence.reservations.pop(rid, None)
                    try:
                        record = Reservation.model_validate(value)
                        if (
                            call.name == "book_reservation" and record.user_id == uid
                        ) or record.reservation_id == rid:
                            evidence.reservations[record.reservation_id] = record
                    except ValidationError:
                        pass
                    # Seat availability may have changed; statuses remain valid
                    # because these writes do not change airline flight status.
                    evidence.flights.clear()
                continue
            if call.name == "get_reservation_details":
                rid = args.get("reservation_id")
                if not isinstance(rid, str):
                    continue
                evidence.reservations.pop(rid, None)
                try:
                    record = Reservation.model_validate(value)
                    if record.reservation_id == rid:
                        evidence.reservations[rid] = record
                except ValidationError:
                    pass
            elif call.name == "get_user_details":
                uid = args.get("user_id")
                if not isinstance(uid, str):
                    continue
                evidence.users.pop(uid, None)
                try:
                    user = User.model_validate(value)
                    if user.user_id == uid:
                        evidence.users[uid] = user
                except ValidationError:
                    pass
            elif call.name == "get_flight_status":
                key = (args.get("flight_number"), args.get("date"))
                if not all(isinstance(part, str) for part in key):
                    continue
                evidence.statuses.pop(key, None)
                if isinstance(value, str) and value in FLIGHT_STATUSES:
                    evidence.statuses[key] = value
                if value != "available":
                    evidence.flights.pop(key, None)
            elif call.name in {"search_direct_flight", "search_onestop_flight"}:
                _search_results(evidence, call, value)
    evidence.pending = list(pending.values())
    return evidence
