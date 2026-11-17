"""Deterministic English/JSON dialogue parsing used by the G2 rule baseline.

This is deliberately separate from backend facts. It parses user choices and
confirmations, not record eligibility. Ambiguous wording fails closed.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from tau2.data_model.message import AssistantMessage, Message, ToolCall, UserMessage
from tau2.domains.airline.data_model import Reservation, User


def normalized(text: str, reference_year: int | None = None) -> str:
    """Normalize dates and common cabin spellings without changing identities."""
    text = (
        text.lower()
        .replace("’", "'")
        .replace("‘", "'")
        .replace("basic economy", "basic_economy")
    )
    text = text.replace("business class", "business").replace(
        "economy class", "economy"
    )
    for match in list(
        re.finditer(r"\b([a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b", text)
    ):
        for fmt in ("%B %d %Y", "%b %d %Y"):
            try:
                date = datetime.strptime(" ".join(match.groups()), fmt).strftime(
                    "%Y-%m-%d"
                )
                text = text.replace(match[0], date)
                break
            except ValueError:
                continue
    if reference_year is not None:

        def dated(match):
            for fmt in ("%B %d %Y", "%b %d %Y"):
                try:
                    return datetime.strptime(
                        f"{match[1]} {match[2]} {reference_year}", fmt
                    ).strftime("%Y-%m-%d")
                except ValueError:
                    continue
            return match[0]

        # Missing years refer to the policy clock's year, never the proposed
        # call's year. This cannot silently approve a different-year booking.
        text = re.sub(r"\b([a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?\b", dated, text)
    # Currency formatting: "$5,700.00" and "$5700" denote the same amount.
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)
    return re.sub(r"(\$\s*\d+)\.00(?!\d)", r"\1", text)


def contains(text: str, value: Any) -> bool:
    """Match a complete value, never a substring of another ID or number."""
    return bool(
        re.search(
            rf"(?<!\w){re.escape(normalized(str(value)))}(?!\w)", normalized(text)
        )
    )


def supplied_identity(transcript: list[Message], user_id: str) -> bool:
    """Require a user-supplied ID; this benchmark has no authentication protocol."""
    supplied = False
    for message in transcript:
        if not isinstance(message, UserMessage) or not message.content:
            continue
        text = message.content
        explicit = re.search(
            r"\b(?:my\s+)?user\s*(?:id|identifier)\s*(?:is|:|=)?\s*[`\"']?([\w-]+)",
            text,
            re.I,
        )
        if explicit:
            supplied = explicit[1] == user_id
        elif contains(text, user_id):
            supplied = True
    return supplied


def cancellation_reason(transcript: list[Message], reservation_id: str) -> str | None:
    """Recognize a reason in the current reservation thread, including corrections."""
    reason = None
    active = True
    for message in transcript:
        if (
            not isinstance(message, (UserMessage, AssistantMessage))
            or not message.content
        ):
            continue
        text = message.content.lower()
        # An explicit switch to another reservation ends this reason's scope.
        targets = re.findall(
            r"\b(?:reservation|booking)\s+(?:id\s*(?:is|:)?\s*)?([a-z0-9]{6})\b", text
        )
        targets = [
            target
            for target in targets
            if any(c.isdigit() for c in target) or target.upper() == reservation_id
        ]
        if targets:
            active = reservation_id.lower() in targets
        if not active or not isinstance(message, UserMessage):
            continue
        if re.search(
            r"\b(if|hypothetically|pretend|suppose)\b|<[^>]*>|\b(system|developer)\s*:",
            text,
        ):
            continue
        covered = re.search(r"\b(health|medical|sick|illness|weather)\b", text)
        if covered:
            # Negation cannot establish a covered reason; a correction revokes it.
            reason = (
                "other"
                if re.search(r"\b(not|no|never|isn't|wasn't|don't)\b", text)
                else "covered"
            )
        elif re.search(
            r"\b(change of plans?|changed my plans|other reasons?|airline cancel(?:led|ed)|plans? (?:have )?changed)\b",
            text,
        ):
            reason = "other"
        elif re.search(
            r"\b(?:cancel\w*\b.*\b(?:because|due to)|reason\s*(?:is|:))\s*\S", text
        ):
            reason = "other"
    return reason


def user_requested_compensation(transcript: list[Message], reservation_id: str) -> bool:
    """Find an explicit request; agent offers and quoted instructions do not count."""
    requested = False
    active = True
    for message in transcript:
        if (
            not isinstance(message, (UserMessage, AssistantMessage))
            or not message.content
        ):
            continue
        text = message.content.lower()
        targets = re.findall(
            r"\b(?:reservation|booking)\s+(?:id\s*(?:is|:)?\s*)?([a-z0-9]{6})\b", text
        )
        targets = [
            target
            for target in targets
            if any(c.isdigit() for c in target) or target.upper() == reservation_id
        ]
        if targets:
            active = reservation_id.lower() in targets
            if not active:
                requested = False
        if not active or not isinstance(message, UserMessage):
            continue
        if re.search(
            r"\b(compensation|compensat\w*|voucher|certificate|travel credit)\b", text
        ):
            if re.search(r"\b(no|not|don't|do not|never)\b", text):
                requested = False
            elif re.search(
                r"\b(want|like|request|give|send|offer|provide|receive|can|could|may|please|get|ask)\b",
                text,
            ) and not re.search(r"\b(system|developer)\s*:|<[^>]*>", text):
                requested = True
    return requested


def compensation_topics(transcript: list[Message], reservation_id: str) -> set[str]:
    """Identify the user's complained-of disruption, scoped to a booking."""
    topics: set[str] = set()
    active = True
    for message in transcript:
        if (
            not isinstance(message, (UserMessage, AssistantMessage))
            or not message.content
        ):
            continue
        text = message.content.lower()
        targets = re.findall(
            r"\b(?:reservation|booking)\s+(?:id\s*(?:is|:)?\s*)?([a-z0-9]{6})\b", text
        )
        targets = [
            target
            for target in targets
            if any(c.isdigit() for c in target) or target.upper() == reservation_id
        ]
        if targets:
            active = reservation_id.lower() in targets
        if active and isinstance(message, UserMessage):
            if re.search(
                r"\b(?:flight\w*.*cancel(?:led|ed)|cancel(?:led|ed).*flight)\b", text
            ):
                topics.add("cancelled")
            if re.search(r"\bdelay\w*\b", text):
                topics.add("delayed")
    return topics


CONFIRMATION_CONDITION = "explicit confirmation of this action and its parameters"

_QUALIFIERS = re.compile(
    r"\b(no|not|don't|but|except|unless|if|instead|different|only|wait)\b|([<>{}])"
)
_OPENING = re.compile(
    r"^(?:yes|i (?:confirm|approve|agree)|confirmed|confirm|go ahead|please proceed|proceed|sounds good|that works|correct)\b"
)
# "You do not need to pay" / "will not be charged" describe a cost, not a refusal.
_REFUSAL = re.compile(
    r"\b(cannot|can't|unable|not allowed|"
    r"(?:will not|won't|do not)(?!\s+(?:need|have to|be charged|incur|require|cost)\b))\b"
)
_AMOUNT = re.compile(
    r"\$\s*(\d+(?:\.\d{1,2})?)(?!\d)|\b(\d+(?:\.\d{1,2})?)\s*(?:dollars?|usd)\b"
)
# Generic descriptions of a saved payment type. They identify a method only
# when the profile holds exactly one method of that type (see _payment_in_text).
_SOURCES = {
    "credit_card": r"\b(?:credit card|visa|mastercard|amex|card ending)\b",
    "gift_card": r"\bgift ?cards?\b",
    "certificate": r"\b(?:travel )?certificates?\b",
}
_ORIGINAL = re.compile(
    r"\boriginal (?:payment(?: method)?|form of payment|method of payment)\b"
)
_ACTIONS = {
    "cancel_reservation": (r"\bcancel(?:lation)?\b", "cancellation"),
    "book_reservation": (r"\b(book\w*|reserv\w*)\b", "booking"),
    "update_reservation_flights": (
        r"\b(chang\w*|updat\w*|upgrad\w*|downgrad\w*|switch\w*|modif\w*)\b",
        "flight/cabin change",
    ),
    "update_reservation_baggages": (r"\b(bag\w*|luggage)\b", "baggage update"),
    "update_reservation_passengers": (
        r"\b(passenger\w*|name\w*|birth|dob)\b",
        "passenger update",
    ),
}
_BASE_VOCABULARY = frozenset(
    """yes i confirm confirmed approve approved agree agreed please go ahead and
    proceed with adding add added updating update updated change changed changes
    changing make made complete completing book booking cancel cancellation
    this that these those exact action it the a an to from for of as described
    under my name passenger passengers flight flights reservation booking cabin
    class bag bags baggage baggages checked luggage dob date birth all other
    payment payments method methods
    details remain same unchanged would like want just s d ll m is correct
    sounds good works thank you thanks your help both in on at now
    refund refunded refunds receive received credit gift card cards certificate
    original full amount total price cost charge charged fee fees pay paid
    difference back via using use visa mastercard ending last four digits
    specified listed above mentioned stated get done once let me know soon
    possible right away immediately process processed today ok okay great
    perfect one""".split()
)


class ConfirmationCheck(BaseModel):
    """Outcome of binding the latest user approval to one proposed call.

    Each list entry names a concrete value or wording problem so a blocked
    agent can repair the proposal instead of guessing. Values come from the
    call and from backend tool results already shown in the transcript.
    """

    confirmed: bool
    reason: str | None = None
    proposal_missing: list[str] = Field(default_factory=list)
    proposal_conflicts: list[str] = Field(default_factory=list)
    reply_problems: list[str] = Field(default_factory=list)

    def feedback(self) -> str:
        """Render the unmet condition with actionable, value-level details."""
        if self.confirmed:
            return CONFIRMATION_CONDITION
        parts = [self.reason] if self.reason else []
        if self.proposal_missing:
            parts.append(
                "the assistant proposal immediately before the user's latest reply "
                "does not state: " + "; ".join(self.proposal_missing)
            )
        if self.proposal_conflicts:
            parts.append("that proposal " + "; ".join(self.proposal_conflicts))
        if self.reply_problems:
            parts.append("the user's latest reply " + "; ".join(self.reply_problems))
        parts.append(
            "present one complete proposal with these exact values, then wait for "
            "a new affirmative user reply before retrying"
        )
        return f"{CONFIRMATION_CONDITION}: " + ". ".join(parts)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text))


def _flatten(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(_flatten(v) for k, v in value.items() if not k.startswith("_"))
    if isinstance(value, list):
        return " ".join(_flatten(v) for v in value)
    return str(value)


def _money_text(amount: float) -> str:
    return f"${amount:g}"


def _source(payment_id: str, user: User | None) -> str | None:
    method = user.payment_methods.get(payment_id) if user else None
    if method is not None:
        return method.source
    return next((s for s in _SOURCES if payment_id.startswith(s + "_")), None)


def _original_payment(record: Reservation | None) -> str | None:
    """The single method charged for a booking, or None if not unique."""
    if record is None:
        return None
    charged = {p.payment_id for p in record.payment_history if p.amount > 0}
    return next(iter(charged)) if len(charged) == 1 else None


def _refund_total(record: Reservation | None) -> float | None:
    """Net amount the cancellation tool returns to the original methods."""
    if record is None:
        return None
    return float(sum(p.amount for p in record.payment_history))


def _action_payment_ids(call: ToolCall, record: Reservation | None) -> list[str]:
    args = call.arguments
    if call.name == "cancel_reservation":
        return sorted({p.payment_id for p in record.payment_history} if record else [])
    ids = [p.get("payment_id", "") for p in args.get("payment_methods", [])]
    if "payment_id" in args:
        ids.append(args["payment_id"])
    return ids


def _allowed_amounts(call: ToolCall, record: Reservation | None) -> set[float]:
    """Dollar amounts a reply may restate: only those computed for this call."""
    args = call.arguments
    amounts = [p.get("amount") for p in args.get("payment_methods", [])]
    if args.get("_expected_amount") is not None:
        amounts.append(abs(args["_expected_amount"]))
    if call.name == "cancel_reservation":
        amounts.append(_refund_total(record))
    return {float(a) for a in amounts if a is not None}


def _stated_amounts(text: str) -> list[float]:
    return [float(a or b) for a, b in _AMOUNT.findall(text)]


def _reply_problems(
    text: str, call: ToolCall, user: User | None, record: Reservation | None
) -> list[str]:
    """Accept grounded affirmative prose without accepting revised parameters.

    An affirmative opening is necessary, not sufficient. Unknown words, new
    values, negation, conditional approval and transfer requests fail closed.
    Restated amounts and payment types must match values computed for the call.
    """
    text = normalized(text, call.arguments.get("_policy_year")).strip()
    problems = []
    qualifiers = sorted({a or b for a, b in _QUALIFIERS.findall(text)})
    if qualifiers:
        problems.append(
            "contains qualifying or negating words (" + ", ".join(qualifiers) + ")"
        )
    if not _OPENING.match(text):
        problems.append("does not begin with an explicit approval such as 'yes'")
    vocabulary = set(_BASE_VOCABULARY)
    if call.name == "update_reservation_baggages":
        vocabulary.update(
            "keep keeping our current existing seats seat they are".split()
        )
    vocabulary |= _tokens(normalized(_flatten(call.arguments)))
    if record:
        vocabulary |= _tokens(
            normalized(_flatten([p.model_dump() for p in record.passengers]))
        )
    allowed = _allowed_amounts(call, record)
    vocabulary.update(f"{amount:g}" for amount in allowed)
    payment_ids = _action_payment_ids(call, record)
    for payment_id in payment_ids:
        vocabulary |= _tokens(normalized(payment_id))
        method = user.payment_methods.get(payment_id) if user else None
        if method is not None and method.source == "credit_card":
            vocabulary.add(method.last_four)
    unknown = sorted(_tokens(text) - vocabulary)
    if unknown:
        problems.append(
            "contains words or values absent from this action ("
            + ", ".join(unknown[:8])
            + ")"
        )
    wrong = sorted({a for a in _stated_amounts(text) if a not in allowed})
    if wrong:
        problems.append(
            "states amounts not computed for this action ("
            + ", ".join(_money_text(a) for a in wrong)
            + ")"
        )
    sources = {_source(payment_id, user) for payment_id in payment_ids}
    named = sorted(
        source
        for source, pattern in _SOURCES.items()
        if source not in sources and re.search(pattern, text)
    )
    if named:
        problems.append(
            "names payment types this action does not use ("
            + ", ".join(s.replace("_", " ") for s in named)
            + ")"
        )
    if call.name != "cancel_reservation" and re.search(r"\bcancel", text):
        problems.append("asks for a cancellation, which is a different action")
    # A repeated bag count is a binding, not just an affirmative filler word.
    for count in re.findall(r"\b(\d+)\s+(?:checked\s+)?bags?\b", text):
        total = call.arguments.get("total_baggages")
        delta = (
            total - record.total_baggages
            if record is not None and total is not None
            else total
        )
        if int(count) not in {total, delta}:
            problems.append(f"states {count} bags, not {total} total")
    if call.name == "update_reservation_passengers":
        changed = [
            normalized(p["first_name"] + " " + p["last_name"])
            for i, p in enumerate(call.arguments["passengers"])
            if record is None
            or i >= len(record.passengers)
            or p != record.passengers[i].model_dump()
        ]
        for name in re.findall(
            r"\b(?:name (?:updated )?to|name to)\s+([a-z]+\s+[a-z]+)\b", text
        ):
            if name not in changed:
                problems.append(f"requests name {name!r}, not one in this call")
    return problems


def _payment_in_text(
    text: str,
    payment_id: str,
    user: User | None,
    record: Reservation | None = None,
) -> bool:
    """Identify a saved method by ID or by one unambiguous description."""
    if contains(text, payment_id):
        return True
    method = user.payment_methods.get(payment_id) if user else None
    if (
        method
        and method.source == "credit_card"
        and sum(
            1
            for saved in user.payment_methods.values()
            if saved.source == "credit_card" and saved.last_four == method.last_four
        )
        == 1
        and re.search(
            rf"(?:ending(?: in)?|last four(?: digits)?|\*{{2,}})\s*{re.escape(method.last_four)}(?!\d)",
            text,
            re.I,
        )
    ):
        return True
    # "Original payment method" resolves only to the booking's single charged
    # method; several original methods (or none) leave it ambiguous.
    if _original_payment(record) == payment_id and _ORIGINAL.search(text.lower()):
        return True
    return bool(
        method
        and sum(1 for m in user.payment_methods.values() if m.source == method.source)
        == 1
        and re.search(_SOURCES[method.source], text, re.I)
    )


def _payment_hint(payment_id: str, user: User | None) -> str:
    method = user.payment_methods.get(payment_id) if user else None
    if method is not None and method.source == "credit_card":
        return f"payment method {payment_id} (or its card ending in {method.last_four})"
    return f"payment method {payment_id}"


def _proposal_problems(
    text: str, call: ToolCall, user: User | None, record: Reservation | None
) -> tuple[list[str], list[str]]:
    """Return (missing values, contradictions) for the proposal's text."""
    args = call.arguments
    missing: list[str] = []
    conflicts: list[str] = []
    if call.name not in _ACTIONS:
        return [], ["has no confirmation rule for this tool"]
    action, label = _ACTIONS[call.name]
    text = normalized(text, args.get("_policy_year"))
    structured = False
    opening = text.find("{")
    if opening >= 0:
        try:
            proposed, _ = json.JSONDecoder().raw_decode(text[opening:])
        except ValueError:
            proposed = None
        if isinstance(proposed, dict) and set(proposed) & set(args):
            expected = {
                key: value for key, value in args.items() if not key.startswith("_")
            }
            if proposed != json.loads(normalized(json.dumps(expected))):
                conflicts.append("shows JSON arguments that differ from this call")
            else:
                structured = True
    if not re.search(action, text.replace("_", " ")):
        missing.append(f"the action ({label})")
    refusal = sorted(set(_REFUSAL.findall(text)))
    if refusal:
        conflicts.append("contains refusal wording (" + ", ".join(refusal) + ")")
    if "reservation_id" in args and not contains(text, args["reservation_id"]):
        missing.append(f"reservation {args['reservation_id']}")
    if call.name in {"book_reservation", "update_reservation_flights"}:
        if not contains(text, args.get("cabin", "")):
            missing.append(f"cabin {args.get('cabin')}")
        for flight in args.get("flights", []):
            if not all(
                contains(text, flight.get(key, "")) for key in ("flight_number", "date")
            ):
                missing.append(
                    f"flight {flight.get('flight_number')} on {flight.get('date')}"
                )
    if call.name == "book_reservation":
        for key in ("origin", "destination"):
            if not contains(text, args.get(key, "")):
                missing.append(f"{key} {args.get(key)}")
        insurance_no = bool(
            re.search(
                r"\b(no|without|decline\w*)\s+(?:travel\s+)?insurance\b|insurance[\s\"':=]+no\b",
                text,
            )
        )
        if not re.search(r"\binsurance\b", text) or insurance_no != (
            args.get("insurance") == "no"
        ):
            missing.append(f"travel insurance: {args.get('insurance')}")
    if call.name in {"book_reservation", "update_reservation_passengers"}:
        for index, passenger in enumerate(args.get("passengers", [])):
            before = (
                record.passengers[index].model_dump()
                if record and index < len(record.passengers)
                else {}
            )
            if call.name == "update_reservation_passengers" and before == passenger:
                continue
            fields = (
                ("first_name", "last_name", "dob")
                if call.name == "book_reservation" or not before
                else tuple(
                    key
                    for key in ("first_name", "last_name", "dob")
                    if before.get(key) != passenger.get(key)
                )
            )
            absent = [
                key for key in fields if not contains(text, passenger.get(key, ""))
            ]
            full_name = passenger["first_name"] + " " + passenger["last_name"]
            if (
                call.name == "update_reservation_passengers"
                and not structured
                and not contains(text, full_name)
            ):
                absent.append("full name")
            if absent:
                missing.append(
                    f"passenger {index + 1} {full_name} ("
                    + ", ".join(
                        f"{key}={passenger.get(key)}" if key in passenger else key
                        for key in dict.fromkeys(absent)
                    )
                    + ")"
                )
    if call.name in {"book_reservation", "update_reservation_baggages"}:
        # Require the total baggage count adjacent to its meaning, or a JSON key.
        total = args.get("total_baggages")
        if not re.search(
            rf"\b{total}\s+(?:free\s+)?(?:checked\s+)?(?:bags?|baggages?)\b"
            rf"|total_baggages[\s\"':=]+{total}(?!\d)"
            # Label form "Total checked bags: 3"; never an extra/paid/free subtotal.
            rf"|(?<!extra )(?<!paid )(?<!additional )(?<!free )"
            rf"\b(?:total\s+(?:checked\s+)?|checked\s+)(?:bags?|baggages?)"
            rf"\s*(?:[:=]|is|are)\s*{total}(?!\d)",
            text,
        ):
            missing.append(f"total checked bags: {total} (e.g. '{total} checked bags')")
    payment_ids = [p.get("payment_id", "") for p in args.get("payment_methods", [])]
    if "payment_id" in args and args.get("_expected_amount") != 0:
        payment_ids.append(args["payment_id"])
    for payment_id in payment_ids:
        if not _payment_in_text(text, payment_id, user, record):
            missing.append(_payment_hint(payment_id, user))
    if call.name == "book_reservation" and not structured:
        clauses = re.split(r"[;\n]|\.(?!\d)|\s*\+\s*|\s+and\s+(?=\$)|,\s*(?=\$)", text)
        for payment in args.get("payment_methods", []):
            if not any(
                _payment_in_text(clause, payment["payment_id"], user)
                and _money(clause, payment["amount"])
                and sum(
                    _payment_in_text(clause, other["payment_id"], user)
                    for other in args["payment_methods"]
                )
                == 1
                for clause in clauses
            ):
                missing.append(
                    f"${payment['amount']} on {payment['payment_id']} in its own clause"
                )
    amount = args.get("_expected_amount")
    if amount is not None and call.name == "book_reservation":
        # The bound per-method amounts above already disclose the entire charge.
        # If a total is stated as well, it must agree rather than contradict it.
        totals = re.findall(
            r"\btotal(?:\s+(?:price|cost|amount))?\s*(?:is|of|:|=)?\s*\$?\s*(\d+(?:\.\d{1,2})?)",
            text,
        )
        wrong = sorted({float(value) for value in totals if float(value) != amount})
        if wrong:
            conflicts.append(
                "states a total of "
                + ", ".join(_money_text(v) for v in wrong)
                + f" instead of {_money_text(amount)}"
            )
    elif amount is not None:
        if amount == 0:
            if not (
                _money(text, 0)
                or re.search(
                    r"\b(no (?:(?:extra|additional) )?(?:baggage )?(?:charges?|costs?|fees?|payment)|free|no price difference)\b",
                    text,
                )
            ):
                missing.append("that there is no additional charge")
        elif not _money(text, abs(amount)):
            kind = "refund" if amount < 0 else "charge"
            missing.append(f"the {kind} of {_money_text(abs(amount))}")
        elif amount < 0 and not re.search(
            r"\b(refund(?:ed)?|return(?:ed)?|credit(?:ed)?)\b", text
        ):
            missing.append(f"that {_money_text(abs(amount))} is a refund")
    if call.name == "cancel_reservation" and record is not None:
        total = _refund_total(record)
        promised = {
            float(value)
            for value in re.findall(
                r"\brefund\w*[^.;\n$\d]{0,40}\$\s*(\d+(?:\.\d{1,2})?)", text
            )
        }
        if promised - {total}:
            conflicts.append(
                "promises a refund of "
                + ", ".join(_money_text(v) for v in sorted(promised - {total}))
                + f", but the reservation's payments total {_money_text(total)}"
            )
    return missing, conflicts


def _money(text: str, amount: int) -> bool:
    """Require a currency/amount context; dates cannot establish a charge."""
    number = rf"{amount}(?:\.00)?(?!\d|\.\d)"
    return bool(
        re.search(
            rf"\$\s*{number}|\b{number}\s*(?:dollars?|usd)\b|\b(?:amount|total|charge|cost|fee|refund|price|difference|pay)[^.;\n\d]{{0,35}}\$?{number}",
            text,
        )
    )


def check_confirmation(
    transcript: list[Message],
    call: ToolCall,
    user: User | None = None,
    record: Reservation | None = None,
) -> ConfirmationCheck:
    """Bind the latest user affirmation to matching action details.

    Read-only tools between confirmation and execution are allowed. A new user
    turn revokes the prior confirmation. Completed writes consume confirmation
    for that action type; another type in the same confirmed proposal can follow.
    """
    previous = None
    result = ConfirmationCheck(
        confirmed=False, reason="no user reply has approved a proposal yet"
    )
    for message in transcript:
        if isinstance(message, UserMessage) and message.content:
            if previous is None:
                result = ConfirmationCheck(
                    confirmed=False,
                    reason="the user's latest message does not reply to an "
                    "assistant proposal",
                )
            else:
                missing, conflicts = _proposal_problems(previous, call, user, record)
                reply = _reply_problems(message.content, call, user, record)
                result = ConfirmationCheck(
                    confirmed=not (missing or conflicts or reply),
                    proposal_missing=missing,
                    proposal_conflicts=conflicts,
                    reply_problems=reply,
                )
            previous = None
        elif isinstance(message, AssistantMessage):
            if message.content:
                previous = message.content
            if any(tc.name == call.name for tc in message.tool_calls or []):
                # A pending write also cannot reuse an already granted approval.
                result = ConfirmationCheck(
                    confirmed=False,
                    reason=f"the previous approval was used by an earlier "
                    f"{call.name} call",
                )
    return result


def confirmed_action(
    transcript: list[Message],
    call: ToolCall,
    user: User | None = None,
    record: Reservation | None = None,
) -> bool:
    """Boolean form of :func:`check_confirmation`."""
    return check_confirmation(transcript, call, user, record).confirmed
