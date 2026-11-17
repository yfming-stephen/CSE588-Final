"""Explicit business-policy predicates for all six airline write tools."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from tau2.data_model.message import Message, ToolCall
from tau2.domains.airline.data_model import (
    CabinClass,
    FlightInfo,
    FlightType,
    Insurance,
    Passenger,
    Payment,
    Reservation,
    User,
)
from tau2.guards.airline_dialogue import (
    cancellation_reason,
    check_confirmation,
    compensation_topics,
    supplied_identity,
    user_requested_compensation,
)
from tau2.guards.airline_evidence import AirlineEvidence, collect_evidence, flight_key


class Arguments(BaseModel):
    """Reject unexpected parameters and coercions at the guard boundary."""

    model_config = ConfigDict(extra="forbid", strict=True)


class CancelArguments(Arguments):
    reservation_id: str = Field(min_length=1)


class FlightArguments(CancelArguments):
    cabin: CabinClass
    flights: list[FlightInfo] = Field(min_length=1)
    payment_id: str = Field(min_length=1)


class BaggageArguments(CancelArguments):
    total_baggages: int = Field(ge=0)
    nonfree_baggages: int = Field(ge=0)
    payment_id: str = Field(min_length=1)


class PassengerArguments(CancelArguments):
    passengers: list[Passenger] = Field(min_length=1, max_length=5)


class BookArguments(Arguments):
    user_id: str = Field(min_length=1)
    origin: str = Field(pattern=r"^[A-Z]{3}$")
    destination: str = Field(pattern=r"^[A-Z]{3}$")
    flight_type: FlightType
    cabin: CabinClass
    flights: list[FlightInfo] = Field(min_length=1)
    passengers: list[Passenger] = Field(min_length=1, max_length=5)
    payment_methods: list[Payment] = Field(min_length=1, max_length=5)
    total_baggages: int = Field(ge=0)
    nonfree_baggages: int = Field(ge=0)
    insurance: Insurance


class CertificateArguments(Arguments):
    user_id: str = Field(min_length=1)
    amount: int = Field(gt=0)


ARGUMENT_MODELS = {
    "cancel_reservation": CancelArguments,
    "book_reservation": BookArguments,
    "update_reservation_flights": FlightArguments,
    "update_reservation_baggages": BaggageArguments,
    "update_reservation_passengers": PassengerArguments,
    "send_certificate": CertificateArguments,
}


class Condition(BaseModel):
    """One independently logged policy predicate."""

    condition: str
    met: bool
    evidence: Any = None
    category: Literal["policy", "evidence", "dialogue", "schema"] = "policy"


class AirlineChecks:
    """Evaluate policy against one call's validated arguments and evidence."""

    def __init__(
        self,
        call: ToolCall,
        transcript: list[Message],
        evidence: AirlineEvidence,
        now: datetime | None,
    ):
        self.call, self.args, self.transcript, self.evidence, self.now = (
            call,
            call.arguments,
            transcript,
            evidence,
            now,
        )
        self.conditions: list[Condition] = []

    def add(
        self, condition: str, met: bool, evidence: Any = None, category="policy"
    ) -> None:
        """Append an auditable check with JSON-compatible evidence."""
        self.conditions.append(
            Condition(
                condition=condition, met=bool(met), evidence=evidence, category=category
            )
        )

    def identity(self, uid: str) -> None:
        """Require the target user's identity to be supplied in user dialogue."""
        self.add(
            "user supplied reservation owner's ID",
            supplied_identity(self.transcript, uid),
            uid,
            "dialogue",
        )

    def user(self, uid: str) -> User | None:
        """Require a validated user-profile result for profile-dependent checks."""
        user = self.evidence.users.get(uid)
        self.add(
            f"matching user profile available; query get_user_details(user_id={uid!r}) before retrying",
            user is not None,
            uid,
            "evidence",
        )
        return user

    def confirmation(self, user: User | None = None, amount: int | None = None) -> None:
        """Check that approval matches this call's action, target and parameters."""
        call = self.call.model_copy(deep=True)
        if self.now is not None:
            call.arguments["_policy_year"] = self.now.year
        if amount is not None:
            call.arguments["_expected_amount"] = amount
        result = check_confirmation(
            self.transcript,
            call,
            user,
            self.evidence.reservations.get(self.args.get("reservation_id")),
        )
        # The condition text carries value-level gaps so feedback is actionable;
        # it always starts with CONFIRMATION_CONDITION for aggregation.
        self.add(
            result.feedback(),
            result.confirmed,
            {
                "arguments": self.call.arguments,
                "expected_amount": amount,
                **result.model_dump(exclude={"confirmed"}),
            },
            "dialogue",
        )

    def reservation(self) -> Reservation | None:
        """Resolve current booking and common identity/status checks."""
        rid = self.args["reservation_id"]
        record = self.evidence.reservations.get(rid)
        self.add(
            "matching reservation tool result available",
            record is not None,
            rid,
            "evidence",
        )
        if record:
            self.identity(record.user_id)
            self.add(
                "reservation not already cancelled",
                record.status != "cancelled",
                record.status,
            )
        return record

    def unflown(self, record: Reservation) -> None:
        """Unknown status cannot establish that a segment has not flown."""
        statuses = [self.evidence.statuses.get(flight_key(f)) for f in record.flights]
        self.add(
            "all flight statuses verified and no segment flown",
            bool(statuses)
            and all(
                s in {"available", "on time", "delayed", "cancelled"} for s in statuses
            ),
            statuses,
            "evidence",
        )

    def passengers(self, passengers: list[dict]) -> None:
        """Check complete passenger identities and actual calendar dates."""
        valid = True
        for passenger in passengers:
            try:
                birthday = datetime.strptime(passenger["dob"], "%Y-%m-%d")
                valid &= bool(
                    passenger["first_name"].strip() and passenger["last_name"].strip()
                ) and birthday <= (self.now or datetime.min)
            except ValueError:
                valid = False
        self.add("valid passenger names and dates of birth", valid, passengers)

    def baggage(self, user: User, cabin: str, count: int) -> None:
        """Translate the cabin/membership allowance table to a paid-bag count."""
        allowance = {"basic_economy": 0, "economy": 1, "business": 2}[cabin] + {
            "regular": 0,
            "silver": 1,
            "gold": 2,
        }[user.membership]
        expected = max(0, self.args["total_baggages"] - allowance * count)
        self.add(
            "paid bags match membership and cabin allowance",
            self.args["nonfree_baggages"] == expected,
            {"free_per_passenger": allowance, "expected_paid_bags": expected},
        )

    def payment(self, user: User, payment_id: str, amount: int) -> None:
        """Modification payment must be a saved credit/gift card with funds."""
        method = user.payment_methods.get(payment_id)
        self.add(
            "saved credit or gift card for modification",
            method is not None and method.source in {"credit_card", "gift_card"},
            payment_id,
        )
        if method and method.source == "gift_card":
            self.add(
                "gift card covers charge",
                amount <= method.amount,
                {"charge": amount, "balance": method.amount},
            )

    def cancel(self) -> None:
        """Cancellation: no flown segments AND at least one eligibility exception."""
        record = self.reservation()
        if not record:
            return
        # The profile, when already retrieved, lets confirmation check payment
        # wording; cancellation itself does not require profile evidence.
        self.confirmation(self.evidence.users.get(record.user_id))
        reason = cancellation_reason(self.transcript, record.reservation_id)
        self.add("cancellation reason supplied", reason is not None, reason, "dialogue")
        self.unflown(record)
        recent = False
        try:
            age = (
                self.now - datetime.fromisoformat(record.created_at)
                if self.now
                else None
            )
            recent = age is not None and timedelta(0) <= age <= timedelta(hours=24)
        except (ValueError, TypeError):
            pass
        exceptions = {
            "within_24_hours": recent,
            "airline_cancelled": any(
                self.evidence.statuses.get(flight_key(f)) == "cancelled"
                for f in record.flights
            ),
            "business": record.cabin == "business",
            "covered_insurance": record.insurance == "yes" and reason == "covered",
        }
        self.add(
            "at least one cancellation eligibility condition",
            any(exceptions.values()),
            exceptions,
        )

    def change_passengers(self) -> None:
        """Passenger identities may change, passenger count may not."""
        record = self.reservation()
        if not record:
            return
        self.add(
            "passenger count unchanged",
            len(self.args["passengers"]) == len(record.passengers),
            {"old": len(record.passengers), "new": len(self.args["passengers"])},
        )
        self.passengers(self.args["passengers"])
        self.confirmation()

    def change_baggage(self) -> None:
        """Only add bags; compute the paid count and check the saved payment."""
        record = self.reservation()
        if not record:
            return
        self.add(
            "checked bags cannot be removed",
            self.args["total_baggages"] >= record.total_baggages,
            {"old": record.total_baggages, "new": self.args["total_baggages"]},
        )
        user = self.user(record.user_id)
        charge = 50 * max(0, self.args["nonfree_baggages"] - record.nonfree_baggages)
        if user:
            self.baggage(user, record.cabin, len(record.passengers))
            self.payment(user, self.args["payment_id"], charge)
        self.confirmation(user, charge)

    def itinerary(self, retained: dict | None = None) -> tuple[list[Any], int | None]:
        """Verify dated flight availability and sum actual per-passenger prices."""
        retained = retained or {}
        flights, prices = [], []
        keys = [flight_key(f) for f in self.args["flights"]]
        self.add("no duplicate flight segments", len(set(keys)) == len(keys), keys)
        for key in keys:
            flight = retained.get(key) or self.evidence.flights.get(key)
            self.add(
                "matching flight search result or retained segment available",
                flight is not None,
                key,
                "evidence",
            )
            if flight is None:
                continue
            flights.append(flight)
            if key in retained:
                prices.append(flight.price)
            else:
                seats = flight.available_seats.get(self.args["cabin"], 0)
                count = len(self.args.get("passengers", [])) or self._passenger_count
                price = flight.prices.get(self.args["cabin"])
                self.add(
                    "new flight available with enough seats",
                    self.evidence.statuses.get(key) == "available"
                    and seats >= count
                    and isinstance(price, int)
                    and price >= 0,
                    {
                        "flight": key,
                        "seats": seats,
                        "passengers": count,
                        "price": price,
                    },
                )
                if price is not None:
                    prices.append(price)
        if len(flights) == len(keys):
            self.add(
                "connected itinerary",
                all(a.destination == b.origin for a, b in zip(flights, flights[1:])),
                keys,
            )
            self.add("flight dates valid and ordered", self._dates_valid(keys), keys)
        return flights, sum(prices) if len(prices) == len(keys) else None

    @staticmethod
    def _dates_valid(keys) -> bool:
        try:
            dates = [datetime.strptime(key[1], "%Y-%m-%d") for key in keys]
            return dates == sorted(dates)
        except ValueError:
            return False

    def route(self, flights, origin: str, destination: str, flight_type: str) -> None:
        """Retain endpoints and trip type; allow connecting segments."""
        valid = bool(flights) and flights[0].origin == origin
        if flight_type == "one_way":
            valid = valid and flights[-1].destination == destination
        else:
            valid = (
                valid
                and flights[-1].destination == origin
                and any(f.destination == destination for f in flights[:-1])
            )
        self.add(
            "itinerary matches origin destination and trip type",
            valid,
            {"origin": origin, "destination": destination, "type": flight_type},
        )

    def change_flights(self) -> None:
        """Distinguish flight changes from cabin-only changes using dated segments."""
        record = self.reservation()
        if not record:
            return
        self._passenger_count = len(record.passengers)
        same_flights = [flight_key(f) for f in record.flights] == [
            flight_key(f) for f in self.args["flights"]
        ]
        cabin_change = self.args["cabin"] != record.cabin
        self.add(
            "basic economy may change cabin but not flights",
            same_flights or record.cabin != "basic_economy",
            {"same_flights": same_flights, "old_cabin": record.cabin},
        )
        if cabin_change:
            self.unflown(record)
        retained = (
            {flight_key(f): f for f in record.flights} if not cabin_change else {}
        )
        flights, price = self.itinerary(retained)
        if len(flights) == len(self.args["flights"]):
            self.route(flights, record.origin, record.destination, record.flight_type)
        amount = (
            (price - sum(f.price for f in record.flights)) * len(record.passengers)
            if price is not None
            else None
        )
        user = self.user(record.user_id)
        if user and amount is not None:
            self.payment(user, self.args["payment_id"], amount)
        self.confirmation(user, amount)

    def book(self) -> None:
        """Booking rules: itinerary, passengers, bags, insurance, payments, approval."""
        self.identity(self.args["user_id"])
        self.passengers(self.args["passengers"])
        self._passenger_count = len(self.args["passengers"])
        flights, price = self.itinerary()
        if len(flights) == len(self.args["flights"]):
            self.route(
                flights,
                self.args["origin"],
                self.args["destination"],
                self.args["flight_type"],
            )
        user = self.user(self.args["user_id"])
        payments = self.args["payment_methods"]
        ids = [p["payment_id"] for p in payments]
        self.add(
            "distinct payment methods and nonnegative amounts",
            len(set(ids)) == len(ids) and all(p["amount"] >= 0 for p in payments),
            payments,
        )
        if user:
            self.baggage(user, self.args["cabin"], self._passenger_count)
            counts = {"credit_card": 0, "gift_card": 0, "certificate": 0}
            for payment in payments:
                method = user.payment_methods.get(payment["payment_id"])
                self.add(
                    "booking payment saved in user profile",
                    method is not None,
                    payment["payment_id"],
                )
                if method:
                    counts[method.source] += 1
                    if method.source in {"gift_card", "certificate"}:
                        self.add(
                            "booking payment has sufficient balance",
                            method.amount >= payment["amount"],
                            payment,
                        )
            self.add(
                "payment method count limits",
                counts["credit_card"] <= 1
                and counts["certificate"] <= 1
                and counts["gift_card"] <= 3,
                counts,
            )
        total = (
            None
            if price is None
            else price * self._passenger_count
            + 50 * self.args["nonfree_baggages"]
            + (30 * self._passenger_count if self.args["insurance"] == "yes" else 0)
        )
        self.add(
            "payments equal flight baggage and insurance total",
            total is not None and sum(p["amount"] for p in payments) == total,
            {"expected": total, "paid": sum(p["amount"] for p in payments)},
        )
        self.confirmation(user, total)

    @staticmethod
    def _compensation_options(evidence, transcript, uid: str) -> list[dict]:
        """Eligible fixed-amount incidents identified from one transcript prefix."""
        user = evidence.users.get(uid)
        candidates = []
        for record in evidence.reservations.values():
            if record.user_id != uid or not user_requested_compensation(
                transcript, record.reservation_id
            ):
                continue
            eligible = user is not None and (
                user.membership in {"silver", "gold"}
                or record.insurance == "yes"
                or record.cabin == "business"
            )
            cancelled = any(
                evidence.statuses.get(flight_key(f)) == "cancelled"
                for f in record.flights
            )
            delay_changes = [
                change
                for change in evidence.changes
                if change.call.name
                in {"cancel_reservation", "update_reservation_flights"}
                and change.before
                and change.before.reservation_id == record.reservation_id
                and "delayed" in change.statuses
            ]
            delay_eligible = any(
                user is not None
                and (
                    user.membership in {"silver", "gold"}
                    or change.before.insurance == "yes"
                    or change.before.cabin == "business"
                )
                for change in delay_changes
            )
            topics = compensation_topics(transcript, record.reservation_id)
            if cancelled and eligible and "cancelled" in topics:
                candidates.append(
                    {
                        "reservation_id": record.reservation_id,
                        "kind": "cancelled",
                        "amount": 100 * len(record.passengers),
                    }
                )
            if delay_changes and delay_eligible and "delayed" in topics:
                candidates.append(
                    {
                        "reservation_id": record.reservation_id,
                        "kind": "delayed",
                        "amount": 50 * len(record.passengers),
                    }
                )
        return candidates

    def compensate(self) -> None:
        """Check requested compensation, actual disruption, remedy, amount and reuse."""
        uid = self.args["user_id"]
        self.identity(uid)
        self.user(uid)
        candidates = self._compensation_options(self.evidence, self.transcript, uid)
        consumed = set()
        # Re-evaluate earlier certificate calls against their own historical
        # prefixes. This is local replay, not a persistent provenance database.
        for change in self.evidence.changes:
            if (
                change.call.name != "send_certificate"
                or change.call.arguments.get("user_id") != uid
            ):
                continue
            prefix = self.transcript[: change.index]
            earlier = collect_evidence(prefix)
            for option in self._compensation_options(earlier, prefix, uid):
                if option["amount"] == change.call.arguments.get("amount"):
                    consumed.add((option["reservation_id"], option["kind"]))
        matching = [
            option
            for option in candidates
            if option["amount"] == self.args["amount"]
            and (option["reservation_id"], option["kind"]) not in consumed
        ]
        self.add(
            "one requested eligible compensation incident with correct amount and completed delay remedy",
            len(matching) == 1
            and len({option["reservation_id"] for option in candidates}) == 1,
            {"candidates": candidates, "already_compensated": sorted(consumed)},
        )
