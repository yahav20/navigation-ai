"""Deterministic group-pricing formulas for the travel agent.

Single source of truth for how group size and room count translate into
cost estimates. No LLM involvement — all calculations are pure functions
on concrete numbers. Import this wherever group-aware prices are needed
so the formulas stay in one place.

Agreed rates:
  - Adults pay full price for flights.
  - Children pay CHILD_RATE (80%) of the adult flight price.
  - Hotels are billed per room, not per person.
  - Activities / restaurants are billed per head (no child discount).
"""

CHILD_RATE: float = 0.8


def flight_group_price(price_per_person: float, adults: int, children: int) -> float:
    """Total flight cost for the whole group.

    Adults pay full price; children pay CHILD_RATE of the adult price.
    """
    return int(price_per_person * adults + price_per_person * children * CHILD_RATE)


def hotel_group_price(price_per_night: float, num_rooms: int, trip_days: int) -> float:
    """Total hotel cost for the stay — billed by rooms, not by heads."""
    return int(price_per_night * num_rooms * trip_days)


def activity_group_price(price: float, adults: int, children: int) -> float:
    """Total cost of an activity / restaurant for the whole group (full price per head)."""
    return int(price * (adults + children))


def group_label(adults: int, children: int) -> str:
    """Human-readable group description, e.g. '2 adults, 1 child'."""
    parts: list[str] = []
    if adults:
        parts.append(f"{adults} adult{'s' if adults != 1 else ''}")
    if children:
        parts.append(f"{children} child{'ren' if children != 1 else ''}")
    return ", ".join(parts) if parts else "1 adult"
