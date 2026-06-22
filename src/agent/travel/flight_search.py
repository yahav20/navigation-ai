"""Deterministic flight search node — pulls outbound + return flights via the API-first helper."""

from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

from agent.core.state import AgentState
from providers.flights import search_flights_with_fallback

# Both legs are queried month-wide (see `_to_month`/`_return_date`) because the
# flight feed is sparse for a single exact date, so the "best value" outbound
# and return picks come from independent searches and can land on dates far
# apart from each other. Accept a return flight only within this many days of
# outbound_date + trip_days, rather than silently pairing mismatched dates.
_RETURN_DATE_TOLERANCE_DAYS = 2


def _is_direct_flight(item: object) -> bool:
    return isinstance(item, dict) and bool(item.get("flight_number"))


def _is_connecting_route(item: object) -> bool:
    return isinstance(item, dict) and bool(item.get("route"))


def _usable_flights(items: object) -> list[dict]:
    if not isinstance(items, list):
        return []
    return [item for item in items if _is_direct_flight(item) or _is_connecting_route(item)]


def _flight_date(item: dict) -> date | None:
    """Extract the departure date from a direct or connecting flight offer."""
    raw = item.get("departure_time")
    if not raw and _is_connecting_route(item):
        raw = (item["route"][0] or {}).get("departure_time")
    if not isinstance(raw, str) or len(raw) < 10:
        return None
    try:
        return datetime.strptime(raw[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _align_by_trip_length(
    outbound: list[dict], inbound: list[dict], trip_days: int,
) -> list[dict]:
    """Prefer return offers within `_RETURN_DATE_TOLERANCE_DAYS` of outbound + trip_days.

    Without this, the outbound and return legs are independently "best value"
    picks from two separate month-wide searches and can end up dates apart that
    don't match the requested trip length at all.

    When nothing falls within tolerance, fall back to the closest available
    dates instead of returning empty — the flight feed is sparse enough that a
    real, slightly-off-date flight is far more useful than a false "no flights"
    (the actual date is still shown downstream, so nothing is misrepresented).
    """
    if not outbound or not inbound:
        return inbound

    outbound_date = _flight_date(outbound[0])
    if outbound_date is None:
        return inbound

    target = outbound_date + timedelta(days=max(trip_days, 1))
    dated = [(f, d) for f in inbound if (d := _flight_date(f)) is not None]
    if not dated:
        return inbound  # no parseable dates to compare — don't filter blind

    aligned = [f for f, d in dated if abs((d - target).days) <= _RETURN_DATE_TOLERANCE_DAYS]
    if aligned:
        return aligned

    dated.sort(key=lambda pair: abs((pair[1] - target).days))
    return [f for f, _d in dated]


def _to_month(date_str: str) -> str:
    """Normalise a 'YYYY-MM-DD' or 'YYYY-MM' string to a month-wide 'YYYY-MM' query.

    The Travelpayouts feed is far sparser for a single specific date than for a
    whole month, so we always query month-wide.
    """
    return date_str[:7] if len(date_str) >= 7 else date_str


def _return_date(trip_start: str, trip_days: int) -> str:
    """Compute the month-wide ('YYYY-MM') query string for the return leg.

    The return leg lands trip_days after trip_start; we search the whole month
    that day falls in rather than the exact date so the flight feed has data to
    return (a single specific date is frequently empty on thinner routes).
    """
    try:
        start = datetime.strptime(trip_start, "%Y-%m-%d").date()
    except ValueError:
        try:
            # "YYYY-MM" month-only format — treat the 1st as the base date
            start = datetime.strptime(trip_start + "-01", "%Y-%m-%d").date()
        except ValueError:
            return _to_month(trip_start)  # unrecognised format — best-effort month
    return (start + timedelta(days=max(trip_days, 1))).strftime("%Y-%m")


class FlightSearchNode:
    """Fetch outbound + return flights once and persist them on the state."""

    def __call__(self, state: AgentState) -> dict:
        """Return flight_options, return_flight_options, and has_flights for the requested route."""
        origin = state.get("current_city")
        destination = state.get("destination_city")
        trip_start = state.get("trip_start")
        trip_days = int(state.get("trip_days") or 3)

        if not origin or not destination or not trip_start:
            return {
                "flight_options": [],
                "return_flight_options": [],
                "has_flights": False,
            }

        outbound_at = _to_month(trip_start)
        return_at = _return_date(trip_start, trip_days)
        with ThreadPoolExecutor(max_workers=2) as pool:
            outbound_future = pool.submit(
                search_flights_with_fallback, origin, destination, outbound_at,
            )
            return_future = pool.submit(
                search_flights_with_fallback, destination, origin, return_at,
            )

        outbound = _usable_flights(outbound_future.result())
        inbound = _align_by_trip_length(outbound, _usable_flights(return_future.result()), trip_days)

        return {
            "flight_options": outbound,
            "return_flight_options": inbound,
            "has_flights": bool(outbound) and bool(inbound),
        }
