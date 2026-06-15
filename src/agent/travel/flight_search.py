"""Deterministic flight search node — pulls outbound + return flights via the API-first helper."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

from agent.core.state import AgentState
from providers.flights import search_flights_with_fallback


def _is_direct_flight(item: object) -> bool:
    return isinstance(item, dict) and bool(item.get("flight_number"))


def _is_connecting_route(item: object) -> bool:
    return isinstance(item, dict) and bool(item.get("route"))


def _usable_flights(items: object) -> list[dict]:
    if not isinstance(items, list):
        return []
    return [item for item in items if _is_direct_flight(item) or _is_connecting_route(item)]


def _return_date(trip_start: str, trip_days: int) -> str:
    """Compute the return-leg query string from trip_start + trip_days."""
    try:
        start = datetime.strptime(trip_start, "%Y-%m-%d").date()
    except ValueError:
        try:
            # "YYYY-MM" month-only format — treat the 1st as the base date
            start = datetime.strptime(trip_start + "-01", "%Y-%m-%d").date()
            return (start + timedelta(days=max(trip_days, 1))).strftime("%Y-%m-%d")
        except ValueError:
            return trip_start  # unrecognised format — pass through unchanged
    return (start + timedelta(days=max(trip_days, 1))).strftime("%Y-%m-%d")


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

        return_at = _return_date(trip_start, trip_days)
        with ThreadPoolExecutor(max_workers=2) as pool:
            outbound_future = pool.submit(
                search_flights_with_fallback, origin, destination, trip_start,
            )
            return_future = pool.submit(
                search_flights_with_fallback, destination, origin, return_at,
            )

        outbound = _usable_flights(outbound_future.result())
        inbound = _usable_flights(return_future.result())

        return {
            "flight_options": outbound,
            "return_flight_options": inbound,
            "has_flights": bool(outbound) and bool(inbound),
        }
