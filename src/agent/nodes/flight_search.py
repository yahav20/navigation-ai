"""Deterministic flight search node for the travel-planning path."""

from agent.state import AgentState
from tools.dependencies import data_provider


def _is_direct_flight(item: object) -> bool:
    return isinstance(item, dict) and bool(item.get("flight_number"))


def _is_connecting_route(item: object) -> bool:
    return isinstance(item, dict) and bool(item.get("route"))


def _usable_flights(items: object) -> list[dict]:
    if not isinstance(items, list):
        return []
    return [item for item in items if _is_direct_flight(item) or _is_connecting_route(item)]


class FlightSearchNode:
    """Fetch flights once and persist only route options needed by the graph."""

    def __call__(self, state: AgentState) -> dict:
        """Return flight_options and has_flights for the requested route."""
        origin = state.get("current_city")
        destination = state.get("destination_city")

        if not origin or not destination:
            return {"flight_options": [], "has_flights": False}

        direct_options = _usable_flights(data_provider.fetch_flights(origin, destination))
        if direct_options:
            return {"flight_options": direct_options, "has_flights": True}

        connecting_options = _usable_flights(
            data_provider.find_connecting_flights(origin, destination),
        )
        return {
            "flight_options": connecting_options,
            "has_flights": bool(connecting_options),
        }
