"""LangChain tools for fetching flight options and filter dimensions."""
from langchain_core.tools import tool

from providers.flights import search_flights_with_fallback
from tools.dependencies import data_provider


@tool
def fetch_flights(origin: str, destination: str, departure_at: str | None = None) -> list[dict]:
    """Fetch available flights between two cities for an approximate departure date.

    `departure_at` is YYYY-MM-DD for a specific day or YYYY-MM for a whole month.
    When omitted, defaults to roughly one month from today (month-wide query).
    Returns direct and connecting offers sorted cheapest first. Pulls from the
    Travelpayouts API first and falls back to the SQLite fixtures if the API
    has no usable data for the route.
    """
    return search_flights_with_fallback(origin, destination, departure_at)


@tool
def find_connecting_flights(origin: str, destination: str, departure_at: str | None = None) -> list[dict]:
    """Return only connecting (>=1 stop) flights for the route.

    Use when the user explicitly wants stopovers or when direct flights are
    missing/too expensive. `departure_at` follows the same format as
    `fetch_flights`. API-first with SQLite fallback.
    """
    offers = search_flights_with_fallback(origin, destination, departure_at)
    return [
        o for o in offers
        if isinstance(o, dict) and ((o.get("transfers") or 0) >= 1 or o.get("route"))
    ]


@tool
def get_flight_filter_options(origin: str, destination: str) -> dict:
    """Fetch distinct flight filtering dimensions for a route, including operating airlines and the ticket price range."""
    return data_provider.get_flight_dimensions(origin, destination)
