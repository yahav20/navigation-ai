"""API-first flight search with SQLite fallback.

Every flight-fetch path in the agent (FlightSearchNode, advisor @tool fetch_flights,
node_alternative) should go through `search_flights_with_fallback` so callers
get the live Travelpayouts feed when it has data and the static SQLite fixtures
as a safety net when it doesn't (unresolvable city, API outage, sparse date).
"""
from __future__ import annotations

from datetime import date, timedelta

from providers.sqlite import SQLiteDataProvider
from providers.travelpayouts import search_flights as _api_search_flights
from providers.travelpayouts.flight_api import search_round_trip_legs as _api_round_trip_legs

_db = SQLiteDataProvider()


def _default_departure_at() -> str:
    """Month-wide query roughly one month from today when the caller omits a date."""
    return (date.today() + timedelta(days=30)).strftime("%Y-%m")


def _usable(items: object) -> list[dict]:
    if not isinstance(items, list):
        return []
    return [
        item for item in items
        if isinstance(item, dict) and ("flight_number" in item or item.get("route"))
    ]


def search_flights_with_fallback(
    origin: str,
    destination: str,
    departure_at: str | None = None,
) -> list[dict]:
    """Return flight offers, preferring the Travelpayouts API and falling back to SQLite.

    `departure_at` is YYYY-MM-DD (specific date) or YYYY-MM (month-wide). When
    omitted, defaults to roughly one month from today.
    """
    api_results = _usable(_api_search_flights(origin, destination, departure_at or _default_departure_at()))
    if api_results:
        return api_results

    direct = _usable(_db.fetch_flights(origin, destination))
    if direct:
        return direct

    return _usable(_db.find_connecting_flights(origin, destination))


def search_round_trip_legs_with_fallback(
    origin: str,
    destination: str,
    departure_at: str,
    return_at: str,
) -> tuple[list[dict], list[dict]]:
    """Round-trip-matched (outbound_legs, return_legs) from the live feed.

    Supplements the one-way search so trip lengths stay close to the request. No SQLite
    fallback — the static fixtures are single-leg — so this returns empty lists when the
    live round-trip feed has nothing, leaving the one-way results untouched.
    """
    out, ret = _api_round_trip_legs(origin, destination, departure_at, return_at)
    return _usable(out), _usable(ret)
