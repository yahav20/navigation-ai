"""Passive background fetch for the Explore feature.

Fetches a structured snapshot for 5 fixed destinations without any user query.
All 5 countries run in parallel; tool calls within each country run sequentially.
"""
from __future__ import annotations

import asyncio
import datetime
from typing import Any

from tools.activities import fetch_attractions, fetch_restaurants
from tools.concerts import search_concerts
from tools.destinations import get_average_weather, get_city_overview
from tools.flights import fetch_flights

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DESTINATIONS: list[dict[str, str]] = [
    {"country": "France",  "city": "Paris"},
    {"country": "Italy",   "city": "Rome"},
    {"country": "Greece",  "city": "Athens"},
    {"country": "Germany", "city": "Berlin"},
    {"country": "England", "city": "London"},
]

_ORIGIN = "TLV"

_SEASON_MAP: dict[int, str] = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}

# Hardcoded English names — strftime("%B") is locale-dependent and emits
# Hebrew month names on this machine, which search_concerts cannot parse.
_MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _current_season() -> str:
    return _SEASON_MAP[datetime.date.today().month]


def _concert_months() -> tuple[str, str]:
    """Return English month strings for today and the following month."""
    today = datetime.date.today()
    first_of_next = (today.replace(day=1) + datetime.timedelta(days=32)).replace(day=1)
    current = f"{_MONTH_NAMES[today.month - 1]} {today.year}"
    nxt = f"{_MONTH_NAMES[first_of_next.month - 1]} {first_of_next.year}"
    return current, nxt


def _safe_invoke(tool_fn: Any, args: dict) -> Any:
    """Call a LangChain tool, returning an error dict on failure."""
    try:
        return tool_fn.invoke(args)
    except Exception as exc:  # noqa: BLE001
        return {"message": str(exc)}


# ---------------------------------------------------------------------------
# Per-country fetch
# ---------------------------------------------------------------------------

def _fetch_country_snapshot(country: str, city: str) -> dict:
    """Run all tool calls for one country sequentially and return a structured snapshot."""
    season = _current_season()
    current_month, next_month = _concert_months()

    weather = _safe_invoke(get_average_weather, {"city": city, "season": season})

    concerts_current = _safe_invoke(search_concerts, {"city": city, "month": current_month})
    concerts_next = _safe_invoke(search_concerts, {"city": city, "month": next_month})
    first_concert    = _fetch_first_concert_for_city(city)   
    overview = _safe_invoke(get_city_overview, {"city": city})

    flights = _safe_invoke(fetch_flights, {"origin": _ORIGIN, "destination": city})

    attractions = _safe_invoke(fetch_attractions, {"city": city})
    restaurants = _safe_invoke(fetch_restaurants, {"city": city})

    return {
        "country": country,
        "city": city,
        "season": season,
        "weather": weather,
                "first_concert": first_concert,   

        "concerts": {
            current_month: concerts_current,
            next_month: concerts_next,
        },
        "overview": overview,
        "flights": flights,
        "attractions": attractions,
        "restaurants": restaurants,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def get_explore_snapshots() -> dict[str, dict]:
    """Fetch snapshots for all 5 destinations in parallel.

    Returns a dict keyed by country name, each value a structured snapshot:
    {
        "country": str,
        "city": str,
        "season": str,
        "weather": ...,        # get_average_weather result
        "concerts": {          # keyed by month string
            "<Month YYYY>": [...],
            "<Month YYYY>": [...],
        },
        "overview": ...,       # get_city_overview result
        "flights": [...],      # fetch_flights result
        "attractions": [...],  # fetch_attractions result — [{name, rating, ...}]
        "restaurants": [...],  # fetch_restaurants result — [{name, rating, ...}]
    }
    """
    loop = asyncio.get_event_loop()

    tasks = [
        loop.run_in_executor(None, _fetch_country_snapshot, d["country"], d["city"])
        for d in _DESTINATIONS
    ]
    results = await asyncio.gather(*tasks)

    return {snapshot["country"]: snapshot for snapshot in results}



def _fetch_first_concert_for_city(city: str) -> dict | None:
    """Fetch the first upcoming concert for a city.

    Prefers individual event pages (/concerts/ URL) over generic listing pages,
    so the formatter can display a real concert title rather than a page heading.
    """
    result = _safe_invoke(search_concerts, {"city": city})
    if not isinstance(result, list):
        return None
    # Prefer individual Songkick event pages
    for item in result:
        if isinstance(item, dict) and "message" not in item:
            if "/concerts/" in item.get("url", ""):
                return item
    # Fall back to first non-error result (listing page with parseable content)
    for item in result:
        if isinstance(item, dict) and "message" not in item:
            return item
    return None