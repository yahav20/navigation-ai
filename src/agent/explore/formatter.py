"""Card-style formatter for the Explore feature.

Converts the raw snapshot dict (from snapshot.get_explore_snapshots) into
5 structured destination cards. Deliberately separate from the conversational
advisor_formatter — no LLM call, no prompt injection surface.
"""
from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_weather_line(weather: Any) -> str:
    if isinstance(weather, dict) and "message" not in weather:
        season = weather.get("season", "")
        avg_low = weather.get("avg_low_c") or weather.get("avg_low")
        avg_high = weather.get("avg_high_c") or weather.get("avg_high")
        desc = weather.get("description") or weather.get("summary") or ""
        parts = []
        if season:
            parts.append(season)
        if avg_low is not None and avg_high is not None:
            parts.append(f"{avg_low}–{avg_high}°C")
        elif avg_high is not None:
            parts.append(f"~{avg_high}°C")
        if desc:
            parts.append(desc)
        return ", ".join(parts) if parts else "Weather data unavailable"
    return "Weather data unavailable"


def _extract_events(concerts: dict[str, Any]) -> list[str]:
    """Collect event titles across both concert month buckets."""
    events: list[str] = []
    for month_label, items in concerts.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                title = item.get("title") or item.get("name") or item.get("event")
                if title and {"message"} - item.keys():  # skip error dicts
                    events.append(f"{title} ({month_label})")
    return events


def _extract_flight_cost(flights: Any) -> str:
    if not isinstance(flights, list) or not flights:
        return "Flight cost unavailable"
    cheapest = None
    for f in flights:
        if not isinstance(f, dict):
            continue
        price = f.get("price") or f.get("cost") or f.get("value")
        try:
            price = float(price)
        except (TypeError, ValueError):
            continue
        if cheapest is None or price < cheapest:
            cheapest = price
    if cheapest is not None:
        return f"From ~${cheapest:.0f}"
    # Fallback: return the first flight's string representation
    first = flights[0]
    if isinstance(first, dict):
        for key in ("price", "cost", "fare", "amount"):
            if key in first:
                return f"From ~{first[key]}"
    return "Flight cost unavailable"


def _extract_top_attractions(attractions: Any) -> list[str]:
    """Return up to 5 attraction names from fetch_attractions result.

    Each item is a dict with a 'name' key (Google Maps normalized schema).
    """
    if not isinstance(attractions, list):
        return []
    names: list[str] = []
    for item in attractions:
        if isinstance(item, dict) and "name" in item and "message" not in item:
            names.append(item["name"])
        if len(names) >= 5:
            break
    return names


def _extract_top_restaurants(restaurants: Any) -> list[str]:
    """Return up to 5 restaurant names from fetch_restaurants result.

    Each item is a dict with a 'name' key (Google Maps normalized schema).
    """
    if not isinstance(restaurants, list):
        return []
    names: list[str] = []
    for item in restaurants:
        if isinstance(item, dict) and "name" in item and "message" not in item:
            names.append(item["name"])
        if len(names) >= 5:
            break
    return names


# ---------------------------------------------------------------------------
# Card builder
# ---------------------------------------------------------------------------

def _build_card(snapshot: dict) -> dict:
    """Convert one country snapshot into a structured display card."""
    return {
        "country": snapshot["country"],
        "city": snapshot["city"],
        "season": snapshot.get("season", ""),
        "weather": _extract_weather_line(snapshot.get("weather")),
        "events": _extract_events(snapshot.get("concerts", {})),
        "flight_cost": _extract_flight_cost(snapshot.get("flights")),
        "top_attractions": _extract_top_attractions(snapshot.get("attractions")),
        "top_restaurants": _extract_top_restaurants(snapshot.get("restaurants")),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def format_explore_cards(snapshots: dict[str, dict]) -> list[dict]:
    """Format all 5 snapshot dicts into card-ready structures.

    Args:
        snapshots: Output of get_explore_snapshots() — dict keyed by country name.

    Returns:
        List of card dicts, one per country, in the same order as the input:
        [
            {
                "country": str,
                "city": str,
                "season": str,
                "weather": str,           # e.g. "Summer, 18–28°C, warm and sunny"
                "events": [str, ...],     # event titles with month labels
                "flight_cost": str,       # e.g. "From ~$210"
                "top_attractions": [str, ...],  # up to 5 attraction names
                "top_restaurants": [str, ...],  # up to 5 restaurant/dining names
            },
            ...
        ]
    """
    return [_build_card(snapshot) for snapshot in snapshots.values()]
