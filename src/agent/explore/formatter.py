"""Card-style formatter for the Explore feature.

Converts the raw snapshot dict (from snapshot.get_explore_snapshots) into
5 structured destination cards. Deliberately separate from the conversational
advisor_formatter — no LLM call, no prompt injection surface.
"""
from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------------------
# Concert parsing helpers
# ---------------------------------------------------------------------------

# Matches common date formats found on Songkick pages:
#   "Fri 4 Jul 2026", "Sun, 6 Jul 2026", "Jul 4, 2026", "4 Jul 2026"
_DATE_RE = re.compile(
    r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)[,\s]+\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}"
    r"|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}"
    r"|\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}",
    re.IGNORECASE,
)

# Standalone genre labels that Songkick lists below the artist name
_GENRE_LABELS = frozenset([
    "electronic", "pop", "rock", "jazz", "metal", "indie", "alternative",
    "funk", "disco", "classical", "hip-hop", "r&b", "soul", "blues",
    "country", "reggae", "punk", "folk", "latin", "dance", "rave",
])

# UI / navigation lines to skip
_UI_FRAGMENTS = (
    "concerts", "festival", "tickets", "filter", "buy ticket",
    "sign in", "log in", "more events", "on tour", "songkick",
    "tour dates", "upcoming events",
)


def _is_artist_line(text: str) -> bool:
    lower = text.lower().strip()
    if len(text) < 2:
        return False
    if lower in _GENRE_LABELS:
        return False
    if any(frag in lower for frag in _UI_FRAGMENTS):
        return False
    return text[0].isupper()


def _parse_concerts_from_content(content: str, limit: int = 8) -> list[str]:
    """Extract 'Artist — Date' strings from a raw Songkick listing page."""
    results: list[str] = []
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]

    for i, line in enumerate(lines):
        if not _DATE_RE.search(line):
            continue
        # Walk up to 4 lines backwards to find the artist name
        for j in range(i - 1, max(i - 5, -1), -1):
            candidate = lines[j]
            if _is_artist_line(candidate):
                results.append(f"{candidate} — {_DATE_RE.search(line).group()}")  # type: ignore[union-attr]
                break
        if len(results) >= limit:
            break

    return results


def _clean_title(title: str) -> str | None:
    """Strip Songkick boilerplate from a page/event title."""
    for suffix in (" | Songkick", " Tickets | Songkick", " - Songkick", " Tickets"):
        if title.lower().endswith(suffix.lower()):
            title = title[: -len(suffix)]
    # Skip generic listing-page titles
    if any(k in title for k in ("Concerts, Festivals", "Tour Dates", "& Tour Dates")):
        return None
    if title.lower().startswith("songkick"):
        return None
    return title.strip() or None


# ---------------------------------------------------------------------------
# Weather / flight / places helpers
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
    """Collect real event names across both concert month buckets."""
    events: list[str] = []
    for month_label, items in concerts.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict) or "message" in item:
                continue
            url = item.get("url", "")
            content = item.get("content", "")

            if "metro-areas" in url or item.get("_browse_url"):
                # Listing page — parse individual concerts from page content
                for name in _parse_concerts_from_content(content, limit=5):
                    events.append(f"{name} ({month_label})")
            else:
                # Individual concert page — clean the page title
                title = item.get("title") or item.get("name") or item.get("event") or ""
                clean = _clean_title(title)
                if clean:
                    events.append(f"{clean} ({month_label})")

    return events


def _extract_first_concert(concert: Any) -> str | None:
    """Return a display string for the first/featured concert, or None."""
    if not isinstance(concert, dict) or "message" in concert:
        return None
    url = concert.get("url", "")
    title = concert.get("title") or concert.get("name") or concert.get("event") or ""
    content = concert.get("content", "")

    # Individual event page → clean the title directly
    if "/concerts/" in url:
        return _clean_title(title)

    # Listing page → parse the first concert from the page content
    if content:
        parsed = _parse_concerts_from_content(content, limit=1)
        if parsed:
            return parsed[0]

    # Last resort: cleaned title
    return _clean_title(title)


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
    first = flights[0]
    if isinstance(first, dict):
        for key in ("price", "cost", "fare", "amount"):
            if key in first:
                return f"From ~{first[key]}"
    return "Flight cost unavailable"


def _extract_top_attractions(attractions: Any) -> list[str]:
    """Return up to 5 attraction names from fetch_attractions result."""
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
    """Return up to 5 restaurant names from fetch_restaurants result."""
    if not isinstance(restaurants, list):
        return []
    names: list[str] = []
    for item in restaurants:
        if isinstance(item, dict) and "name" in item and "message" not in item:
            names.append(item["name"])
        if len(names) >= 5:
            break
    return names


def _extract_top_hotels(hotels: Any) -> list[dict]:
    """Return up to 3 hotels with name, stars, and price from fetch_hotels result."""
    if not isinstance(hotels, list):
        return []
    result: list[dict] = []
    for item in hotels:
        if not isinstance(item, dict) or "message" in item or "name" not in item:
            continue
        result.append({
            "name": item["name"],
            "stars": item.get("stars"),
            "price": item.get("price_per_night"),
        })
        if len(result) >= 3:
            break
    return result


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
        "first_concert": _extract_first_concert(snapshot.get("first_concert")),
        "flight_cost": _extract_flight_cost(snapshot.get("flights")),
        "top_attractions": _extract_top_attractions(snapshot.get("attractions")),
        "top_restaurants": _extract_top_restaurants(snapshot.get("restaurants")),
        "top_hotels": _extract_top_hotels(snapshot.get("hotels")),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def format_explore_cards(snapshots: dict[str, dict]) -> list[dict]:
    """Format all 5 snapshot dicts into card-ready structures."""
    return [_build_card(snapshot) for snapshot in snapshots.values()]
