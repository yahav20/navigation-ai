"""Tavily-powered concert and live event search for the advisor agent."""
from __future__ import annotations

import datetime
import os

from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Whitelisted domains — restricts Tavily to trusted concert listing sites only.
# This prevents indirect prompt injection from arbitrary web pages and ensures
# consistent, structured event data.
# ---------------------------------------------------------------------------

_ALL_CONCERT_DOMAINS = [
    "songkick.com",
    "ticketmaster.com",
    "livenation.com",
    "ra.co",
]

# Artist queries → artist tour/event pages are most useful
_ARTIST_DOMAINS = ["songkick.com", "ticketmaster.com", "livenation.com"]
# City/month queries → event listing and festival pages are most useful
_CITY_DOMAINS   = ["songkick.com", "ticketmaster.com", "livenation.com"]


def _get_tavily(domains: list[str], max_results: int = 20):
    """Lazily import and instantiate TavilySearch restricted to the given domains."""
    try:
        from langchain_tavily import TavilySearch  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "langchain-tavily is required for concert search. "
            "Run: pip install langchain-tavily"
        ) from exc

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key or api_key.startswith("tvly-your"):
        raise ValueError(
            "TAVILY_API_KEY is not set. "
            "Get a free key at https://app.tavily.com and add it to your .env file."
        )

    return TavilySearch(
        max_results=max_results,
        include_domains=domains,
        search_depth="advanced",
    )


# ---------------------------------------------------------------------------
# City disambiguation — prevents e.g. "London" matching London, Ontario
# ---------------------------------------------------------------------------

_DISAMBIGUATE: dict[str, str] = {
    "london":     "London UK",
    "paris":      "Paris France",
    "rome":       "Rome Italy",
    "florence":   "Florence Italy",
    "cambridge":  "Cambridge UK",
    "oxford":     "Oxford UK",
    "manchester": "Manchester UK",
    "birmingham": "Birmingham UK",
}


def _build_query(
    city: str | None,
    artist: str | None,
    month: str | None,
    genre: str | None = None,
) -> str:
    """Build a Tavily query that surfaces comprehensive event listing pages.

    Design goals:
    - City+month: target broad listing pages (Songkick metro-area, Ticketmaster
      city search) that contain many events, not individual event pages.
    - Artist: target the artist's own tour/event page.
    - Genre: prepend the genre so search results are filtered to that style.
    """
    parts: list[str] = []

    if genre:
        parts.append(genre)

    if artist:
        parts.append(artist)

    if city:
        city_str = _DISAMBIGUATE.get(city.lower().strip(), city)
        if artist:
            # Artist + city: find that artist's event page for the city
            parts.append(f"concert {city_str}")
        else:
            # City only: "concerts in X" matches Songkick metro-area page titles
            # "festivals" ensures /festivals/ pages surface alongside /concerts/ pages
            parts.append(f"concerts in {city_str} festivals")
    else:
        parts.append("concert tour dates")

    parts.append(month if month else str(datetime.date.today().year))
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Main tool
# ---------------------------------------------------------------------------

@tool
def search_concerts(
    city: str | None = None,
    artist: str | None = None,
    month: str | None = None,
    genre: str | None = None,
) -> list[dict]:
    """Search for upcoming concerts and live shows using real-time web data.

    At least one of city, artist, or month must be provided. Combine parameters
    to narrow the search:

    - city + month         → all concerts/shows in that city during that period
    - artist + month       → cities where the artist performs that month
    - artist + city        → dates when the artist performs in that city
    - genre + city + month → genre-filtered concerts (e.g. rock concerts in London)
    - artist alone         → all upcoming tour dates for the artist

    Data is sourced from: Songkick, Ticketmaster, Live Nation, Resident Advisor.

    Returns a list of results, each with a title, content snippet, and source URL.
    """
    if not any([city, artist, month, genre]):
        return [{"message": "Please provide at least one of: city, artist, month, or genre."}]

    domains    = _ARTIST_DOMAINS if artist else _CITY_DOMAINS
    query      = _build_query(city, artist, month, genre)
    today      = datetime.date.today().isoformat()

    # City+month listings benefit from more results since each page typically
    # covers only 1-3 events; artist searches need fewer pages.
    max_results = 20 if (city and month and not artist) else 15

    try:
        tavily   = _get_tavily(domains, max_results)
        response = tavily.invoke({"query": query, "start_date": today})
    except (ImportError, ValueError) as exc:
        return [{"message": str(exc)}]
    except Exception as exc:  # noqa: BLE001
        return [{"message": f"Concert search failed: {exc}"}]

    if isinstance(response, dict):
        raw = response.get("results", [])
    elif isinstance(response, list):
        raw = response
    else:
        raw = []

    if not raw:
        return [{"message": "No concert results found for this search."}]

    return [
        {"title": r.get("title", ""), "content": r.get("content", ""), "url": r.get("url", "")}
        for r in raw
        if isinstance(r, dict)
    ]
