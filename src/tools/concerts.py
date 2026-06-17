"""Tavily-powered concert and live event search for the advisor agent."""
from __future__ import annotations

import os

from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Whitelisted domains — restricts Tavily to trusted concert listing sites only.
# This prevents indirect prompt injection from arbitrary web pages and ensures
# consistent, structured event data.
# ---------------------------------------------------------------------------

_CONCERT_DOMAINS = [
    "songkick.com",      # city/date listings, best for "concerts in X in August"
    "bandsintown.com",   # artist tour pages, best for "where is [artist] in [month]"
    "ticketmaster.com",  # official tickets, global venue coverage
    "livenation.com",    # major promoter, strong European coverage
    "ra.co",             # Resident Advisor — essential for electronic/DJ acts
]


def _get_tavily():
    """Lazily import and instantiate TavilySearch so missing API keys fail at call time."""
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
        max_results=8,
        include_domains=_CONCERT_DOMAINS,
    )


def _build_query(city: str | None, artist: str | None, month: str | None) -> str:
    """Build a focused, minimal Tavily query from the available parameters."""
    parts: list[str] = []
    if artist:
        parts.append(artist)
    if city and artist:
        parts.append(f"concert {city}")
    elif city:
        parts.append(f"concerts {city}")
    else:
        parts.append("concert tour dates")
    if month:
        parts.append(month)
    else:
        parts.append("2026")
    return " ".join(parts)


@tool
def search_concerts(
    city: str | None = None,
    artist: str | None = None,
    month: str | None = None,
) -> list[dict]:
    """Search for upcoming concerts and live shows using real-time web data.

    At least one of city, artist, or month must be provided. Combine parameters
    to narrow the search:

    - city + month       → all concerts/shows in that city during that period
    - artist + month     → cities where the artist performs that month (find where to travel)
    - artist + city      → dates when the artist performs in that city (find when to travel)
    - artist alone       → all upcoming tour dates for the artist

    Data is sourced exclusively from: Songkick, Bandsintown, Ticketmaster,
    Live Nation, and Resident Advisor (RA).

    Returns a list of results, each with a title, content snippet, and source URL.
    """
    if not any([city, artist, month]):
        return [{"message": "Please provide at least one of: city, artist, or month."}]

    query = _build_query(city, artist, month)
    try:
        tavily = _get_tavily()
        # TavilySearch._run returns Dict{"query", "results": [...], "response_time"}
        response = tavily.invoke({"query": query})
    except (ImportError, ValueError) as exc:
        return [{"message": str(exc)}]
    except Exception as exc:  # noqa: BLE001
        return [{"message": f"Concert search failed: {exc}"}]

    # Extract the nested results list from the response dict
    if isinstance(response, dict):
        raw = response.get("results", [])
    elif isinstance(response, list):
        raw = response
    else:
        raw = []

    if not raw:
        return [{"message": "No concert results found for this search."}]

    return [
        {
            "title": r.get("title", ""),
            "content": r.get("content", ""),
            "url": r.get("url", ""),
        }
        for r in raw
        if isinstance(r, dict)
    ]
