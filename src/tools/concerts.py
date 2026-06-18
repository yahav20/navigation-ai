"""Tavily-powered concert and live event search for the advisor agent."""
from __future__ import annotations

import datetime
import os

from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Whitelisted domains
# ---------------------------------------------------------------------------

_ALL_CONCERT_DOMAINS = [
    "songkick.com",
    "ticketmaster.com",
    "ra.co",
]

_ARTIST_DOMAINS = ["songkick.com"]
_CITY_DOMAINS   = ["songkick.com"]


def _get_tavily_search(domains: list[str], max_results: int = 20):
    """TavilySearch — semantic search returning short snippets across domains."""
    try:
        from langchain_tavily import TavilySearch  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "langchain-tavily is required. Run: pip install langchain-tavily"
        ) from exc

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key or api_key.startswith("tvly-your"):
        raise ValueError("TAVILY_API_KEY is not set.")

    return TavilySearch(
        max_results=max_results,
        include_domains=domains,
        search_depth="advanced",
    )


def _get_tavily_extract():
    """TavilyExtract — fetches full page content for a specific URL via Tavily."""
    try:
        from langchain_tavily import TavilyExtract  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "langchain-tavily is required. Run: pip install langchain-tavily"
        ) from exc

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key or api_key.startswith("tvly-your"):
        raise ValueError("TAVILY_API_KEY is not set.")

    return TavilyExtract(extract_depth="advanced")


# ---------------------------------------------------------------------------
# City disambiguation
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

# ---------------------------------------------------------------------------
# Songkick metro-area dynamic lookup + TavilyExtract
# ---------------------------------------------------------------------------

_MONTH_NUMS = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
}
_MONTH_LAST_DAY = {
    "01": "31", "02": "28", "03": "31", "04": "30", "05": "31", "06": "30",
    "07": "31", "08": "31", "09": "30", "10": "31", "11": "30", "12": "31",
}


def _find_songkick_metro_base_url(city: str) -> str | None:
    """Look up the Songkick metro-area base URL for a city.

    Strategy:
    1. Try the Songkick filter_metro_area endpoint — returns ~20 popular cities fast.
    2. If the city isn't in that list, fall back to a targeted Tavily search.

    Returns e.g. 'https://www.songkick.com/metro-areas/24426-uk-london' or None.
    """
    import re  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415
    from urllib.parse import quote, urlparse, urlunparse  # noqa: PLC0415

    city_lower = city.lower().strip()
    city_str   = _DISAMBIGUATE.get(city_lower, city)
    city_slug  = city_lower.replace(" ", "-")   # "new york" → "new-york"

    # --- Tier 1: Songkick filter_metro_area endpoint (fast, ~20 popular cities) ---
    try:
        filter_url = (
            "https://www.songkick.com/session/filter_metro_area"
            f"?utf8=true&filters%5Bmetro_area_name%5D={quote(city_str)}&commit=Go"
        )
        req  = urllib.request.Request(filter_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as r:
            body = r.read().decode()

        for slug in re.findall(r"/metro-areas/([\w\-]+)", body):
            if city_slug in slug:
                return f"https://www.songkick.com/metro-areas/{slug}"
    except Exception:  # noqa: BLE001
        pass

    # --- Tier 2: Tavily search (works for any city not in the popular list) ---
    try:
        search   = _get_tavily_search(["songkick.com"], max_results=5)
        response = search.invoke({"query": f"Songkick {city_str} concerts upcoming"})
        results  = response.get("results", []) if isinstance(response, dict) else response
        for r in results:
            url = r.get("url", "")
            if "/metro-areas/" in url:
                p = urlparse(url)
                return urlunparse((p.scheme, p.netloc, p.path.rstrip("/"), "", "", ""))
    except Exception:  # noqa: BLE001
        pass

    return None


def _tavily_extract_metro(city: str, month_str: str, genre: str | None = None) -> dict | None:
    """Use TavilyExtract to fetch the full Songkick metro-area listing page.

    Finds the correct metro URL dynamically via a Tavily search — no hardcoded slug map.
    The snippet includes '_browse_url' so the formatter can surface a correct browse link.
    """
    metro_base = _find_songkick_metro_base_url(city)
    if not metro_base:
        return None

    parts = month_str.strip().split()
    if len(parts) != 2:
        return None
    month_num = _MONTH_NUMS.get(parts[0].lower())
    year      = parts[1]
    if not month_num or not year.isdigit():
        return None

    last_day    = _MONTH_LAST_DAY[month_num]
    extract_url = (
        f"{metro_base}"
        f"?filters%5BminDate%5D={month_num}%2F01%2F{year}"
        f"&filters%5BmaxDate%5D={month_num}%2F{last_day}%2F{year}"
    )
    browse_url = f"{metro_base}/{parts[0].lower()}-{year}"  # e.g. .../august-2026

    try:
        extractor = _get_tavily_extract()
        query     = f"concerts {genre + ' ' if genre else ''}{city} {month_str}"
        response  = extractor.invoke({"urls": [extract_url], "query": query})
        results   = response.get("results") if isinstance(response, dict) else []
        if results and results[0].get("raw_content"):
            content = results[0]["raw_content"]
            title   = f"Songkick {genre + ' ' if genre else ''}concerts {city} {month_str}"
            return {
                "title":       title,
                "content":     content[:20000],
                "url":         extract_url,
                "_browse_url": browse_url,   # forwarded to formatter for the browse link
            }
    except Exception:  # noqa: BLE001
        pass
    return None


# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------

def _build_query(
    city: str | None,
    artist: str | None,
    month: str | None,
    genre: str | None = None,
) -> str:
    parts: list[str] = []

    if genre:
        parts.append(genre)
    if artist:
        parts.append(artist)

    if city:
        city_str = _DISAMBIGUATE.get(city.lower().strip(), city)
        if artist:
            parts.append(f"concert {city_str}")
        else:
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

    At least one of city, artist, or month must be provided. Combine parameters:

    - city + month         → all concerts/shows in that city during that period
    - artist + month       → cities where the artist performs that month
    - artist + city        → dates when the artist performs in that city
    - genre + city + month → genre-filtered concerts (e.g. rock concerts in London)
    - artist alone         → all upcoming tour dates for the artist

    Data is sourced from Songkick via Tavily.
    """
    if not any([city, artist, month, genre]):
        return [{"message": "Please provide at least one of: city, artist, month, or genre."}]

    today = datetime.date.today().isoformat()
    results: list[dict] = []

    # For city+month queries: use TavilyExtract on the Songkick metro-area page
    # to get the full listing (not just snippets). Falls back to search if unknown city.
    if city and month and not artist:
        metro = _tavily_extract_metro(city, month, genre)
        if metro:
            results.append(metro)

    # TavilySearch supplements with individual event pages and artist-specific results
    domains     = _ARTIST_DOMAINS if artist else _CITY_DOMAINS
    query       = _build_query(city, artist, month, genre)
    max_results = 15 if results else 20  # fewer needed when we already have the full listing

    try:
        tavily   = _get_tavily_search(domains, max_results)
        response = tavily.invoke({"query": query, "start_date": today})
    except (ImportError, ValueError) as exc:
        if not results:
            return [{"message": str(exc)}]
        response = {}
    except Exception as exc:  # noqa: BLE001
        if not results:
            return [{"message": f"Concert search failed: {exc}"}]
        response = {}

    if isinstance(response, dict):
        raw = response.get("results", [])
    elif isinstance(response, list):
        raw = response
    else:
        raw = []

    results += [
        {"title": r.get("title", ""), "content": r.get("content", ""), "url": r.get("url", "")}
        for r in raw
        if isinstance(r, dict)
    ]

    return results or [{"message": "No concert results found for this search."}]
