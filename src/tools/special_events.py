"""Tavily-powered special events search for the itinerary pipeline.

Restricted to trusted travel editorial domains at the API level.
Inputs are derived from validated state fields — no raw user text ever reaches Tavily.
"""
from __future__ import annotations

import calendar
import os
import re
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Domain lock — enforced at the API call, not just a convention
# Matches the domains used by the advisor's search_special_events tool so
# both pipelines pull from the same trusted editorial sources.
# ---------------------------------------------------------------------------

_EVENTS_DOMAINS: list[str] = ["timeout.com", "theculturetrip.com", "lonelyplanet.com"]

# Token-budget constants — keep LLM context manageable (avoids "lost in the middle")
_MAX_SNIPPET_CHARS = 800
_MAX_TOTAL_CHARS   = 8_000

# Months with well-known seasonal events get a second targeted query alongside the
# generic one. This prevents Christmas markets (and similar recurring events) from
# being silently dropped when the generic query returns unrelated articles.
_SEASONAL_QUERIES: dict[int, str] = {
    2:  "carnival mardi gras winter festival",
    6:  "summer festival open air",
    10: "Oktoberfest harvest festival",
    11: "Christmas market holiday winter",
    12: "Christmas market holiday winter",
}


def _get_tavily_search(max_results: int = 10):
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
        include_domains=_EVENTS_DOMAINS,   # API-level domain lock
        search_depth="advanced",            # richer snippets without full HTML
    )


def _is_trusted_url(url: str) -> bool:
    """Guard: only process URLs from our allowed editorial domains."""
    try:
        host = urlparse(url).netloc.lower()
        return any(
            host == d or host.endswith("." + d)
            for d in _EVENTS_DOMAINS
        )
    except Exception:
        return False


def search_special_events(city: str, date_from: str, date_to: str) -> list[dict]:
    """Search travel editorial sites for events in *city* during the month of *date_from*.

    Parameters
    ----------
    city      : already validated via validate_city() by the caller
    date_from : YYYY-MM-DD (derived from state, regex-checked by caller)
    date_to   : YYYY-MM-DD (derived from state, regex-checked by caller)

    Returns a list of ``{title, content, url}`` dicts with content truncated
    to ``_MAX_SNIPPET_CHARS`` each and total capped at ``_MAX_TOTAL_CHARS``.
    Returns ``[]`` on any error (never raises).
    """
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_from):
        return []
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_to):
        return []

    try:
        month_num  = int(date_from[5:7])
        month_name = calendar.month_name[month_num]
    except (ValueError, IndexError):
        return []

    # Year intentionally omitted: articles describing annual events (Christmas markets,
    # summer festivals) are written each year. Including the year would restrict Tavily
    # to articles that don't exist yet for future travel dates, returning empty results.
    primary_query = f"events festivals things to do {city} {month_name}"

    seasonal_keyword = _SEASONAL_QUERIES.get(month_num)
    secondary_query = (
        f"{city} {seasonal_keyword} {month_name}"
        if seasonal_keyword
        else f"what's on {city} {month_name}"
    )
    queries = [primary_query, secondary_query]

    try:
        searcher = _get_tavily_search(max_results=10)
    except Exception:  # noqa: BLE001
        return []

    # Collect raw results from all queries, deduplicated by URL.
    seen_urls: set[str] = set()
    combined_raw: list[dict] = []
    for q in queries:
        try:
            response = searcher.invoke({"query": q})
        except Exception:  # noqa: BLE001
            continue
        batch: list[dict] = []
        if isinstance(response, dict):
            batch = response.get("results", [])
        elif isinstance(response, list):
            batch = response
        for r in batch:
            if not isinstance(r, dict):
                continue
            url = r.get("url", "")
            if not _is_trusted_url(url) or url in seen_urls:
                continue
            seen_urls.add(url)
            combined_raw.append(r)

    results: list[dict] = []
    total_chars = 0

    for r in combined_raw:
        url = r.get("url", "")
        content = str(r.get("content") or r.get("snippet") or "")
        content = content[:_MAX_SNIPPET_CHARS]

        remaining = _MAX_TOTAL_CHARS - total_chars
        if remaining <= 0:
            break
        if len(content) > remaining:
            content = content[:remaining]

        results.append({
            "title":   str(r.get("title", "")),
            "content": content,
            "url":     url,
        })
        total_chars += len(content)

    return results
