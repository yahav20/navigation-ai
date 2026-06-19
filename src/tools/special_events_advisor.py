"""Tavily-powered special events search for the advisor agent.

Searches timeout.com, theculturetrip.com, and lonelyplanet.com for events,
festivals, and seasonal happenings at a destination.

Inputs come from the advisor planner — never raw user text.
"""
from __future__ import annotations

import datetime
import os

from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Domain list — travel editorial sources for events/festivals
# ---------------------------------------------------------------------------

_EVENTS_DOMAINS = ["timeout.com", "theculturetrip.com", "lonelyplanet.com"]

_MAX_SNIPPET_CHARS = 1_000
_MAX_TOTAL_CHARS   = 12_000


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
        include_domains=_EVENTS_DOMAINS,
        search_depth="advanced",
    )


@tool
def search_special_events(city: str, month: str | None = None) -> list[dict]:
    """Search for special events, festivals, and seasonal happenings at a destination.

    Parameters
    ----------
    city  : destination city name
    month : "Month YYYY" format, e.g. "December 2026". Omit to search for current year.

    Returns a list of raw {title, content, url} snippets for LLM extraction.
    Sources: timeout.com, theculturetrip.com, lonelyplanet.com
    """
    period = month or str(datetime.date.today().year)
    query  = f"events festivals things to do {city} {period}"

    try:
        searcher = _get_tavily_search(max_results=10)
        response = searcher.invoke({"query": query})
    except (ImportError, ValueError) as exc:
        return [{"message": str(exc)}]
    except Exception as exc:  # noqa: BLE001
        return [{"message": f"Special events search failed: {exc}"}]

    raw: list[dict] = []
    if isinstance(response, dict):
        raw = response.get("results", [])
    elif isinstance(response, list):
        raw = response

    results: list[dict] = []
    total_chars = 0

    for r in raw:
        if not isinstance(r, dict):
            continue

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
            "url":     str(r.get("url", "")),
        })
        total_chars += len(content)

    return results or [{"message": "No special events results found for this search."}]
