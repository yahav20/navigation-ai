"""Tavily-powered special events search for the itinerary pipeline.

Restricted exclusively to timeout.com at the API-call level.
Inputs are derived from validated state fields — no raw user text ever reaches Tavily.
"""
from __future__ import annotations

import calendar
import os
import re
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Domain lock — enforced at the API call, not just a convention
# ---------------------------------------------------------------------------

_EVENTS_DOMAIN: list[str] = ["timeout.com"]

# Token-budget constants — keep LLM context manageable (avoids "lost in the middle")
_MAX_SNIPPET_CHARS = 800
_MAX_TOTAL_CHARS   = 8_000


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
        include_domains=_EVENTS_DOMAIN,   # API-level domain lock
        search_depth="advanced",           # richer snippets without full HTML
    )


def _is_timeout_url(url: str) -> bool:
    """Guard: only process URLs that actually belong to timeout.com."""
    try:
        host = urlparse(url).netloc.lower()
        return host == "timeout.com" or host.endswith(".timeout.com")
    except Exception:
        return False


def search_special_events(city: str, date_from: str, date_to: str) -> list[dict]:
    """Search timeout.com for events in *city* between *date_from* and *date_to*.

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
        year       = date_from[:4]
        month_name = calendar.month_name[month_num]
    except (ValueError, IndexError):
        return []

    # Query built from validated state fields only — no user message content
    query = f"events festivals things to do {city} {month_name} {year}"

    try:
        searcher = _get_tavily_search(max_results=10)
        response = searcher.invoke({"query": query})
    except Exception:  # noqa: BLE001
        return []

    raw_results: list[dict] = []
    if isinstance(response, dict):
        raw_results = response.get("results", [])
    elif isinstance(response, list):
        raw_results = response

    results: list[dict] = []
    total_chars = 0

    for r in raw_results:
        if not isinstance(r, dict):
            continue
        url = r.get("url", "")
        if not _is_timeout_url(url):
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
            "url":     url,
        })
        total_chars += len(content)

    return results
