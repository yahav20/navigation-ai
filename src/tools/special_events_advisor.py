"""Special events search for the advisor agent.

Delegates to the shared dual-query core in ``tools.special_events`` so the advisor,
the itinerary builder, and the travel agent all pull from the SAME Tavily search
(same dual query, same domains, same snippet budget) — guaranteeing that whichever
pipeline searches for special events sees the same source articles.

Inputs come from the advisor planner — never raw user text.
"""
from __future__ import annotations

import calendar
import datetime

from langchain_core.tools import tool


def _month_to_range(month: str | None) -> tuple[str, str]:
    """Convert a "Month YYYY" string to (date_from, date_to) covering that whole month.

    Falls back to a 30-day window starting today when *month* is missing or unparseable,
    so the shared search (which requires YYYY-MM-DD) always receives valid dates.
    """
    if month:
        parts = month.strip().split()
        try:
            month_name = parts[0]
            year       = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else datetime.date.today().year
            month_num  = list(calendar.month_name).index(month_name.capitalize())  # 1–12
            if month_num:
                last_day = calendar.monthrange(year, month_num)[1]
                return f"{year}-{month_num:02d}-01", f"{year}-{month_num:02d}-{last_day:02d}"
        except (ValueError, IndexError):
            pass

    today = datetime.date.today()
    return today.isoformat(), (today + datetime.timedelta(days=30)).isoformat()


@tool
def search_special_events(city: str, month: str | None = None) -> list[dict]:
    """Search for special events, festivals, and seasonal happenings at a destination.

    Parameters
    ----------
    city  : destination city name
    month : "Month YYYY" format, e.g. "December 2026". Omit to search the next 30 days.

    Returns a list of raw {title, content, url} snippets for LLM extraction.
    Sources: timeout.com, theculturetrip.com, lonelyplanet.com
    """
    date_from, date_to = _month_to_range(month)

    # Single source of truth: the same dual-query, domain-locked search the itinerary
    # and travel pipelines use, so all three see identical source articles.
    from tools.special_events import search_special_events as _core_search  # noqa: PLC0415

    try:
        results = _core_search(city, date_from, date_to)
    except Exception as exc:  # noqa: BLE001
        return [{"message": f"Special events search failed: {exc}"}]

    return results or [{"message": "No special events results found for this search."}]
