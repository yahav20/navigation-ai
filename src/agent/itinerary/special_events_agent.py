"""SpecialEventsSubAgent — fetches and parses date-specific events for the destination.

Designed to run in a background thread launched by ItineraryPlannerNode so it
executes in parallel with the (fast, synchronous) plan-generation step, keeping
the total latency cost close to zero.
"""
from __future__ import annotations

import datetime
import re

from agent.itinerary.event_extractor import extract_special_events
from security import validate_city
from tools.special_events import search_special_events

# City → approximate UTC offset in hours (ignores DST — good enough for date-boundary use).
# Used only when trip_start contains a UTC timestamp (rare edge case).
_CITY_UTC_OFFSET: dict[str, int] = {
    "bangkok":     7,
    "tokyo":       9,
    "osaka":       9,
    "singapore":   8,
    "bali":        8,
    "dubai":       4,
    "new york":   -5,
    "los angeles":-8,
    "london":      0,
    "paris":       1,
    "berlin":      1,
    "munich":      1,
    "prague":      1,
    "amsterdam":   1,
    "rome":        1,
    "barcelona":   1,
    "madrid":      1,
    "istanbul":    3,
    "tel aviv":    2,
    "jerusalem":   2,
    "sydney":     10,
    "melbourne":  10,
}


class SpecialEventsSubAgent:
    """Fetch and parse special events for the itinerary destination.

    Called in a ThreadPoolExecutor from ItineraryPlannerNode so the I/O-bound
    Tavily request runs in parallel with synchronous plan generation.

    Returns a list of event dicts (serialised SpecialEventRaw) or [] on any error.
    """

    def __init__(self, llm) -> None:
        self._llm = llm

    def __call__(self, state: dict) -> list[dict]:
        destination = state.get("destination_city", "")
        trip_start  = state.get("trip_start", "")
        trip_days   = int(state.get("trip_days") or 3)

        try:
            city      = validate_city(destination)
            date_from = self._parse_date(trip_start, destination)
            date_to   = (
                datetime.date.fromisoformat(date_from)
                + datetime.timedelta(days=trip_days - 1)
            ).isoformat()
        except (ValueError, TypeError):
            return []

        raw    = search_special_events(city, date_from, date_to)
        events = extract_special_events(self._llm, raw, city, date_from, date_to)
        return [e.model_dump() for e in events]

    @staticmethod
    def _parse_date(trip_start: str, destination: str = "") -> str:
        """Normalise trip_start to a local YYYY-MM-DD date at the destination.

        Handles:
          - "YYYY-MM-DD"  → returned as-is
          - "YYYY-MM"     → first day of that month
          - ISO datetime with Z/offset → converted to destination local date
        """
        s = (trip_start or "").strip()
        if not s:
            raise ValueError("trip_start is empty")

        # Pure date
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            return s

        # Month only
        if re.fullmatch(r"\d{4}-\d{2}", s):
            return s + "-01"

        # ISO datetime — convert to local destination date to avoid midnight off-by-one.
        # trip_start is almost always a plain date from MetadataNode; this branch handles
        # the rare case where flight API data supplies a full UTC timestamp.
        if "T" in s or "Z" in s:
            try:
                s_clean = s.replace("Z", "+00:00")
                dt_utc  = datetime.datetime.fromisoformat(s_clean)
                # Apply UTC offset for the destination city (avoids external tz libraries)
                offset_h = _CITY_UTC_OFFSET.get(destination.lower().strip(), 0)
                dt_local = dt_utc + datetime.timedelta(hours=offset_h)
                return dt_local.date().isoformat()
            except Exception:
                # Last-resort: truncate to date portion (may be off by ±1 day near midnight)
                return s[:10]

        # Last-resort truncation
        candidate = s[:10]
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", candidate):
            return candidate

        raise ValueError(f"Unrecognised trip_start format: {trip_start!r}")
