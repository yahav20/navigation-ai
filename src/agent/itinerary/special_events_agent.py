"""SpecialEventsSubAgent — fetches and parses date-specific events for the destination.

Designed to run in a background thread launched by ItineraryPlannerNode so it
executes in parallel with the (fast, synchronous) plan-generation step, keeping
the total latency cost close to zero.
"""
from __future__ import annotations

import calendar
import datetime
import re

from agent.itinerary.event_extractor import extract_special_events
from security import validate_city
from tools.special_events import search_special_events


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
        # Mode decides the search window:
        #   "with_travel_data" (itinerary built around booked flights) → the specific
        #       flight dates, so the day-by-day schedule only weaves in events on those days.
        #   everything else (travel overview, standalone itinerary, advisor) → the whole
        #       month, giving a broad view of what's on.
        mode = state.get("itinerary_mode", "")

        try:
            city      = validate_city(destination)
            base_date = datetime.date.fromisoformat(self._parse_date(trip_start))
            if mode == "with_travel_data":
                date_from = base_date.isoformat()
                date_to   = (base_date + datetime.timedelta(days=trip_days - 1)).isoformat()
            else:
                last_day  = calendar.monthrange(base_date.year, base_date.month)[1]
                date_from = base_date.replace(day=1).isoformat()
                date_to   = base_date.replace(day=last_day).isoformat()
        except (ValueError, TypeError):
            return []

        raw    = search_special_events(city, date_from, date_to)
        events = extract_special_events(self._llm, raw, city, date_from, date_to)
        return [e.model_dump() for e in events]

    @staticmethod
    def _parse_date(trip_start: str) -> str:
        """Normalise trip_start to a YYYY-MM-DD date.

        Handles:
          - "YYYY-MM-DD"  → returned as-is
          - "YYYY-MM"     → first day of that month
          - ISO datetime  → date portion as-is (no timezone shift — event times stay in
                            the destination's local time, exactly as the sources report them)
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

        # ISO datetime or other — use the date portion as-is.
        candidate = s[:10]
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", candidate):
            return candidate

        raise ValueError(f"Unrecognised trip_start format: {trip_start!r}")
