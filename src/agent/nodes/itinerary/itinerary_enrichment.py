"""
ItineraryEnrichmentNode
=======================
Replaces the travel EnrichmentNode for the build_itinerary flow.
Handles everything in one place: metadata extraction, missing-field prompts,
preference collection, and (for with_travel_data mode) resolving the raw
hotel and flight records the executor needs for DayConfig.

Required fields (asks if missing):
  current_city      — origin / home city
  destination_city  — where to go
  trip_days         — how many days (defaults to 3 after one ask)
  total_budget      — optional; skip after one ask

Preferences collected (non-blocking, all optional):
  activity_pace       — relaxed / moderate / packed
  day_start_time      — 08:00 / 09:00 / 10:00
  day_end_time        — 20:00 / 21:00 / 22:00
  dietary_restrictions
  interests

with_travel_data resolution:
  When state["travel_plan"] is set (travel agent already ran), this node:
    1. Matches the curated hotel name back to the raw DB record → lat/lng
    2. Finds the cheapest outbound and return flights from state["flight_options"]
  Both are stored as itinerary_selected_hotel / itinerary_selected_outbound_flight /
  itinerary_selected_return_flight for the Executor to build DayConfig with.

Sets itinerary_enrichment_complete=True when passing through to the Planner.
"""
from __future__ import annotations

from typing import Literal, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from agent.state import AgentState
from tools.dependencies import data_provider

DEFAULT_TRIP_DAYS = 3


# ---------------------------------------------------------------------------
# Extraction schemas
# ---------------------------------------------------------------------------

class ItineraryMetadata(BaseModel):
    """Basic trip fields extracted from the user's message."""
    current_city:     Optional[str]   = Field(default=None, description="Origin / home city")
    destination_city: Optional[str]   = Field(default=None, description="Destination city")
    trip_days:        Optional[int]   = Field(default=None, description="Number of days")
    total_budget:     Optional[float] = Field(default=None, description="Total budget in USD")


class SchedulePreferences(BaseModel):
    """Schedule-specific preferences extracted from the user's message."""
    activity_pace: Optional[Literal["relaxed", "moderate", "packed"]] = Field(
        default=None,
        description="'relaxed'=2-3 activities/day, 'moderate'=4-5 (default), 'packed'=6+",
    )
    day_start_time: Optional[Literal["08:00", "09:00", "10:00"]] = Field(
        default=None,
        description="Preferred start time: '08:00'=early, '09:00'=normal, '10:00'=late",
    )
    day_end_time: Optional[Literal["20:00", "21:00", "22:00"]] = Field(
        default=None,
        description="Preferred end time: '20:00'=early evening, '21:00'=normal, '22:00'=late night",
    )
    dietary_restrictions: Optional[str] = Field(
        default=None,
        description="Food preferences (kosher, vegan, vegetarian, halal, etc.)",
    )
    interests: Optional[str] = Field(
        default=None,
        description="Category preferences e.g. 'history, food, outdoor, nightlife'",
    )


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class ItineraryEnrichmentNode:
    """
    Full enrichment gate for the itinerary flow. Runs instead of (not after)
    the travel EnrichmentNode.
    """

    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm
        self._meta_extractor = llm.with_structured_output(ItineraryMetadata)
        self._pref_extractor = llm.with_structured_output(SchedulePreferences)

    def __call__(self, state: AgentState) -> dict:
        asked   = set(state.get("enrichment_asked_fields") or [])
        updates: dict = {}

        # ── Step 1: Extract metadata from the latest user message ──────────
        meta_updates = self._extract_metadata(state)
        updates.update(meta_updates)

        # Merge extracted values with existing state for the remainder of the checks
        destination = updates.get("destination_city") or state.get("destination_city", "")
        origin      = updates.get("current_city")     or state.get("current_city", "")
        trip_days   = updates.get("trip_days")         or state.get("trip_days", 0)
        budget      = updates.get("total_budget")      or state.get("total_budget", 0)

        # ── Step 2: Ask for missing required fields ────────────────────────
        if not destination:
            return self._ask(
                "To build your schedule I need to know the destination city. "
                "Which city are you visiting?",
                asked | {"destination_city"},
                updates,
            )

        if not origin:
            if "current_city" in asked:
                pass  # user was asked but didn't answer — proceed without origin
            else:
                return self._ask(
                    f"Got it — {destination}! What city will you be travelling from?",
                    asked | {"current_city"},
                    updates,
                )

        if not trip_days:
            if "trip_days" in asked:
                updates["trip_days"] = DEFAULT_TRIP_DAYS
                trip_days = DEFAULT_TRIP_DAYS
            else:
                return self._ask(
                    f"How many days are you planning to spend in {destination}?",
                    asked | {"trip_days"},
                    updates,
                )

        if not budget and "total_budget" not in asked:
            return self._ask(
                f"What's your total budget for this trip to {destination}? "
                "(You can skip this if you prefer — just say 'no budget')",
                asked | {"total_budget"},
                updates,
            )

        # ── Step 3: Extract schedule preferences (non-blocking) ────────────
        pref_updates = self._extract_prefs(state, destination)
        if pref_updates:
            current_prefs = dict(updates.get("user_preferences") or state.get("user_preferences") or {})
            current_prefs.update(pref_updates)
            updates["user_preferences"] = current_prefs

        # ── Step 4: Detect mode and resolve hotel/flight if available ──────
        has_travel_data = bool(state.get("travel_plan"))
        updates["itinerary_mode"] = "with_travel_data" if has_travel_data else "standalone"

        if has_travel_data:
            resolved = self._resolve_travel_data(state, destination)
            updates.update(resolved)

        # ── Step 5: Pass through to Planner ───────────────────────────────
        updates["itinerary_enrichment_complete"] = True
        updates["enrichment_asked_fields"] = list(asked)
        return updates

    # ── Helpers ────────────────────────────────────────────────────────────

    def _ask(self, question: str, new_asked: set, pending_updates: dict) -> dict:
        return {
            **pending_updates,
            "messages":              [AIMessage(content=question, name="itinerary_enrichment")],
            "enrichment_asked_fields": list(new_asked),
        }

    def _extract_metadata(self, state: AgentState) -> dict:
        """Extract origin, destination, budget, trip_days from the latest user message."""
        messages = state.get("messages", [])
        last_user = next(
            (m for m in reversed(messages) if getattr(m, "type", "") == "human"),
            None,
        )
        if not last_user:
            return {}
        try:
            meta: ItineraryMetadata = self._meta_extractor.invoke([
                {
                    "role": "system",
                    "content": (
                        "Extract trip basics from the user's message. "
                        "Return null for anything not mentioned."
                    ),
                },
                {"role": "user", "content": last_user.content},
            ])
        except Exception:
            return {}
        result = {}
        if meta.current_city     and not state.get("current_city"):
            result["current_city"]     = meta.current_city
        if meta.destination_city and not state.get("destination_city"):
            result["destination_city"] = meta.destination_city
        if meta.trip_days        and not state.get("trip_days"):
            result["trip_days"]        = meta.trip_days
        if meta.total_budget     and not state.get("total_budget"):
            result["total_budget"]     = meta.total_budget
        return result

    def _extract_prefs(self, state: AgentState, destination: str) -> dict:
        """Extract schedule preferences from the latest user message."""
        messages = state.get("messages", [])
        last_user = next(
            (m for m in reversed(messages) if getattr(m, "type", "") == "human"),
            None,
        )
        if not last_user:
            return {}
        try:
            extracted: SchedulePreferences = self._pref_extractor.invoke([
                {
                    "role": "system",
                    "content": (
                        f"Extract schedule preferences for a trip to {destination}. "
                        "Return null for anything not mentioned. "
                        "Recommend day_start_time='09:00' and day_end_time='21:00' as defaults "
                        "if the user implies a normal pace."
                    ),
                },
                {"role": "user", "content": last_user.content},
            ])
        except Exception:
            return {}
        return {k: v for k, v in extracted.model_dump().items() if v is not None}

    def _resolve_travel_data(self, state: AgentState, destination: str) -> dict:
        """
        For with_travel_data mode: find the raw hotel record (with lat/lng) and the
        raw outbound/return flight records (with departure/arrival times) so the
        Executor can build an accurate DayConfig.
        """
        result: dict = {}

        travel_plan = state.get("travel_plan") or {}
        hotels_curated  = travel_plan.get("hotels", [])
        flights_curated = travel_plan.get("flights", [])
        flight_options  = state.get("flight_options") or []

        # ── Hotel ──────────────────────────────────────────────────────────
        selected_hotel_name = hotels_curated[0].get("name") if hotels_curated else None
        if selected_hotel_name and destination:
            try:
                raw_hotels = data_provider.fetch_hotels(destination)
                raw_hotels = [h for h in raw_hotels if isinstance(h, dict) and not h.get("message")]
                # Match by name (case-insensitive)
                matched = next(
                    (h for h in raw_hotels
                     if h.get("name", "").lower() == selected_hotel_name.lower()),
                    raw_hotels[0] if raw_hotels else None,
                )
                if matched:
                    result["itinerary_selected_hotel"] = matched
            except Exception:
                pass

        # ── Outbound flight ────────────────────────────────────────────────
        # Use the cheapest available flight from flight_options, or match by flight number
        valid_flights = [f for f in flight_options if isinstance(f, dict) and not f.get("message")]

        if valid_flights:
            outbound_label = flights_curated[0].get("label", "") if flights_curated else ""
            outbound = _match_flight(valid_flights, outbound_label) or valid_flights[0]
            result["itinerary_selected_outbound_flight"] = outbound

        # ── Return flight ──────────────────────────────────────────────────
        origin = state.get("current_city", "")
        if origin and destination:
            try:
                raw_returns = data_provider.fetch_flights(destination, origin)
                raw_returns = [f for f in raw_returns if isinstance(f, dict) and not f.get("message")]
                if raw_returns:
                    # Prefer cheapest
                    raw_returns.sort(key=lambda f: float(f.get("price", 9999)))
                    result["itinerary_selected_return_flight"] = raw_returns[0]
            except Exception:
                pass

        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _match_flight(flights: list[dict], label: str) -> Optional[dict]:
    """Try to match a flight from flight_options by flight_number or label substring."""
    if not label:
        return None
    for f in flights:
        fn = str(f.get("flight_number", "")).strip()
        if fn and fn in label:
            return f
    return None
