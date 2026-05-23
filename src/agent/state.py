"""Typed state shared across nodes of the travel-agent graph."""
from typing import Annotated, NotRequired, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # ── Core ────────────────────────────────────────────────────────────
    messages: Annotated[list, add_messages]
    summary: str                    # rolling conversation summary
    intent: str                     # classified intent from RouterNode

    # ── Trip parameters ─────────────────────────────────────────────────
    current_city: str
    destination_city: str
    total_budget: float
    trip_days: int                  # defaults to 3
    user_preferences: dict          # kosher, dietary, accessibility, etc.

    # ── Enrichment / metadata flow ──────────────────────────────────────
    enrichment_complete: bool
    is_adjustment: bool             # True when user modified trip parameters
    enrichment_asked_fields: list   # fields already asked to avoid re-asking
    budget_optional: bool           # True when user skipped budget

    # ── Flight search results ────────────────────────────────────────────
    flight_options: NotRequired[list[dict]]   # outbound flights from FlightSearchNode
    has_flights: NotRequired[bool]

    # ── Data bundle (built once by FlightSearchNode, reused downstream) ──
    itinerary_data_bundle: NotRequired[dict]
    # Contains: flights, return_flights, hotels, activities, weather, best_time

    # ── Standard travel plan (TravelAgentNode output) ───────────────────
    travel_plan: NotRequired[dict]

    # ── Itinerary sub-graph ──────────────────────────────────────────────
    build_itinerary: NotRequired[bool]        # flag set by RouterNode
    itinerary_plan: NotRequired[dict]         # final enriched plan
    itinerary_feasible: NotRequired[bool]
    itinerary_fallback_reason: NotRequired[str]   # "no_flights"|"budget_exceeded"|"no_hotels"
    itinerary_fallback_action: NotRequired[str]   # what fallback did
    itinerary_fallback_alternatives: NotRequired[list]

    # ── Context flags ────────────────────────────────────────────────────
    has_existing_trip_context: bool
    alternative_destinations: list