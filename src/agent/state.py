"""Typed state shared across nodes of the travel-agent graph."""
from typing import Annotated, NotRequired, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Track the state of the agent across the graph execution.

    'add_messages' ensures history is preserved.
    """

    messages: Annotated[list, add_messages]
    current_city: str
    destination_city: str
    total_budget: float
    step_count: int
    enrichment_complete: bool
    user_preferences: dict
    is_adjustment: bool            # True when the user just modified their trip parameters
    enrichment_asked_fields: list  # field keys already requested from the user
    budget_optional: bool          # True when user explicitly declined to provide a budget
    trip_days: int                 # number of trip days; defaults to 3 if user skips after being asked
    summary: str                   # rolling conversation summary maintained by summary_node
    alternative_destinations: list # populated when fetch_flights returns no results for the route
    flight_options: NotRequired[list[dict]]  # deterministic flight results used only for branching/response context
    has_flights: NotRequired[bool]           # True when flight_options contains usable route data
    travel_plan: NotRequired[dict]           # curated TravelPlan dump produced by TravelAgentNode for the formatter
 
    itinerary_plan: NotRequired[dict]        # curated day-by-day itinerary produced by itinerary agent
    intent: str                     # intent classification
    has_existing_trip_context: bool # True when the user has an active trip in the system
 
    # ── Itinerary routing flag ───────────────────────────────────────────
    build_itinerary: NotRequired[bool]
    # Set to True by RouterNode when intent is 'itinerary' or 'build_itinerary'.
    # Read by after_flight_search and after_travel_agent to branch into the
    # itinerary sub-graph instead of the standard formatter path.
 
    # ── Itinerary sub-graph fields ───────────────────────────────────────
    itinerary_feasible: NotRequired[bool]
    # True  → plan passed feasibility check, builder can execute it.
    # False → fallback node must intervene.
 
    itinerary_fallback_reason: NotRequired[str]
    # One of: "no_flights" | "budget_exceeded" | "no_hotels" | None
 
    itinerary_fallback_action: NotRequired[str]
    # What the fallback node actually did, e.g.:
    # "adjusted_days_2" | "budget_bumped_500" | "suggested_alternatives"
 
    itinerary_fallback_alternatives: NotRequired[list]
    # Up to 3 alternative city names when the fallback suggests new destinations.
 
    itinerary_data_bundle: NotRequired[dict]
    # Raw data collected by the planner (hotels, activities, flights, weather).
    # Passed to fallback node to avoid a second DB round-trip.
 
    itinerary_formatted: NotRequired[str]
    # Final Markdown string produced by ItineraryFormatterNode.