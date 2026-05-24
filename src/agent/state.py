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
    intent: str                    # intent classification
    
    # --- New Fields for Step-by-Step Itinerary Execution ---
    current_step_index: NotRequired[int]              # Tracks which step the Executor should run next
    itinerary_feasible: NotRequired[bool]             # True if the plan is progressing successfully, False if failure occurred
    itinerary_fallback_reason: NotRequired[str]       # The error message/reason that triggered the Replanner or Fallback
    observer_action: NotRequired[str]                 # Edge routing signal from Observer: "continue", "complete", etc.
    itinerary_fallback_action: NotRequired[str]       # The action taken by the Fallback node (e.g., "adjusted_days", "suggested_alternatives")
    itinerary_fallback_alternatives: NotRequired[list[str]] # List of alternative cities suggested by Fallback