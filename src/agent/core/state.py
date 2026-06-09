"""Typed state shared across nodes of the travel-agent graph."""
import operator
from typing import Annotated, NotRequired, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Track the state of the agent across the graph execution.

    'add_messages' ensures history is preserved.
    """

    messages: Annotated[list, add_messages]
    ui: Annotated[list, operator.add]  
    # Internal, non-UI trace accumulated by reasoning nodes (planner/executor/
    # observer). Kept out of `messages` so the chat UI never renders it; visible
    # in LangSmith / state inspection. `operator.add` appends across steps.
    progress_log: Annotated[list, operator.add]
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
    num_adults: NotRequired[int]   # travelling adults; defaults to 1 if user skips after being asked
    num_children: NotRequired[int] # travelling children; defaults to 0
    num_rooms: NotRequired[int]    # hotel rooms; auto-derived from group size unless the user sets it explicitly
    summary: str                   # rolling conversation summary maintained by summary_node
    last_summarized_index: NotRequired[int]  # watermark: count of messages already folded into `summary`
    alternative_destinations: list # populated when fetch_flights returns no results for the route
    trip_start: NotRequired[str]             # approximate trip start as YYYY-MM-DD or YYYY-MM
    flight_options: NotRequired[list[dict]]  # deterministic outbound flight results
    return_flight_options: NotRequired[list[dict]]  # deterministic return flight results (destination -> origin)
    has_flights: NotRequired[bool]           # True when both outbound and return options are usable
    travel_plan: NotRequired[dict]           # curated TravelPlan dump produced by TravelAgentNode for the formatter
    itinerary_plan: NotRequired[dict]        # curated day-by-day itinerary produced by itinerary agent
    intent: str                    # intent classification

    # --- Itinerary Step-by-Step Execution ---
    current_step_index: NotRequired[int]               # Tracks which step the Executor should run next
    itinerary_feasible: NotRequired[bool]              # True if plan is progressing, False on hard failure
    itinerary_fallback_reason: NotRequired[str]        # Set by Replanner on max retries; read by Formatter
    itinerary_mode: NotRequired[str]                   # "with_travel_data" | "standalone" | "redirect_to_travel"
    replanner_action: NotRequired[str]                 # Edge routing signal: "continue" | "replan" | "done"
    itinerary_selected_hotel: NotRequired[dict]        # Raw hotel dict (lat/lng/price) from DB
    itinerary_selected_outbound_flight: NotRequired[dict]  # Raw outbound flight dict (times/price)
    itinerary_selected_return_flight: NotRequired[dict]    # Raw return flight dict (times/price)
    critic_attempts: NotRequired[int]           # Number of auto-replan attempts fired by critic
    critic_action: NotRequired[str]             # Routing signal: "pass"|"replan_cheaper"|"switch_travel"|"min_travel"|"reduce_day"|"adjust_prefs"|"ignore_budget"|"abort"
    critic_adjustment_request: NotRequired[str] # Free-text adjustment captured via HITL "adjust_prefs" option
    critic_min_prices: NotRequired[dict]        # Minimum available flight/hotel prices (standalone only, set on "min_travel" action)
    use_min_prices_for_budget: NotRequired[bool]  # When True, verify_budget uses fetch_min_prices data instead of fetch_avg_prices
    switch_travel_triggered: NotRequired[bool]    # Set by critic; tells planner to inject switch_travel_options step

    # --- Advisor ---
    advisor_plan: list                 # list of {tool_name, args} dicts produced by advisor_planner
    advisor_last_tool_results: list    # [{tool_name, args, result}] accumulated within the current turn
    advisor_shown_cities: list         # accumulates every city name presented to the user across advisor turns
    advisor_replan_count: int          # number of replan cycles completed in the current turn
    advisor_data_collected: NotRequired[str]  # DATA COLLECTED block passed to the advisor formatter (kept out of `messages`)
    advisor_out_of_scope: NotRequired[bool]  # True when planner detects a non-travel question
