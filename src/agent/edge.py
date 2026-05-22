"""Routing edges for the LangGraph travel agent."""
# src/agent/edge.py
from langgraph.graph import END

from agent.state import AgentState


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _has_itinerary_data(state: AgentState) -> bool:
    """
    True when the state already contains enough data to run itinerary_planner:
      - a destination
      - flight_options already fetched (from a previous flight_search run)
    """
    return bool(
        state.get("destination_city")
        and state.get("flight_options")  # populated by FlightSearchNode
    )


# ---------------------------------------------------------
# Routing Function 1: From Enrichment
# ---------------------------------------------------------
def after_enrichment(state: AgentState) -> str:
    """Route to deterministic flight search when enrichment is complete."""
    return "flight_search" if state.get("enrichment_complete", False) else END


# ---------------------------------------------------------
# Routing Function 2: From FlightSearch
# ---------------------------------------------------------
def after_flight_search(state: AgentState) -> str:
    """
    Decision point after flights are fetched from the DB.

    Possible outcomes:
      - No flights found at all            → alternative_destination
      - Flights found + itinerary intent   → itinerary_planner
      - Flights found + normal plan intent → travel_agent
    """
    has_flights = state.get("has_flights") and state.get("flight_options")

    if not has_flights:
        return "alternative_destination"

    # User explicitly asked for a full itinerary plan
    if state.get("build_itinerary"):
        return "itinerary_planner"

    return "travel_agent"


# ---------------------------------------------------------
# Routing Function 3: From TravelAgent
# ---------------------------------------------------------
def after_travel_agent(state: AgentState) -> str:
    """
    After TravelAgentNode produces a travel_plan, check whether the user
    also wants a full itinerary (they may have asked mid-conversation).

    Intent is re-checked here because the router may have classified the
    follow-up message as 'itinerary' AFTER travel_agent already ran.
    """
    if state.get("travel_plan"):
        # If user subsequently asked for full itinerary, data is ready → go directly
        if state.get("build_itinerary") and _has_itinerary_data(state):
            return "itinerary_planner"
        return "formatter"

    # travel_agent produced nothing useful → fallback to summary
    return "summary"


# ---------------------------------------------------------
# Routing Function 4: From Router
# ---------------------------------------------------------
def after_router(state: AgentState) -> str:
    """
    Route from the RouterNode based on the classified intent.

    Key logic:
      - 'itinerary' intent + data already in state (mid-conversation)
        → skip metadata/enrichment/flight_search, go directly to itinerary_planner
      - 'itinerary' intent + no data yet (fresh conversation)
        → go through extract_metadata first to collect origin/destination/days
      - 'update_travel_plan' + itinerary was already built
        → treat as itinerary continuation
    """
    intent = state.get("intent", "other")

    # ── Itinerary intent ────────────────────────────────────────────────
    if intent in ("itinerary", "build_itinerary"):
        # Mark state so downstream nodes (flight_search, travel_agent) know
        # we want a full itinerary at the end.
        # NOTE: state mutations inside edge functions are NOT persisted by
        # LangGraph. We return the flag via the node return value instead.
        # The actual flag is set by RouterNode (see router.py update below).

        if _has_itinerary_data(state):
            # Mid-conversation: flights + destination already known → skip to planner
            return "itinerary_planner"
        else:
            # Fresh start: need to collect metadata first
            return "extract_metadata"

    # ── Standard intents ────────────────────────────────────────────────
    if intent == "new_travel_plan":
        return "extract_metadata"

    if intent == "update_travel_plan":
        return "adjustments"

    if intent == "recommendations":
        return "rec_agent"

    if intent == "general_chat":
        return "general_chat"

    return END


# ---------------------------------------------------------
# Routing Function 5: From AlternativeDestination
# ---------------------------------------------------------
def after_alternative_destination(state: AgentState) -> str:
    """
    After showing alternative destinations, check if the user's follow-up
    asked for a full itinerary for one of them.

    This edge is triggered on the NEXT turn when the user replies something
    like "great, plan the full trip to Lisbon".

    The node itself just renders; routing happens here on re-entry via the
    router → after_router path (intent='itinerary' + data in state).
    So this function just routes to formatter_alternative as before.
    """
    return "formatter_alternative"


# ---------------------------------------------------------
# Routing Function 6: Security gate
# ---------------------------------------------------------
def after_security_gate(state: AgentState) -> str:
    """Route to router if safe, or skip directly to summary if blocked."""
    last_message = state["messages"][-1]

    if getattr(last_message, "name", "") == "security_gate":
        return "summary"

    return "router"


# ---------------------------------------------------------
# Routing Function 7: Itinerary sub-graph
# ---------------------------------------------------------
def after_itinerary_planner(state: AgentState) -> str:
    """
    Route after the planner runs feasibility check.

      → "itinerary_builder"  plan is feasible and parsed correctly
      → "itinerary_fallback" something needs recovery
    """
    if state.get("itinerary_feasible", False):
        plan = state.get("itinerary_plan", {})
        if plan and not plan.get("error"):
            return "itinerary_builder"
    return "itinerary_fallback"


def after_itinerary_fallback(state: AgentState) -> str:
    """
    Route after fallback decides its recovery strategy.

      → "itinerary_planner"   fallback adjusted dates/budget → retry planning
      → "itinerary_formatter" fallback found alternatives or gave up
    """
    action = state.get("itinerary_fallback_action", "")
    feasible = state.get("itinerary_feasible", False)

    if feasible and action in (
        "adjusted_days_1", "adjusted_days_2", "adjusted_days_3",
        "adjusted_days_4", "adjusted_days_5", "adjusted_days_6",
        "budget_bumped_500",
        "relaxed_kosher_for_hotels",
    ):
        return "itinerary_planner"

    return "itinerary_formatter"


# ---------------------------------------------------------
# Routing Function 8: Chat / Rec loops
# ---------------------------------------------------------
CHAT_MAX_STEPS = 2


def chat_should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    step_count = state.get("step_count", 0)

    if step_count >= CHAT_MAX_STEPS:
        return "summary"
    if getattr(last_message, "tool_calls", None):
        return "chat_tools"
    return "summary"


REC_MAX_STEPS = 4


def rec_should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    step_count = state.get("step_count", 0)

    if step_count >= REC_MAX_STEPS:
        return "rec_formatter"
    if getattr(last_message, "tool_calls", None):
        return "rec_tools"
    return "rec_formatter"