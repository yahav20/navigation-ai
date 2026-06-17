"""Routing edges for the LangGraph travel agent."""
from langgraph.graph import END

from agent.core.state import AgentState


def after_enrichment(state: AgentState) -> str:
    """Route after enrichment.

    build_itinerary          → plan_check   (HITL checkpoint before itinerary)
    adjustment + standalone  → plan_check   (re-run HITL mode selection)
    everything else          → flight_search (standard travel-planning path)
    Incomplete               → END          (enrichment asked a follow-up question)
    """
    if not state.get("enrichment_complete", False):
        return END

    if state.get("intent") == "build_itinerary":
        return "plan_check"

    # When the user adjusts an active itinerary (standalone or with_travel_data),
    # route back to plan_check to rebuild it instead of running a flight search.
    # itinerary_mode is only set after plan_check ran, so this condition is only
    # true when a prior itinerary existed. itinerary_mode=None → flight_search.
    if state.get("is_adjustment") and state.get("itinerary_mode") in ("standalone", "with_travel_data"):
        return "plan_check"

    return "flight_search"


def after_flight_search(state: AgentState) -> str:
    """Route to the travel agent when flights exist, else to the alternatives path."""
    if state.get("has_flights") and state.get("flight_options"):
        return "travel_agent"
    return "alternative_destination"


def after_travel_agent(state: AgentState) -> str:
    """Render the curated plan when the travel agent produced one, else skip the formatter."""
    return "formatter" if state.get("travel_plan") else "summary"


def after_security_gate(state: AgentState) -> str:
    """Route to router if safe, or skip directly to summary if blocked."""
    last_message = state["messages"][-1]

    if getattr(last_message, "name", "") == "security_gate":
        return "summary"

    return "router"


def after_router(state: AgentState) -> str:
    """Route from the RouterNode based on the classified intent."""
    intent = state.get("intent", "other")

    if intent == "new_travel_plan":
        return "extract_metadata"
    elif intent == "update_travel_plan":
        return "adjustments"
    elif intent == "advisor":
        return "advisor_planner"
    elif intent == "build_itinerary":
        # Skip metadata re-extraction when a travel plan already exists: MetadataNode
        # reads flight dates from the rendered plan output and may call _invalidate_flights(),
        # wiping travel_plan/flight_options before plan_check can use them.
        if state.get("travel_plan"):
            return "plan_check"
        return "extract_metadata"
    elif intent == "update_itinerary":
        return "itinerary_planner"
    elif intent == "general_chat":
        return "advisor_planner"
    elif intent == "out_of_scope":
        return "out_of_scope"
    else:
        return END


def after_travel_formatter(state: AgentState) -> str:
    """Show the HITL confirmation when a full travel plan exists; skip to summary otherwise."""
    travel_plan = state.get("travel_plan") or {}
    if travel_plan.get("hotels") and state.get("has_flights"):
        return "travel_confirmation"
    return "summary"


def after_travel_confirmation(state: AgentState) -> str:
    """Route based on what the user chose in TravelConfirmationNode.

    intent="build_itinerary" is written by the node on "yes" — plan_check
    then resolves hotel/flight selections and launches the itinerary builder.
    Anything else (including an unanswered or "no" choice) ends the turn.
    """
    if state.get("intent") == "build_itinerary":
        return "plan_check"
    return "summary"


def after_advisor_planner(state: AgentState) -> str:
    """Route to out_of_scope when planner detected a non-travel question, else to executor."""
    if state.get("advisor_out_of_scope"):
        return "out_of_scope"
    return "advisor_executor"
