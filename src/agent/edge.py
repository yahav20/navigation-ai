"""Routing edges for the LangGraph travel agent."""
# src/agent/edge.py
from langgraph.graph import END

from agent.state import AgentState


# ---------------------------------------------------------
# Routing Function 1: From Enrichment
# ---------------------------------------------------------
def after_enrichment(state: AgentState) -> str:
    """Route after enrichment.

    build_itinerary  → plan_check   (HITL checkpoint before itinerary)
    everything else  → flight_search (standard travel-planning path)
    Incomplete       → END          (enrichment asked a follow-up question)
    """
    if not state.get("enrichment_complete", False):
        return END

    if state.get("intent") == "build_itinerary":
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
        # Share the same metadata + enrichment path as travel planning.
        # plan_check (reached after enrichment) handles the HITL fork.
        return "extract_metadata"
    elif intent == "general_chat":
        return "advisor_planner"
    elif intent == "out_of_scope":
        return "out_of_scope"
    else:
        return END


def after_advisor_planner(state: AgentState) -> str:
    """Route to out_of_scope when planner detected a non-travel question, else to executor."""
    if state.get("advisor_out_of_scope"):
        return "out_of_scope"
    return "advisor_executor"
