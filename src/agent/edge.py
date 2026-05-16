"""Routing edges for the LangGraph travel agent."""
# src/agent/edge.py
from langgraph.graph import END

from agent.state import AgentState


# ---------------------------------------------------------
# Routing Function 1: From Enrichment
# ---------------------------------------------------------
def after_enrichment(state: AgentState) -> str:
    """Route to deterministic flight search when enrichment is complete."""
    return "flight_search" if state.get("enrichment_complete", False) else END


def after_flight_search(state: AgentState) -> str:
    """Route to the travel agent when flights exist, else to the alternatives path."""
    if state.get("has_flights") and state.get("flight_options"):
        return "travel_agent"
    return "alternative_destination"


def after_travel_agent(state: AgentState) -> str:
    """Render the curated plan when the travel agent produced one, else skip the formatter."""
    return "formatter" if state.get("travel_plan") else "summary"


def after_adjustments(state: AgentState) -> str:
    """Route directly to enrichment if an adjustment was made, skipping standard metadata extraction."""
    if state.get("is_adjustment"):
        return "enrichment"
    return "extract_metadata"

def after_router(state: AgentState) -> str:
    """Route from the RouterNode based on the classified intent."""
    intent = state.get("intent", "other")
    routes = {
        "new_travel_plan": "extract_metadata",
        "update_travel_plan": "adjustments",
        "recommendations": "rec_agent",
    }

    return routes.get(intent, END)


REC_MAX_STEPS = 4


def rec_should_continue(state: AgentState) -> str:
    """Route to rec_tools if the agent issued tool calls, otherwise send to rec_formatter.

    REC_MAX_STEPS is intentionally low — recommendation queries should rarely need more
    than 3 tool turns. A high step count is a symptom of the agent over-tooling.
    """
    last_message = state["messages"][-1]
    step_count = state.get("step_count", 0)

    if step_count >= REC_MAX_STEPS:
        return "rec_formatter"

    if getattr(last_message, "tool_calls", None):
        return "rec_tools"

    return "rec_formatter"
