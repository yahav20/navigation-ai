"""Routing edges for the LangGraph travel agent."""
# src/agent/edge.py
import json

from langgraph.graph import END

from agent.state import AgentState

MAX_STEPS = 5


# ---------------------------------------------------------
# Routing Function 1: From Enrichment
# ---------------------------------------------------------
def after_enrichment(state: AgentState) -> str:
    """Conditional edge: route to agent when enrichment is complete, else surface question to user."""
    return "agent" if state.get("enrichment_complete", False) else END

# ---------------------------------------------------------
# Helper for the Main Agent Edge
# ---------------------------------------------------------
def _no_real_flights_in_history(state: AgentState) -> bool:
    """Return True when the message history contains no usable flight records."""
    has_searched = False
    has_flights = False

    for msg in state.get("messages", []):
        if getattr(msg, "type", "") != "tool":
            continue
        if getattr(msg, "name", "") != "fetch_flights":
            continue

        has_searched = True
        
        if "flight_number" in msg.content or "route" in msg.content:
            has_flights = True
            break
        
    if not has_searched:
        return False

    return not has_flights
# ---------------------------------------------------------
# Routing Function 2: From Agent Core
# ---------------------------------------------------------
def should_continue(state: AgentState) -> str:
    """Return the next graph path based on the model's output.

    Returns 'tools' if the model wants to call a function, 'alternative_destination'
    when fetch_flights came back empty for a known origin/destination, otherwise 'formatter'.
    """
    last_message = state["messages"][-1]
    step_count = state.get("step_count", 0)
    origin = state.get("current_city")
    dest = state.get("destination_city")

    # Safety Check: Stop after 5 tool invocations in a single turn
    if step_count >= MAX_STEPS:
        if origin and dest and _no_real_flights_in_history(state):
            return "alternative_destination"
        return "formatter"

    if getattr(last_message, "tool_calls", None):
        return "tools"

    if origin and dest and _no_real_flights_in_history(state):
        return "alternative_destination"

    return "formatter"


def after_adjustments(state: AgentState) -> str:
    """Route directly to enrichment if an adjustment was made, skipping standard metadata extraction."""
    if state.get("is_adjustment"):
        return "enrichment"
    return "extract_metadata"

def after_router(state: AgentState) -> str:
    """Route from the RouterNode based on the classified intent."""
    intent = state.get("intent", "other")

    if intent == "new_travel_plan":
        return "extract_metadata"
    elif intent == "update_travel_plan":
        return "adjustments"
    elif intent == "recommendations":
        return "rec_agent"
    else:
        return END


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
