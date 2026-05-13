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
    for msg in state.get("messages", []):
        if getattr(msg, "type", "") != "tool":
            continue
        if getattr(msg, "name", "") != "fetch_flights":
            continue
        
        if "flight_number" in msg.content or "route" in msg.content:
            return False
            
    return True
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