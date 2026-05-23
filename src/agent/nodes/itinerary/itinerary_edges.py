# src/agent/nodes/itinerary/itinerary_edges.py
from agent.state import AgentState

def after_itinerary_planner(state: AgentState) -> str:
    if state.get("itinerary_feasible", False):
        return "itinerary_executor"
    return "itinerary_fallback"

def after_itinerary_executor(state: AgentState) -> str:
    return "itinerary_observer"

def after_itinerary_observer(state: AgentState) -> str:
    feasible = state.get("itinerary_feasible", False)
    action   = state.get("observer_action", "")
    reason   = state.get("itinerary_fallback_reason", "")

    if feasible and action == "complete":
        return "itinerary_formatter" 
    # =========================================================
        
    if feasible and action == "continue":
        return "itinerary_executor"

    if not feasible:
        if reason in ("no_flights", "no_hotels", "max_retries_exceeded"):
            return "itinerary_fallback" 
        return "itinerary_planner" 

def after_itinerary_fallback(state: AgentState) -> str:
   
    if state.get("itinerary_fallback_action") == "suggested_alternatives":
        return "itinerary_formatter"
    return "itinerary_planner"