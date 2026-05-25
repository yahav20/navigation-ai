"""
Routing edges for the Plan-and-Execute itinerary agent.
======================================================
These functions handle the conditional routing between the Planner, 
Executor, Observer, and Fallback nodes.
"""
from __future__ import annotations

from agent.state import AgentState


def after_itinerary_planner(state: AgentState) -> str:
    """
    Route from Planner to Executor, or Fallback if planning failed.
    
    The v3 Planner checks MAX_REPLANS and MAX_RETRIES. If it hits the hard limit, 
    it sets 'itinerary_feasible' to False.
    """
    # If the Planner explicitly flagged the plan as unfeasible (e.g., hit max limits)
    if not state.get("itinerary_feasible", True):
        return "itinerary_fallback"
        
    # Standard path: Plan generated successfully, go execute the first/next step
    return "itinerary_executor"


def after_itinerary_observer(state: AgentState) -> str:
    """
    Route from Observer based on the validation outcome.
    
    The Observer acts as the System Controller and explicitly sets 'observer_action' 
    to one of four states: "continue", "replan", "fallback", or "complete".
    """
    action = state.get("observer_action", "complete")
    
    if action == "continue":
        # Step was valid, but plan has more steps. Loop back to Executor.
        return "itinerary_executor"
        
    elif action == "replan":
        # Step failed or soft validation failed. Loop back to Planner.
        return "itinerary_planner"
        
    elif action == "fallback":
        # Repeated failures reached MAX_RETRIES. Hard stop.
        return "itinerary_fallback"
        
    else:
        # action == "complete" -> All steps valid, final markdown generated.
        return "itinerary_formatter"


def after_itinerary_fallback(state: AgentState) -> str:
    """
    Route from Fallback to either replan with relaxed constraints or format final output.
    """
    action = state.get("itinerary_fallback_action", "format")
    
    if action == "replan":
        # The Fallback node relaxed constraints (e.g., lowered star rating, increased budget)
        # and wants the Planner to try one more time from a clean slate.
        return "itinerary_planner"
        
    # The Fallback node either generated a minimal viable itinerary or gave