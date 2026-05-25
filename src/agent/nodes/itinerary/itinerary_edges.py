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
    Four outcomes:
      1. Still running  → back to executor
      2. Done & valid   → formatter
      3. No flights     → alternative_destination (skip replanning entirely)
      4. Failed         → planner (retry) OR fallback (hard stop)
    """
    feasible = state.get("itinerary_feasible", False)
    action   = state.get("observer_action", "")
    reason   = state.get("itinerary_fallback_reason", "") or ""

    plan_state  = state.get("itinerary_plan") or {}
    retry_count = plan_state.get("retry_count", 0)

    # ── Success ──
    if feasible and action == "complete":
        return "itinerary_formatter"

    # ── Mid-execution: next step ──
    if feasible and action == "continue":
        return "itinerary_executor"

    # ── No flights: go directly to alternative_destination node ──
    if action == "no_flights":
        return "alternative_destination"

    # ── Failure path ──
    
    # Hard stop 1: The Observer explicitly passed the 'max_retries_exceeded' flag
    if MAX_RETRIES_REASON in reason:
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