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

    The Planner checks MAX_REPLANS and MAX_RETRIES. If it hits the hard limit,
    it sets 'itinerary_feasible' to False.
    """
    if not state.get("itinerary_feasible", True):
        return "itinerary_fallback"
    return "itinerary_executor"


def after_itinerary_observer(state: AgentState) -> str:
    """
    Route from Observer based on the explicit `observer_action` it set:
      continue    → executor (more steps to run)
      complete    → formatter (all steps valid)
      no_flights  → alternative_destination (skip replanning entirely)
      replan      → planner (soft failure, retry with hint)
      fallback    → fallback (hard stop, max retries hit)
    """
    action = state.get("observer_action", "complete")

    if action == "continue":
        return "itinerary_executor"
    if action == "no_flights":
        return "alternative_destination"
    if action == "replan":
        return "itinerary_planner"
    if action == "fallback":
        return "itinerary_fallback"
    # "complete" or any unknown value: ship what we have.
    return "itinerary_formatter"


def after_itinerary_fallback(state: AgentState) -> str:
    """
    Route from Fallback to either Planner (relaxed constraints, retry) or Formatter
    (show alternatives / whatever we have).

    Fallback emits one of:
      days_adjusted_to_N      → retry via Planner
      budget_bumped_500       → retry via Planner
      suggested_alternatives  → render output
    """
    action = state.get("itinerary_fallback_action", "") or ""
    feasible = state.get("itinerary_feasible", False)

    retry_prefixes = ("days_adjusted", "budget_bumped_500")
    if feasible and action.startswith(retry_prefixes):
        return "itinerary_planner"

    return "itinerary_formatter"
