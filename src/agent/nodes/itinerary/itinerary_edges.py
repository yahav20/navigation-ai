"""
ITINERARY EDGE FUNCTIONS
========================
Add these three functions to your existing agent/edge.py.

Graph wiring in graph.py:

    builder.add_conditional_edges(
        "itinerary_planner", after_itinerary_planner,
        {"itinerary_executor": "itinerary_executor",
         "itinerary_fallback": "itinerary_fallback"},
    )
    builder.add_edge("itinerary_executor", "itinerary_observer")

    builder.add_conditional_edges(
        "itinerary_observer", after_itinerary_observer,
        {"itinerary_planner":  "itinerary_planner",   # re-plan loop
         "itinerary_fallback": "itinerary_fallback",  # unrecoverable
         "summary":            "summary"},             # done
    )
    builder.add_conditional_edges(
        "itinerary_fallback", after_itinerary_fallback,
        {"itinerary_planner": "itinerary_planner",    # fix applied → retry
         "summary":           "summary"},              # show alternatives
    )
"""

from agent.state import AgentState

FALLBACK_RETRY_ACTIONS = {
    "adjusted_days_1", "adjusted_days_2", "adjusted_days_3",
    "adjusted_days_4", "adjusted_days_5", "adjusted_days_6",
    "budget_bumped_500",
}


def after_itinerary_planner(state: AgentState) -> str:
    """
    Planner either produced a plan (→ executor) or hit max retries / no data (→ fallback).
    """
    if state.get("itinerary_feasible", False):
        return "itinerary_executor"
    return "itinerary_fallback"


def after_itinerary_observer(state: AgentState) -> str:
    """
    Observer decided:
      feasible=True  → complete, go to summary
      feasible=False + reason fixable → re-plan (planner)
      feasible=False + hard reason → fallback
    """
    feasible = state.get("itinerary_feasible", False)
    reason   = state.get("itinerary_fallback_reason", "")

    if feasible:
        return "summary"

    # Hard failures → fallback (no re-plan loop)
    if reason in ("no_flights", "no_hotels", "max_retries_exceeded"):
        return "itinerary_fallback"

    # Fixable → re-plan
    return "itinerary_planner"


def after_itinerary_fallback(state: AgentState) -> str:
    """
    Fallback either applied a fix (days/budget) → retry planner,
    or showed alternatives → summary.
    """
    action  = state.get("itinerary_fallback_action", "")
    feasible = state.get("itinerary_feasible", False)

    if feasible and action in FALLBACK_RETRY_ACTIONS:
        return "itinerary_planner"

    return "summary"
