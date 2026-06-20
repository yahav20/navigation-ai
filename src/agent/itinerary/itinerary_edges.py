"""
Routing edges for the Plan-and-Execute itinerary agent.
=======================================================
Flow:
  plan_check          →(conditional)→ itinerary_planner | summary
  itinerary_planner   →(conditional)→ itinerary_executor | itinerary_formatter
  itinerary_executor  → itinerary_replanner
  itinerary_replanner →(conditional)→ itinerary_executor | itinerary_planner | itinerary_critic
  itinerary_critic    →(conditional)→ itinerary_planner | itinerary_formatter
"""
from __future__ import annotations

from langgraph.graph import END

from agent.core.state import AgentState


def after_plan_check(state: AgentState) -> str:
    """
    Route from PlanCheckNode.
      with_travel_data / standalone → segment_planner  (decides single vs. multi-city)
      redirect_to_travel            → extract_metadata  (run travel planning flow)
    """
    mode = state.get("itinerary_mode")
    if mode in ("with_travel_data", "standalone"):
        return "segment_planner"
    if mode == "redirect_to_travel":
        return "extract_metadata"
    return "summary"


def after_itinerary_planner(state: AgentState) -> str:
    """
    Route from Planner.
    If the Planner hit MAX_REPLANS it sets itinerary_feasible=False — finish the
    trip (trip_formatter renders the error for a single leg exactly as before).
    Otherwise always go to Executor.
    """
    if not state.get("itinerary_feasible", True):
        return "trip_formatter"
    return "itinerary_executor"


def after_itinerary_replanner(state: AgentState) -> str:
    """
    Route from Replanner based on the replanner_action it set:
      continue → executor  (last step succeeded, more steps remain)
      replan   → planner   (last step failed, retry with corrective context)
      done     → critic    (all steps complete — critic validates budget before formatter)
    """
    action = state.get("replanner_action", "done")

    if action == "continue":
        return "itinerary_executor"
    if action == "replan":
        return "itinerary_planner"
    return "itinerary_critic"


def after_itinerary_critic(state: AgentState) -> str:
    """
    Route from CriticNode based on critic_action:
      replan_cheaper / reduce_day / switch_travel / adjust_prefs → planner (loop back)
      pass / ignore_budget / abort                               → leg_collect
    leg_collect captures the finished leg, then either loops to the next city or
    hands off to trip_formatter.
    """
    action = state.get("critic_action", "pass")
    if action in ("replan_cheaper", "reduce_day", "switch_travel", "adjust_prefs", "fetch_min_prices"):
        return "itinerary_planner"
    return "leg_collect"
