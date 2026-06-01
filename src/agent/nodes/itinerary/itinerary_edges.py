"""
Routing edges for the Plan-and-Execute itinerary agent.
=======================================================
Flow:
  plan_check        →(conditional)→ itinerary_planner | summary
  itinerary_planner →(conditional)→ itinerary_executor | itinerary_formatter
  itinerary_executor → itinerary_replanner
  itinerary_replanner →(conditional)→ itinerary_executor | itinerary_planner | itinerary_formatter
"""
from __future__ import annotations

from langgraph.graph import END

from agent.state import AgentState


def after_plan_check(state: AgentState) -> str:
    """
    Route from PlanCheckNode.
      with_travel_data / standalone → itinerary_planner
      redirect_to_travel            → extract_metadata  (run travel planning flow)
    """
    mode = state.get("itinerary_mode")
    if mode in ("with_travel_data", "standalone"):
        return "itinerary_planner"
    if mode == "redirect_to_travel":
        return "extract_metadata"
    return "summary"


def after_itinerary_planner(state: AgentState) -> str:
    """
    Route from Planner.
    If the Planner hit MAX_REPLANS it sets itinerary_feasible=False — go to Formatter.
    Otherwise always go to Executor.
    """
    if not state.get("itinerary_feasible", True):
        return "itinerary_formatter"
    return "itinerary_executor"


def after_itinerary_replanner(state: AgentState) -> str:
    """
    Route from Replanner based on the replanner_action it set:
      continue → executor  (last step succeeded, more steps remain)
      replan   → planner   (last step failed, retry with corrective context)
      done     → formatter (all steps complete, or retries exhausted)
    """
    action = state.get("replanner_action", "done")

    if action == "continue":
        return "itinerary_executor"
    if action == "replan":
        return "itinerary_planner"
    return "itinerary_formatter"
