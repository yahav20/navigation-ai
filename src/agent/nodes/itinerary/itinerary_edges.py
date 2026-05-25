# src/agent/nodes/itinerary/itinerary_edges.py
"""
Routing functions for the itinerary sub-graph.

STATE FIELDS USED FOR ROUTING:
  itinerary_feasible       bool  — True = plan OK so far, False = something failed
  observer_action          str   — "continue" | "complete" | ""
  itinerary_fallback_reason str  — error text; "max_retries_exceeded" is the hard-stop signal
  itinerary_fallback_action str  — what fallback did: "suggested_alternatives" |
                                   "days_adjusted_to_N" | "budget_bumped_500"
  itinerary_plan.retry_count int — incremented by Observer._trigger_replan()

ROUTING MAP:
  itinerary_planner  →  executor   (feasible=True)
                     →  fallback   (feasible=False — Planner hit MAX_RETRIES itself)

  itinerary_observer →  executor   (feasible=True,  action="continue")
                     →  formatter  (feasible=True,  action="complete")
                     →  planner    (feasible=False, reason != max_retries_exceeded)
                     →  fallback   (feasible=False, reason == max_retries_exceeded)

  itinerary_fallback →  formatter  (action == "suggested_alternatives" → show user)
                     →  planner    (action in days_adjusted / budget_bumped → retry)
                     →  formatter  (fallback itself failed / unknown → show what we have)
"""
from agent.state import AgentState

MAX_RETRIES_REASON = "max_retries_exceeded"


# ---------------------------------------------------------------------------
# after_itinerary_planner
# ---------------------------------------------------------------------------

def after_itinerary_planner(state: AgentState) -> str:
    """
    Planner already checks retry_count internally and sets itinerary_feasible=False
    when MAX_RETRIES is reached. We just honour that flag.
    """
    if state.get("itinerary_feasible", True):
        return "itinerary_executor"
    return "itinerary_fallback"


# ---------------------------------------------------------------------------
# after_itinerary_observer
# ---------------------------------------------------------------------------

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

    # Hard stop 2: Fallback safety - just in case the string didn't propagate
    if retry_count >= 3:
        return "itinerary_fallback"

    if not feasible and retry_count >= 2:
        return "itinerary_fallback"

    # Soft failure: let Planner try again with the observer_reason hint
    return "itinerary_planner"


# ---------------------------------------------------------------------------
# after_itinerary_fallback
# ---------------------------------------------------------------------------

def after_itinerary_fallback(state: AgentState) -> str:
    """
    Fallback has two outcomes:
      - It adjusted something (days / budget) and set itinerary_feasible=True
        → go back to Planner for a fresh attempt.
      - It gave up and suggested alternatives (or had no action)
        → go to Formatter to show the user whatever we have.

    CRITICAL: we must NEVER return "itinerary_planner" when feasible=False,
    because that creates the infinite loop seen in the logs.
    """
    action   = state.get("itinerary_fallback_action", "") or ""
    feasible = state.get("itinerary_feasible", False)

    retry_actions = {"days_adjusted", "budget_bumped_500"}

    # Only retry if Fallback explicitly fixed something AND marked it feasible
    if feasible and any(action.startswith(a) for a in retry_actions):
        return "itinerary_planner"

    # Everything else (suggested_alternatives, unknown, empty) → show output
    return "itinerary_formatter"