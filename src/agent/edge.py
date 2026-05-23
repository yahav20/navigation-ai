"""Routing edges for the LangGraph travel agent."""
from langgraph.graph import END
from agent.state import AgentState


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _has_itinerary_data(state: AgentState) -> bool:
    """True when state already has flights + destination from a previous turn."""
    return bool(state.get("destination_city") and state.get("flight_options"))


# ---------------------------------------------------------------------------
# 1. After enrichment
# ---------------------------------------------------------------------------

def after_enrichment(state: AgentState) -> str:
    return "flight_search" if state.get("enrichment_complete", False) else END


# ---------------------------------------------------------------------------
# 2. After flight search
# ---------------------------------------------------------------------------

def after_flight_search(state: AgentState) -> str:
    has_flights = state.get("has_flights") and state.get("flight_options")

    if not has_flights:
        return "alternative_destination"

    if state.get("build_itinerary"):
        return "itinerary_planner"

    return "travel_agent"


# ---------------------------------------------------------------------------
# 3. After travel agent
# ---------------------------------------------------------------------------

def after_travel_agent(state: AgentState) -> str:
    if state.get("build_itinerary") and _has_itinerary_data(state):
        return "itinerary_planner"
    if state.get("travel_plan"):
        return "formatter"
    return "summary"


# ---------------------------------------------------------------------------
# 4. After router  ← KEY FIX for update-itinerary mid-conversation
# ---------------------------------------------------------------------------

def after_router(state: AgentState) -> str:
    intent = state.get("intent", "other")

    # ── Itinerary intents ──────────────────────────────────────────────
    if intent in ("itinerary", "build_itinerary"):
        if _has_itinerary_data(state):
            return "itinerary_planner"   # data exists → skip to planner
        return "extract_metadata"        # fresh → collect params first

    # ── Update existing itinerary (e.g. "I'm vegetarian, update my plan") ──
    # This is the KEY FIX: update_itinerary goes to itinerary_planner directly
    # if we already have an itinerary, otherwise treat as adjustment
    if intent == "update_itinerary":
        if state.get("itinerary_plan"):
            return "itinerary_planner"   # re-plan with updated prefs in state
        return "adjustments"             # no itinerary yet → standard adjustment

    # ── Standard intents ──────────────────────────────────────────────
    if intent == "new_travel_plan":
        return "extract_metadata"

    if intent == "update_travel_plan":
        return "adjustments"

    if intent == "recommendations":
        return "rec_agent"

    if intent == "general_chat":
        return "general_chat"

    return END


# ---------------------------------------------------------------------------
# 5. After alternative destination
# ---------------------------------------------------------------------------

def after_alternative_destination(state: AgentState) -> str:
    return "formatter_alternative"


# ---------------------------------------------------------------------------
# 6. Security gate
# ---------------------------------------------------------------------------

def after_security_gate(state: AgentState) -> str:
    last = state["messages"][-1]
    if getattr(last, "name", "") == "security_gate":
        return "summary"
    return "router"


# ---------------------------------------------------------------------------
# 7. Itinerary sub-graph
# ---------------------------------------------------------------------------

def after_itinerary_planner(state: AgentState) -> str:
    if state.get("itinerary_feasible", False):
        plan = state.get("itinerary_plan", {})
        if plan and not plan.get("error"):
            return "itinerary_executor"
    return "itinerary_fallback"


def after_itinerary_executor(state: AgentState) -> str:
    """Always go to observer after executor runs."""
    return "itinerary_observer"


def after_itinerary_observer(state: AgentState) -> str:
    """
    Observer decides: re-plan, fallback, or done.
    Re-plan only if retries remain AND the issue is fixable.
    """
    plan_state = state.get("itinerary_plan", {})
    retries = plan_state.get("observer_retries", 0)
    issues = plan_state.get("observer_issues", [])
    feasible = state.get("itinerary_feasible", True)

    # Has issues AND retries left → re-plan
    if issues and not feasible and retries < 2:
        fixable = {"missing_days", "budget_exceeded"}
        if any(i in fixable for i in issues):
            return "itinerary_planner"

    # Budget exceeded and no retries → fallback
    if not feasible:
        return "itinerary_fallback"

    # All good
    return "summary"


def after_itinerary_fallback(state: AgentState) -> str:
    action = state.get("itinerary_fallback_action", "")
    feasible = state.get("itinerary_feasible", False)

    RETRY_ACTIONS = {
        "adjusted_days_1", "adjusted_days_2", "adjusted_days_3",
        "adjusted_days_4", "adjusted_days_5", "adjusted_days_6",
        "budget_bumped_500", "relaxed_kosher_for_hotels",
    }
    if feasible and action in RETRY_ACTIONS:
        return "itinerary_planner"

    return "itinerary_formatter"


# ---------------------------------------------------------------------------
# 8. Chat / Rec loops
# ---------------------------------------------------------------------------

CHAT_MAX_STEPS = 2
REC_MAX_STEPS = 4


def chat_should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if state.get("step_count", 0) >= CHAT_MAX_STEPS:
        return "summary"
    if getattr(last, "tool_calls", None):
        return "chat_tools"
    return "summary"


def rec_should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if state.get("step_count", 0) >= REC_MAX_STEPS:
        return "rec_formatter"
    if getattr(last, "tool_calls", None):
        return "rec_tools"
    return "rec_formatter"