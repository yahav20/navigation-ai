"""
ItineraryExecutorNode — thin dispatcher.

Runs ONE step per graph invocation. Dispatches to the matching handler in
step_handlers.py, then writes the structured result to state.

Flow per invocation:
  1. Load plan + accumulated results from state
  2. Pick the current step by index
  3. Dispatch to the correct handler via STEP_HANDLER_MAP
  4. Commit result to step_results / execution_history
  5. Advance current_step_index; set itinerary_feasible based on status

See step_handlers.py for all step logic (fetch_activities, fetch_weather,
build_day_schedule). Budget verification is handled by the Critic node, not
as a plan step.
"""
from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from agent.nodes.itinerary.schemas import ExecutionPlan
from agent.nodes.itinerary.step_handlers import (
    handle_fetch_activities,
    handle_fetch_weather,
    handle_build_day,
    handle_verify_budget,
    _wrap_result,
    _minimal_trace,
)
from agent.state import AgentState


class ItineraryExecutorNode:
    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm

    def __call__(self, state: AgentState) -> dict:
        plan_state = state.get("itinerary_plan", {})
        plan       = ExecutionPlan(**plan_state["execution_plan"])
        results    = dict(plan_state.get("step_results", {}))
        history: list[dict] = list(plan_state.get("execution_history", []))
        mode       = state.get("itinerary_mode", "standalone")

        destination = state.get("destination_city", "")
        origin      = state.get("current_city", "")
        trip_days   = state.get("trip_days", plan.total_days)
        budget      = state.get("total_budget", 0)
        prefs       = state.get("user_preferences", {})

        current_index = state.get("current_step_index", 0)

        if current_index >= len(plan.steps):
            return _state_update(current_index, plan_state, results, history,
                                 ["✅ All steps complete — passing to Replanner."],
                                 feasible=True)

        step     = plan.steps[current_index]
        step_key = f"{step.step_type}_{step.step_id}"
        log_lines = [
            f"⚙️ **Step {current_index + 1}/{len(plan.steps)}:** `{step.step_type}`"
            + (f" (Day {step.day})" if step.day else "")
        ]

        ctx = dict(destination=destination, origin=origin,
                   trip_days=trip_days, budget=budget, prefs=prefs)

        current_plan_keys = {f"{s.step_type}_{s.step_id}" for s in plan.steps}

        # ── Dispatch ──────────────────────────────────────────────────────────
        if step.step_type == "fetch_weather":
            result = handle_fetch_weather(step, results, history, ctx)

        elif step.step_type == "fetch_activities":
            result = handle_fetch_activities(step, results, history, ctx, trip_days, self.llm)

        elif step.step_type == "build_day_schedule":
            result = handle_build_day(
                step, results, destination, trip_days, current_plan_keys, mode, state,
            )

        elif step.step_type == "verify_budget":
            result = handle_verify_budget(
                results, budget, trip_days, destination, origin, mode, state,
            )

        else:
            result = _wrap_result(
                status="failed",
                data=None,
                error=f"Unknown step_type: `{step.step_type}`",
                replan_hint=f"Remove unknown step `{step.step_type}` from the plan.",
                trace=_minimal_trace(step.step_type, {}, "No handler registered.", ""),
            )

        return _commit(step, step_key, result, current_index,
                       plan_state, results, history, log_lines)


# ---------------------------------------------------------------------------
# Commit + state update (executor infrastructure, not step logic)
# ---------------------------------------------------------------------------

def _commit(step, step_key, result, current_index,
            plan_state, results, history, log_lines) -> dict:
    results[step_key] = result
    history.append(_history_entry(step, result))

    status = result.get("status", "unknown")
    icon   = {"success": "✅", "failed": "❌", "fallback_used": "🚫", "over_budget": "⚠️"}.get(status, "•")
    log_lines.append(
        f"{icon} **{status}**"
        + (f" — {result['error']}" if result.get("error") else "")
    )
    if result.get("replan_hint"):
        log_lines.append(f"💡 *Hint:* {result['replan_hint']}")

    # over_budget is a soft completion (critic will handle it); only "failed" is a hard stop
    return _state_update(current_index, plan_state, results, history,
                         log_lines, feasible=(status != "failed"))


def _history_entry(step, result) -> dict:
    return {
        "step_type":   step.step_type,
        "step_id":     step.step_id,
        "status":      result.get("status", "unknown"),
        "error":       result.get("error"),
        "replan_hint": result.get("replan_hint", ""),
        "args":        result.get("trace", {}).get("args", {}),
    }


def _state_update(current_index, plan_state, results, history,
                  log_lines, feasible) -> dict:
    return {
        "current_step_index": current_index + 1,
        "itinerary_feasible": feasible,
        "itinerary_plan": {
            **plan_state,
            "step_results":      results,
            "execution_history": history,
        },
        "messages": [AIMessage(content="\n".join(log_lines), name="executor_log")],
    }
