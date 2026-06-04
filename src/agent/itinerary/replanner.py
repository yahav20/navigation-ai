"""
ItineraryReplannerNode
======================
All quality checks are deterministic — no LLM calls in this node.

Reviews the result of each Executor step and decides what to do next:

  continue  — last step succeeded, more steps remain → back to Executor
  replan    — step failed or failed quality check → back to Planner
  done      — all steps complete → forward to Critic / Formatter

Quality checks (all deterministic):
  fetch_activities / fetch_weather: non-empty result
  build_day_schedule: at least 1 activity slot per day

On "done": computes budget roll-up and generates markdown via the
deterministic _generate_fallback_markdown template.

Replan context written to itinerary_plan["replan_context"]:
  {
    "error_code":      str,
    "error_message":   str,
    "failed_step":     str,
    "replan_hint":     str,
    "completed_steps": list,
    "system_state":    dict,
  }
"""
from __future__ import annotations

import json

from langchain_core.messages import AIMessage

from agent.core.state import AgentState
from agent.itinerary.formatter import _generate_fallback_markdown
from agent.itinerary.step_handlers import _drop_stale_budget, handle_verify_budget

MAX_REPLANS = 3

_MIN_ACTIVITIES_PER_DAY = 1
_FETCH_STEP_TYPES = {"fetch_activities", "fetch_weather"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _completed_step_types(step_results: dict) -> list[str]:
    completed: list[str] = []
    for key, val in step_results.items():
        if not isinstance(val, dict) or val.get("status") != "success":
            continue
        parts = key.rsplit("_", 1)
        step_type = parts[0] if len(parts) == 2 and parts[1].isdigit() else key
        if step_type and step_type not in completed:
            completed.append(step_type)
    return completed


def _build_replan_context(
    error_code: str,
    error_message: str,
    failed_step: str,
    replan_hint: str,
    step_results: dict,
    state: AgentState,
) -> str:
    ctx = {
        "error_code":      error_code,
        "error_message":   error_message,
        "failed_step":     failed_step,
        "replan_hint":     replan_hint,
        "completed_steps": _completed_step_types(step_results),
        "system_state": {
            "budget":      state.get("total_budget", 0),
            "trip_days":   state.get("trip_days", 3),
            "destination": state.get("destination_city", ""),
            "origin":      state.get("current_city", ""),
        },
    }
    return json.dumps(ctx, ensure_ascii=False)


def _unwrap(val: dict) -> dict:
    if not isinstance(val, dict):
        return {}
    if "data" in val and isinstance(val["data"], dict):
        return val["data"]
    return val


def _is_result_empty(val) -> bool:
    if val is None:
        return True
    if isinstance(val, list):
        return len(val) == 0
    if isinstance(val, dict):
        if val.get("error"):
            return True
        return not any(v for k, v in val.items() if k != "error")
    return False


def _validate_day_quality(day_data: dict) -> tuple[bool, str]:
    """Deterministic quality check for a single built day."""
    slots = day_data.get("slots", [])
    if not slots:
        return False, "No slots were generated for this day."
    activity_count = sum(1 for s in slots if s.get("slot_type") == "activity")
    if activity_count < _MIN_ACTIVITIES_PER_DAY:
        return False, (
            f"Day has {activity_count} scheduled activities "
            f"(minimum {_MIN_ACTIVITIES_PER_DAY} required)."
        )
    return True, ""


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class ItineraryReplannerNode:
    def __call__(self, state: AgentState) -> dict:
        plan_state    = state.get("itinerary_plan", {})
        results       = plan_state.get("step_results", {})
        replan_count  = plan_state.get("replan_count", 0)
        plan_steps    = plan_state.get("execution_plan", {}).get("steps", [])
        current_index = state.get("current_step_index", 0)
        feasible      = state.get("itinerary_feasible", True)
        budget        = state.get("total_budget", 0)
        trip_days     = state.get("trip_days", 3)

        last_step   = plan_steps[current_index - 1] if (0 < current_index <= len(plan_steps)) else {}
        step_type   = last_step.get("step_type", "unknown")
        step_id     = last_step.get("step_id", 0)
        last_key    = f"{step_type}_{step_id}"
        last_result = results.get(last_key, {})

        # ── A. Hard failure detected by Executor ──────────────────────────
        if not feasible:
            error_msg   = last_result.get("error", "step failed")
            replan_hint = last_result.get("replan_hint", "")

            if replan_count >= MAX_REPLANS:
                hard_reason = json.dumps({
                    "error_code":    "MAX_REPLANS",
                    "error_message": (
                        f"Gave up after {replan_count} replan attempts. "
                        f"Last error: {error_msg}"
                    ),
                    "failed_step":   step_type,
                    "replan_hint":   replan_hint,
                }, ensure_ascii=False)
                return {
                    "itinerary_feasible":       False,
                    "replanner_action":         "done",
                    "itinerary_fallback_reason": hard_reason,
                    "messages": [AIMessage(
                        content=(
                            f"❌ **REPLANNER → DONE (max replans={MAX_REPLANS} reached)**\n"
                            f"*Last error:* `{step_type}` — {error_msg}"
                        ),
                        name="replanner_log",
                    )],
                }

            replan_ctx = _build_replan_context(
                error_code="STEP_ERROR",
                error_message=error_msg,
                failed_step=step_type,
                replan_hint=replan_hint or f"`{step_type}` failed. Retry or skip.",
                step_results=results,
                state=state,
            )
            return {
                "itinerary_feasible": False,
                "replanner_action":   "replan",
                "itinerary_plan": {
                    **plan_state,
                    "step_results":   _drop_stale_budget(results),
                    "replan_context": replan_ctx,
                    "replan_count":   replan_count,
                },
                "messages": [AIMessage(
                    content=(
                        f"🔄 **REPLANNER → REPLAN** "
                        f"(attempt {replan_count}/{MAX_REPLANS})\n"
                        f"*Failed step:* `{step_type}`\n"
                        f"*Error:* {error_msg}\n"
                        f"*Hint:* {replan_hint}"
                    ),
                    name="replanner_log",
                )],
            }

        # ── B. Deterministic fetch-step quality check ──────────────────────
        if step_type in _FETCH_STEP_TYPES:
            data  = last_result.get("data")
            if _is_result_empty(data):
                replan_hint = f"`{step_type}` returned no usable data."
                if replan_count >= MAX_REPLANS:
                    hard_reason = json.dumps({
                        "error_code":    "MAX_REPLANS",
                        "error_message": (
                            f"Gave up after {replan_count} replan attempts. "
                            f"`{step_type}` returned empty data."
                        ),
                        "failed_step":   step_type,
                        "replan_hint":   replan_hint,
                    }, ensure_ascii=False)
                    return {
                        "itinerary_feasible":       False,
                        "replanner_action":         "done",
                        "itinerary_fallback_reason": hard_reason,
                        "messages": [AIMessage(
                            content=(
                                f"❌ **REPLANNER → DONE (max replans={MAX_REPLANS} reached)**\n"
                                f"*Step:* `{step_type}` returned empty data."
                            ),
                            name="replanner_log",
                        )],
                    }
                replan_ctx = _build_replan_context(
                    error_code="EMPTY_DATA",
                    error_message=f"`{step_type}` returned no usable data.",
                    failed_step=step_type,
                    replan_hint=replan_hint,
                    step_results=results,
                    state=state,
                )
                return {
                    "itinerary_feasible": False,
                    "replanner_action":   "replan",
                    "itinerary_plan": {
                        **plan_state,
                        "step_results":   _drop_stale_budget(results),
                        "replan_context": replan_ctx,
                        "replan_count":   replan_count,
                    },
                    "messages": [AIMessage(
                        content=(
                            f"🔄 **REPLANNER → REPLAN** (empty data, "
                            f"attempt {replan_count}/{MAX_REPLANS})\n"
                            f"*Step:* `{step_type}` returned no usable data."
                        ),
                        name="replanner_log",
                    )],
                }

        # ── C. Per-day quality check for build_day_schedule ───────────────
        if step_type == "build_day_schedule":
            day_data = _unwrap(last_result)
            day_num  = day_data.get("day", "?")
            valid, reason = _validate_day_quality(day_data)
            if not valid:
                if replan_count >= MAX_REPLANS:
                    hard_reason = json.dumps({
                        "error_code":    "MAX_REPLANS",
                        "error_message": f"Day {day_num} quality check failed: {reason}",
                        "failed_step":   "build_day_schedule",
                        "replan_hint":   "",
                    }, ensure_ascii=False)
                    return {
                        "itinerary_feasible":       False,
                        "replanner_action":         "done",
                        "itinerary_fallback_reason": hard_reason,
                        "messages": [AIMessage(
                            content=(
                                f"❌ **REPLANNER → DONE (max replans={MAX_REPLANS} reached)**\n"
                                f"*Day {day_num}:* {reason}"
                            ),
                            name="replanner_log",
                        )],
                    }
                # Drop all day results + fetch_activities so ActivitySelector reruns
                results_for_replan = {
                    k: v for k, v in results.items()
                    if not k.startswith("build_day_schedule")
                    and not k.startswith("fetch_activities")
                    and not k.startswith("verify_budget")
                }
                replan_ctx = _build_replan_context(
                    error_code="DAY_QUALITY_FAIL",
                    error_message=f"Day {day_num}: {reason}",
                    failed_step="build_day_schedule",
                    replan_hint=(
                        f"Day {day_num} failed quality check: {reason}. "
                        "Re-run fetch_activities so the ActivitySelector can assign "
                        "more activities across all days, then rebuild all day schedules."
                    ),
                    step_results=results_for_replan,
                    state=state,
                )
                return {
                    "itinerary_feasible": False,
                    "replanner_action":   "replan",
                    "itinerary_plan": {
                        **plan_state,
                        "replan_context": replan_ctx,
                        "replan_count":   replan_count,
                        "step_results":   results_for_replan,
                    },
                    "messages": [AIMessage(
                        content=(
                            f"🔄 **REPLANNER → REPLAN** (day quality check)\n"
                            f"*Day {day_num}:* {reason}"
                        ),
                        name="replanner_log",
                    )],
                }

        # ── D. More steps remain — continue ───────────────────────────────
        if current_index < len(plan_steps):
            return {
                "replanner_action":   "continue",
                "itinerary_feasible": True,
            }

        # ── E. All steps done — compute budget + generate markdown ────────
        mode        = state.get("itinerary_mode", "standalone")
        origin      = state.get("current_city", "")
        destination = state.get("destination_city", "")

        results_with_budget = dict(results)
        try:
            budget_result = handle_verify_budget(
                results, budget, trip_days, destination, origin, mode, state,
            )
            results_with_budget["verify_budget_0"] = budget_result
        except Exception:
            pass

        markdown = _generate_fallback_markdown(results_with_budget, trip_days, budget, mode)

        return {
            "itinerary_plan": {
                **plan_state,
                "final_markdown": markdown,
                "step_results":   results_with_budget,
            },
            "itinerary_feasible": True,
            "replanner_action":   "done",
            "messages": [AIMessage(content="✅ **Itinerary complete.**", name="replanner_log")],
        }
