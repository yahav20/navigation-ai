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

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.core.llm import silent
from agent.core.state import AgentState
from agent.itinerary.formatter import _generate_fallback_markdown
from agent.itinerary.step_handlers import _drop_stale_budget, handle_verify_budget
from agent.shared.pricing import activity_group_price, group_label
from agent.shared.travelers import compute_default_rooms

MAX_REPLANS = 3   # must match planner.py

# ── LLM prompt: final quality review + markdown generation ────────────────

REVIEW_SYSTEM = """
You are the final travel itinerary quality reviewer AND copywriter.

STRICT RULES — violating any of these causes immediate failure:
1. NEVER use "..." to abbreviate. Every table row must be fully written out.
2. NEVER invent events, activities, or slots not present in the input data.
3. NEVER leave placeholder text like "$XX", "Tips here", or "[Theme]" in the output.
4. Output ONLY the markdown — no JSON, no preamble, no explanation.
5. Do NOT add a hotel/accommodation section — the formatter prepends it automatically.

QUALITY CHECK — before writing, verify:
- Every day has at least 2 activities (meal slots do NOT count as activities).
- At least one food/restaurant slot exists somewhere in the trip.
- No logically impossible times (e.g. dinner at 07:00, activity ending before it starts).
If a SEVERE problem is found, reply ONLY with this JSON (nothing else):
{"reject": true, "reason": "<one-sentence explanation>"}

MARKDOWN FORMAT (output this if the plan is acceptable):

# ✈️ Your [N]-Day [Destination] Itinerary

## 📅 Day 1 — [fill in the actual theme from the data]
| Time | Activity | Duration | Cost |
|------|----------|----------|------|
| HH:MM | [icon] [Actual name from data] | X min | $Y |

*(Render EVERY slot from the data for this day. Never skip rows.)*
**Day total: $[actual day_cost from data]**

[Repeat ## 📅 Day N section for every day in the trip]

---
💡 **[Destination] tips:** [Write 1-2 genuine, specific tips for this destination — never leave this as a placeholder]
"""


STEP_CRITIC_SYSTEM = """
You are a travel data quality critic evaluating the result of a single data-fetching step.

Rules:
- A result with real, usable data → "success"
- An empty result, missing key fields, or an error → "failed"

Respond ONLY as JSON (no markdown fences):
{
  "status": "success" | "failed",
  "verdict": "<one-sentence quality assessment>",
  "replan_hint": "<specific corrective action for the Planner; empty string on success>"
}
"""
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


def _build_summary(
    results: dict,
    budget: float,
    trip_days: int,
    mode: str = "standalone",
    travel_plan: Optional[dict] = None,
    origin: str = "",
    destination: str = "",
    num_adults: int = 1,
    num_children: int = 0,
) -> str:
    """
    Build a compact, LLM-readable summary of the execution results.
    Avoids sending large raw JSON blobs that cause the LLM to truncate output.
    """
    is_group = num_adults + num_children > 1
    glabel   = group_label(num_adults, num_children)
    lines = [
        f"TRIP: {trip_days} days | Destination: {destination} | Origin: {origin} | "
        f"Budget: ${budget or 'flexible'} | Mode: {mode}",
        "",
    ]
    if is_group:
        lines.append(f"GROUP_COMPOSITION: {glabel} ({num_adults} adults, {num_children} children)")
    lines.append("")
    # ── Day schedules (most important — render every slot compactly) ────────
    for d in range(1, trip_days + 1):
        key = next(
            (k for k in results
             if k.startswith("build_day_schedule")
             and isinstance(_unwrap(results[k]), dict)
             and _unwrap(results[k]).get("day") == d),
            None,
        )
        if not key:
            continue
        inner = _unwrap(results[key])
        if not isinstance(inner, dict):
            continue
        theme    = inner.get("theme", f"Day {d}")
        day_cost = float(inner.get("day_cost", 0))
        if is_group:
            day_group = activity_group_price(day_cost, num_adults, num_children)
            lines.append(f"DAY {d} — {theme} | day_cost: ${day_cost:.0f} pp / ${day_group:.0f} group")
        else:
            lines.append(f"DAY {d} — {theme} | day_cost: ${day_cost:.0f}")
        for slot in inner.get("slots", []):
            t    = slot.get("time", "")
            et   = slot.get("end_time", "")
            dur  = slot.get("duration_minutes", 0)
            stype = slot.get("slot_type", "")
            name = slot.get("name", "")
            cost = float(slot.get("estimated_cost", 0))
            if is_group:
                gcost = activity_group_price(cost, num_adults, num_children)
                lines.append(f"  {t}-{et} ({dur}min) [{stype}] {name} ${cost:.0f} pp / ${gcost:.0f} group")
            else:
                lines.append(f"  {t}-{et} ({dur}min) [{stype}] {name} ${cost:.0f}")
        lines.append("")

    # ── Weather (brief) ──────────────────────────────────────────────────────
    weather_key = next((k for k in results if k.startswith("fetch_weather")), None)
    if weather_key:
        w = _unwrap(results.get(weather_key, {}))
        if isinstance(w, dict):
            temp = w.get("temperature") or w.get("avg_temp") or ""
            season = w.get("season", "")
            if temp:
                lines.append(f"WEATHER: {season} {temp}")

    return "\n".join(lines)


def _build_critic_summary(step_type: str, data) -> str:
    """Build a concise, non-truncated summary of fetch results for the critic."""
    if data is None:
        return "null — no data returned"
    if step_type == "fetch_activities":
        activities = data.get("activities", []) if isinstance(data, dict) else data
        if not isinstance(activities, list) or not activities:
            return "empty activity list"
        names = [a.get("name", "?") for a in activities[:10]]
        return f"{len(activities)} activities fetched. Sample: {', '.join(names)}"
    if step_type == "fetch_weather":
        if isinstance(data, dict):
            pairs = [f"{k}: {v}" for k, v in data.items() if k != "error"]
            return f"Weather data: {'; '.join(pairs)}" if pairs else "empty weather dict"
        return f"weather result: {str(data)[:200]}"
    return f"{len(data)} items" if isinstance(data, list) else str(data)[:200]


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
    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = silent(llm)

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
                    "progress_log": [
                        f"❌ **REPLANNER → DONE (max replans={MAX_REPLANS} reached)**\n"
                        f"*Last error:* `{step_type}` — {error_msg}"
                    ],
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
                "progress_log": [
                    f"🔄 **REPLANNER → REPLAN** "
                    f"(attempt {replan_count}/{MAX_REPLANS})\n"
                    f"*Failed step:* `{step_type}`\n"
                    f"*Error:* {error_msg}\n"
                    f"*Hint:* {replan_hint}"
                ],
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

        try:
            summary = _build_summary(
                results, budget, trip_days,
                mode=mode, travel_plan=travel_plan,
                origin=origin, destination=destination,
                num_adults=int(state.get("num_adults") or 1),
                num_children=int(state.get("num_children") or 0),
            )
            output  = self.llm.invoke([
                SystemMessage(content=REVIEW_SYSTEM),
                HumanMessage(content=summary),
            ])
            content = output.content.strip()
        except Exception:
            content = ""

        # JSON rejection
        if content.startswith("{"):
            try:
                rejection = json.loads(content)
                if rejection.get("reject"):
                    reason = rejection.get("reason", "LLM quality check failed")
                    if replan_count < MAX_REPLANS:
                        # For quality rejects we must rebuild the day schedules.
                        # Remove build_day_schedule from completed_steps so the
                        # Planner re-emits those steps with the corrective hint.
                        results_for_replan = {
                            k: v for k, v in results_with_budget.items()
                            if not k.startswith("build_day_schedule")
                            and not k.startswith("verify_budget")
                        }
                        replan_ctx = _build_replan_context(
                            error_code="LLM_REJECT",
                            error_message=reason,
                            failed_step="build_day_schedule",
                            replan_hint=(
                                f"LLM rejected the plan: {reason}. "
                                "Rebuild ALL day schedules. Each day MUST include "
                                "at least 2 sightseeing activities AND at least 1 "
                                "food/restaurant venue. ActivitySelector must assign "
                                "food venues to every day."
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
                                # Drop stale day schedules so executor rebuilds them
                                "step_results": results_for_replan,
                            },
                            "progress_log": [
                                f"🔄 **REPLANNER → REPLAN** (LLM quality reject)\n"
                                f"*Reason:* {reason}"
                            ],
                        }
                    # Out of replans — fall through to fallback markdown
                    content = ""
            except json.JSONDecodeError:
                pass  # not a rejection — treat as markdown

        # Clean up markdown fences if present
        if content.startswith("```"):
            content = content.split("```")[1].lstrip("markdown").strip().rstrip("```").strip()

        if content:
            markdown = content
        else:
            markdown = _generate_fallback_markdown(results_with_budget, trip_days, budget, mode, state=state)

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

    # ── Per-step critic ────────────────────────────────────────────────────

    def _evaluate_step_result(self, step_type: str, result: dict) -> dict:
        """LLM critic that evaluates the quality of a fetch step's output."""
        data    = result.get("data")
        summary = _build_critic_summary(step_type, data)
        prompt  = f"Step: {step_type}\nResult summary: {summary}"
        try:
            raw    = self.llm.invoke([
                SystemMessage(content=STEP_CRITIC_SYSTEM),
                HumanMessage(content=prompt),
            ]).content.strip()
            parsed = json.loads(_strip_json_fences(raw))
            if isinstance(parsed, dict) and "status" in parsed:
                return parsed
        except Exception:
            pass
        # Fallback: deterministic empty check
        empty = _is_result_empty(data)
        return {
            "status":      "failed" if empty else "success",
            "verdict":     "LLM critic unavailable — falling back to empty check.",
            "replan_hint": f"`{step_type}` returned no usable data." if empty else "",
        }
