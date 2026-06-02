"""
ItineraryReplannerNode
======================
Reviews the result of each Executor step and decides what to do next:

  continue  — last step succeeded, more steps remain → back to Executor
  replan    — last step failed, retries remain → back to Planner with context
  done      — all steps complete OR retries exhausted → forward to Formatter

On "done" (success path): runs an LLM quality review and generates the final
markdown itinerary. The markdown is stored in itinerary_plan["final_markdown"]
for the Formatter to render.

On "done" (failure path): sets itinerary_feasible=False and writes a
human-readable error into itinerary_fallback_reason for the Formatter.

Replan context written to itinerary_plan["replan_context"]:
  {
    "error_code":      str,   # machine-readable failure category
    "error_message":   str,   # human-readable description
    "failed_step":     str,   # step_type that failed
    "replan_hint":     str,   # specific corrective suggestion for the Planner
    "completed_steps": list,  # step_types with successful results
    "system_state":    dict,  # budget, trip_days, destination, origin snapshot
  }
"""
from __future__ import annotations

import json
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.state import AgentState

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
) -> str:
    """
    Build a compact, LLM-readable summary of the execution results.
    Avoids sending large raw JSON blobs that cause the LLM to truncate output.
    """
    lines = [
        f"TRIP: {trip_days} days | Destination: {destination} | Origin: {origin} | "
        f"Budget: ${budget or 'flexible'} | Mode: {mode}",
        "",
    ]

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
        day_cost = inner.get("day_cost", 0)
        lines.append(f"DAY {d} — {theme} | day_cost: ${day_cost:.0f}")
        for slot in inner.get("slots", []):
            t    = slot.get("time", "")
            et   = slot.get("end_time", "")
            dur  = slot.get("duration_minutes", 0)
            stype = slot.get("slot_type", "")
            name = slot.get("name", "")
            cost = slot.get("estimated_cost", 0)
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


def _generate_fallback_markdown(results: dict, trip_days: int, budget: float, mode: str = "standalone") -> str:
    """Plain-text itinerary renderer — used when the LLM quality review fails."""
    lines = ["# ✈️ Your Trip Itinerary\n"]

    # Hotel: only show in with_travel_data mode (standalone has no real hotel name).
    hotel_name = None
    for key, val in results.items():
        if key.startswith("build_day_schedule"):
            inner = _unwrap(val)
            hotel = inner.get("hotel") if isinstance(inner, dict) else None
            if hotel and hotel.strip():  # skip empty placeholder
                hotel_name = hotel
                break
    if hotel_name:
        lines.append(f"## 🏨 Accommodation\n\n**{hotel_name}**\n")

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
        day_data = _unwrap(results[key])
        lines.append(f"\n## 📅 Day {d} — {day_data.get('theme', '')}")
        lines.append("\n| Time | Activity | Duration | Est. Cost |")
        lines.append("|------|----------|----------|-----------|")
        for slot in day_data.get("slots", []):
            icon = {"activity": "🎯", "meal": "🍽️", "transport": "🚕",
                    "rest": "😴", "checkin": "🏨"}.get(slot.get("slot_type", ""), "•")
            lines.append(
                f"| {slot.get('time', '')} | {icon} {slot.get('name', '')} | "
                f"{slot.get('duration_minutes', '')} min | ${slot.get('estimated_cost', 0):.0f} |"
            )
        lines.append(f"\n**Day total: ${day_data.get('day_cost', 0):.0f}**")

    budget_md = _budget_section_md(results, budget, mode)
    if budget_md:
        lines.append(budget_md)

    return "\n".join(lines)


def _budget_section_md(results: dict, budget: float, mode: str) -> str:
    """Deterministic budget summary section — appended after the LLM day-schedule markdown."""
    budget_key = next((k for k in results if k.startswith("verify_budget")), None)
    if not budget_key:
        return ""
    b = _unwrap(results[budget_key])
    if not isinstance(b, dict) or b.get("grand_total") is None:
        return ""

    lines: list[str] = ["\n\n---------------------------------------------------------------------------------------\n"]
    lines.append("## 💰 Budget Summary\n")
    lines.append("| Category | Cost |")
    lines.append("|----------|------|")
    hotel_label_prefix = "~ " if mode == "standalone" else ""
    hotel_suffix       = " (estimated)" if mode == "standalone" else ""

    if mode == "with_travel_data":
        ob_price  = float(b.get("outbound_flight",  0) or 0)
        ret_price = float(b.get("return_flight",    0) or 0)
        if ob_price:
            lines.append(f"| Outbound Flight | ${ob_price:.0f} |")
        if ret_price:
            lines.append(f"| Return Flight | ${ret_price:.0f} |")
    elif mode == "standalone":
        avg_p = b.get("avg_prices") or {}
        out = float(avg_p.get("avg_flight_price", 0) or 0)
        ret = float(avg_p.get("avg_return_flight_price", 0) or 0)
        if out or ret:
            lines.append(f"| ~ Flights (estimated) | ${out + ret:.0f} |")

    for cat, val in b.items():
        if cat in ("grand_total", "avg_prices", "outbound_flight", "return_flight"):
            continue
        if not isinstance(val, (int, float)):
            continue
        label = cat.replace("_", " ").title()
        if "hotel" in cat.lower():
            label = f"{hotel_label_prefix}{label}{hotel_suffix}"
        lines.append(f"| {label} | ${float(val):.0f} |")

    grand = float(b.get("grand_total", 0))
    lines.append(f"\n**Grand Total: ${grand:.0f}**")
    if budget:
        remaining = budget - grand
        emoji = "✅" if remaining >= 0 else "⚠️"
        lines.append(f"\n{emoji} Budget remaining: ${remaining:.0f}")

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


def _drop_stale_budget(results: dict) -> dict:
    """Remove any verify_budget entries so the next run gets a fresh calculation."""
    return {k: v for k, v in results.items() if not k.startswith("verify_budget")}


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


def _strip_json_fences(s: str) -> str:
    if s.startswith("```"):
        parts = s.split("```")
        s = parts[1] if len(parts) > 1 else s
        s = s.lstrip("json").strip().rstrip("```").strip()
    return s


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class ItineraryReplannerNode:
    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm

    def __call__(self, state: AgentState) -> dict:
        plan_state    = state.get("itinerary_plan", {})
        results       = plan_state.get("step_results", {})
        replan_count  = plan_state.get("replan_count", 0)
        plan_steps    = plan_state.get("execution_plan", {}).get("steps", [])
        current_index = state.get("current_step_index", 0)
        feasible      = state.get("itinerary_feasible", True)
        budget        = state.get("total_budget", 0)
        trip_days     = state.get("trip_days", 3)

        # Identify the last executed step (used by all branches below)
        last_step   = plan_steps[current_index - 1] if current_index > 0 else {}
        step_type   = last_step.get("step_type", "unknown")
        step_id     = last_step.get("step_id", 0)
        last_key    = f"{step_type}_{step_id}"
        last_result = results.get(last_key, {})

        # ── A. Hard failure detected by Executor ──────────────────────────
        if not feasible:
            error_msg   = last_result.get("error", "step failed")
            replan_hint = last_result.get("replan_hint", "")

            # Hard stop
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
                    "itinerary_feasible":      False,
                    "replanner_action":        "done",
                    "itinerary_fallback_reason": hard_reason,
                    "messages": [AIMessage(
                        content=(
                            f"❌ **REPLANNER → DONE (max replans={MAX_REPLANS} reached)**\n"
                            f"*Last error:* `{step_type}` — {error_msg}"
                        ),
                        name="replanner_log",
                    )],
                }

            # Soft replan
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
                    "step_results":  _drop_stale_budget(results),
                    "replan_context": replan_ctx,
                    "replan_count":   replan_count,  # Planner increments on its side
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

        # ── B. Per-step critic for fetch results ───────────────────────────
        if step_type in _FETCH_STEP_TYPES:
            verdict = self._evaluate_step_result(step_type, last_result)
            if verdict["status"] == "failed":
                if replan_count >= MAX_REPLANS:
                    hard_reason = json.dumps({
                        "error_code":    "MAX_REPLANS",
                        "error_message": (
                            f"Gave up after {replan_count} replan attempts. "
                            f"Critic: {verdict['verdict']}"
                        ),
                        "failed_step":   step_type,
                        "replan_hint":   verdict.get("replan_hint", ""),
                    }, ensure_ascii=False)
                    return {
                        "itinerary_feasible":       False,
                        "replanner_action":         "done",
                        "itinerary_fallback_reason": hard_reason,
                        "messages": [AIMessage(
                            content=(
                                f"❌ **REPLANNER → DONE (max replans={MAX_REPLANS} reached)**\n"
                                f"*Critic:* `{step_type}` — {verdict['verdict']}"
                            ),
                            name="replanner_log",
                        )],
                    }
                replan_ctx = _build_replan_context(
                    error_code="CRITIC_REJECT",
                    error_message=verdict["verdict"],
                    failed_step=step_type,
                    replan_hint=verdict.get("replan_hint") or f"Data quality check failed for `{step_type}`.",
                    step_results=results,
                    state=state,
                )
                return {
                    "itinerary_feasible": False,
                    "replanner_action":   "replan",
                    "itinerary_plan": {
                        **plan_state,
                        "step_results":  _drop_stale_budget(results),
                        "replan_context": replan_ctx,
                        "replan_count":   replan_count,
                    },
                    "messages": [AIMessage(
                        content=(
                            f"🔄 **REPLANNER → REPLAN** (critic reject, "
                            f"attempt {replan_count}/{MAX_REPLANS})\n"
                            f"*Step:* `{step_type}`\n"
                            f"*Verdict:* {verdict['verdict']}\n"
                            f"*Hint:* {verdict.get('replan_hint', '')}"
                        ),
                        name="replanner_log",
                    )],
                }

        # ── C. More steps remain — keep going ─────────────────────────────
        if current_index < len(plan_steps):
            return {
                "replanner_action": "continue",
                "itinerary_feasible": True,
            }

        # ── D. All steps done — LLM quality review + markdown generation ──
        mode        = state.get("itinerary_mode", "standalone")
        travel_plan = state.get("travel_plan")
        origin      = state.get("current_city", "")
        destination = state.get("destination_city", "")

        # Compute budget once here so the LLM summary includes a budget table
        # and the Critic can read the result without recomputing.
        results_with_budget = dict(results)
        try:
            from agent.nodes.itinerary.step_handlers import handle_verify_budget
            budget_result = handle_verify_budget(
                results, budget, trip_days, destination, origin, mode, state,
            )
            results_with_budget["verify_budget_0"] = budget_result
        except Exception:
            pass

        try:
            # Pass plain results (no budget data) — the LLM reviews schedule quality only.
            # Budget is appended deterministically after the LLM generates day schedules.
            summary = _build_summary(
                results, budget, trip_days,
                mode=mode, travel_plan=travel_plan,
                origin=origin, destination=destination,
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
                            k: v for k, v in results.items()
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
                            "messages": [AIMessage(
                                content=(
                                    f"🔄 **REPLANNER → REPLAN** (LLM quality reject)\n"
                                    f"*Reason:* {reason}"
                                ),
                                name="replanner_log",
                            )],
                        }
                    # Out of replans — fall through to fallback markdown
                    content = ""
            except json.JSONDecodeError:
                pass  # not a rejection — treat as markdown

        # Clean up markdown fences if present
        if content.startswith("```"):
            content = content.split("```")[1].lstrip("markdown").strip().rstrip("```").strip()

        if content:
            # LLM generated day schedules + tips; append the deterministic budget table.
            budget_md = _budget_section_md(results_with_budget, budget, mode)
            markdown  = content + ("\n" + budget_md if budget_md else "")
        else:
            markdown = _generate_fallback_markdown(results_with_budget, trip_days, budget, mode)

        return {
            "itinerary_plan": {
                **plan_state,
                "final_markdown": markdown,
                # Persist the budget result so the Critic can read it without recomputing.
                "step_results": results_with_budget,
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
