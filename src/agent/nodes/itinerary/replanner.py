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
## 💰 Trip Budget Summary
| Category | Cost |
|----------|------|
| [hotel label] ([N] nights) | $[hotel_total from verify_budget] |
| Activities | $[activities_total from verify_budget] |
| Meals | $[meals_total from verify_budget] |
| **Grand Total** | **$[grand_total from verify_budget]** |

Label rules (read Mode from the TRIP header line):
  - mode=with_travel_data → hotel label = "Hotel", no flights row (already booked)
  - mode=standalone       → hotel label = "~ Hotel (estimated)"
                            AND add a "~ Flights (estimated)" row using approx_flights_estimated from BUDGET TOTALS
                            Place the flights row ABOVE the hotel row.

[✅ Within budget / ⚠️ Over budget — based on grand_total vs budget]
If mode=standalone, add this footnote after the budget status: "*Grand total covers activities, meals & estimated hotel. Flights are shown separately as an estimate.*"

---
💡 **[Destination] tips:** [Write 1-2 genuine, specific tips for this destination — never leave this as a placeholder]
"""


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

    # ── Budget totals ────────────────────────────────────────────────────────
    budget_key = next((k for k in results if k.startswith("verify_budget")), None)
    if budget_key:
        b = _unwrap(results.get(budget_key, {}))
        if isinstance(b, dict):
            totals = {k: v for k, v in b.items()
                      if k not in ("avg_prices",) and isinstance(v, (int, float))}
            # Standalone: surface estimated flight costs so the LLM can render them
            avg_p = b.get("avg_prices") or {}
            if mode == "standalone" and avg_p:
                out = float(avg_p.get("avg_flight_price", 0) or 0)
                ret = float(avg_p.get("avg_return_flight_price", 0) or 0)
                if out or ret:
                    totals["approx_flights_estimated"] = round(out + ret, 2)
            lines.append(f"BUDGET TOTALS: {json.dumps(totals)}")
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

    budget_key = next((k for k in results if k.startswith("verify_budget")), None)
    if budget_key:
        b = _unwrap(results[budget_key])
        if isinstance(b, dict) and b.get("grand_total") is not None:
            lines.append("\n---\n## 💰 Budget Summary\n")
            lines.append("| Category | Cost |")
            lines.append("|----------|------|")
            hotel_label_prefix = "~ " if mode == "standalone" else ""
            hotel_suffix = " (estimated)" if mode == "standalone" else ""
            # Standalone: prepend estimated flights row
            if mode == "standalone":
                avg_p = b.get("avg_prices") or {}
                out = float(avg_p.get("avg_flight_price", 0) or 0)
                ret = float(avg_p.get("avg_return_flight_price", 0) or 0)
                if out or ret:
                    lines.append(f"| ~ Flights (estimated) | ${out + ret:.0f} |")
            for cat, val in b.items():
                if cat in ("grand_total", "avg_prices"):
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

        # ── A. Last step failed ────────────────────────────────────────────
        if not feasible:
            last_step  = plan_steps[current_index - 1] if current_index > 0 else {}
            step_type  = last_step.get("step_type", "unknown")
            step_id    = last_step.get("step_id", 0)
            last_key   = f"{step_type}_{step_id}"
            last_result = results.get(last_key, {})

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

        # ── B. More steps remain — keep going ─────────────────────────────
        if current_index < len(plan_steps):
            return {
                "replanner_action": "continue",
                "itinerary_feasible": True,
            }

        # ── C. All steps done — LLM quality review + markdown generation ──
        mode        = state.get("itinerary_mode", "standalone")
        travel_plan = state.get("travel_plan")
        origin      = state.get("current_city", "")

        try:
            summary = _build_summary(
                results, budget, trip_days,
                mode=mode, travel_plan=travel_plan,
                origin=origin, destination=state.get("destination_city", ""),
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

        markdown = content if content else _generate_fallback_markdown(results, trip_days, budget, mode)

        return {
            "itinerary_plan": {**plan_state, "final_markdown": markdown},
            "itinerary_feasible": True,
            "replanner_action":   "done",
            "messages": [AIMessage(content="✅ **Itinerary complete.**", name="replanner_log")],
        }
