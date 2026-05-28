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

You receive a structured trip plan summary that includes a Mode field:
  - "with_travel_data": real flights and hotel were booked by the travel agent.
  - "standalone": no booking; average market prices were used for estimation.

Your tasks:

1. Check for SEVERE structural problems only:
   - A day has fewer than 2 activities
   - No food venue is present across the entire trip
   - The schedule is logically impossible (e.g. dinner at 07:00)

2. If you find a SEVERE problem, reply ONLY with a JSON object:
   {"reject": true, "reason": "<one-sentence explanation>"}

3. If the plan is acceptable, output ONLY the final markdown itinerary
   using the format below. Do NOT wrap it in JSON.

MARKDOWN FORMAT:

# ✈️ Your [N]-Day [Destination] Itinerary

## 🛫 Flights & Accommodation
*(with_travel_data mode — show real details from OUTBOUND FLIGHT / RETURN FLIGHT / HOTEL lines)*
**Outbound:** [Airline] [Flight#] | $[price]
**Return:**   [Airline] [Flight#] | $[price]
**Hotel:** [Name] — [stars]★ | $[price]/night

*(standalone mode — show estimated averages with ~ marker)*
> 💡 **Estimated prices** (averages — no booking confirmed)
> ✈️ Outbound flight (~$[price]) · Return flight (~$[price])
> 🏨 Hotel in [destination] (~$[price]/night)

## 📅 Day 1 — [Theme]
| Time | Activity | Duration | Est. Cost |
|------|----------|----------|-----------|
| HH:MM | [icon] [Name] | X min | $Y |
...
**Day total: $XX**

[Repeat for each day]

---
## 💰 Trip Budget Summary
| Category | Cost |
|----------|------|
| Flights (outbound + return) | $X  *(or ~$X avg)* |
| Hotel ([N] nights) | $X  *(or ~$X avg)* |
| Activities | $X |
| Meals | $X |
| **Grand Total** | **$X** *(or ~$X estimated)* |

✅ Within budget  /  ⚠️ Over budget

---
*💡 Tips and local recommendations here.*
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
    lines = [
        f"Trip: {trip_days} days | Budget: ${budget or 'flexible'} | Mode: {mode}",
    ]

    # Include flight/hotel context so LLM can render the correct section
    if mode == "with_travel_data" and travel_plan:
        flights = travel_plan.get("flights", [])
        hotels  = travel_plan.get("hotels", [])
        if flights:
            f0 = flights[0]
            lines.append(f"OUTBOUND FLIGHT: {f0.get('label','')} | {f0.get('airline','')} | ${f0.get('price',0):.0f}")
            if len(flights) > 1:
                f1 = flights[1]
                lines.append(f"RETURN FLIGHT: {f1.get('label','')} | {f1.get('airline','')} | ${f1.get('price',0):.0f}")
        if hotels:
            h0 = hotels[0]
            lines.append(f"HOTEL: {h0.get('name','')} | {h0.get('stars','')}★ | ${h0.get('price_per_night',0):.0f}/night")
    else:
        # Standalone — pull average prices from verify_budget result if available
        budget_key = next((k for k in results if k.startswith("verify_budget")), None)
        if budget_key:
            b = _unwrap(results.get(budget_key, {}))
            avg = b.get("avg_prices") if isinstance(b, dict) else None
            if avg:
                lines.append(
                    f"ESTIMATED PRICES (averages, no booking): "
                    f"flight {origin}→{destination} ~${avg.get('avg_flight_price',400):.0f}, "
                    f"return ~${avg.get('avg_return_flight_price',400):.0f}, "
                    f"hotel ~${avg.get('avg_hotel_per_night',120):.0f}/night"
                )

    for key, val in results.items():
        if not isinstance(val, dict):
            continue
        if val.get("status") == "failed":
            continue
        inner = _unwrap(val)
        lines.append(f"--- {key.upper()} ---")
        lines.append(json.dumps(inner, ensure_ascii=False)[:800])
    return "\n".join(lines)


def _generate_fallback_markdown(results: dict, trip_days: int, budget: float) -> str:
    """Plain-text itinerary renderer — used when the LLM quality review fails."""
    lines = ["# ✈️ Your Trip Itinerary\n"]

    # Hotel is sourced from travel_plan (with_travel_data mode) or not shown
    # (standalone mode). Either way there is no fetch_hotels result key.
    hotel_name = None
    for key, val in results.items():
        if key.startswith("build_day_schedule"):
            inner = _unwrap(val)
            if isinstance(inner, dict) and inner.get("hotel"):
                hotel_name = inner["hotel"]
                break
    if hotel_name:
        lines.append(f"## 🏨 Accommodation\n**{hotel_name}**\n")

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
            for cat, val in b.items():
                if cat in ("grand_total", "avg_prices"):
                    continue
                if not isinstance(val, (int, float)):
                    continue
                lines.append(f"| {cat.replace('_', ' ').title()} | ${float(val):.0f} |")
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

        markdown = content if content else _generate_fallback_markdown(results, trip_days, budget)

        return {
            "itinerary_plan": {**plan_state, "final_markdown": markdown},
            "itinerary_feasible": True,
            "replanner_action":   "done",
            "messages": [AIMessage(content="✅ **Itinerary complete.**", name="replanner_log")],
        }
