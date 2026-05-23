"""
ItineraryObserverNode — Plan & Execute, Step 3: OBSERVE

Reviews the assembled itinerary produced by the Executor.
No LLM calls — pure validation + Markdown rendering.

Outcomes:
  - Issues found + retries remaining  → signals re-plan (itinerary_planner)
  - Budget exceeded, no more retries  → signals fallback (itinerary_fallback)
  - All good                          → renders Markdown, goes to summary
"""
from __future__ import annotations
import logging
from langchain_core.messages import AIMessage
from agent.state import AgentState

logger = logging.getLogger(__name__)

MAX_RETRIES = 2

TYPE_EMOJI = {"activity": "🎯", "meal": "🍽️", "rest": "🛌", "transport": "🚌"}


class ItineraryObserverNode:
    """Validate, optionally re-plan, or render the final itinerary."""

    def __call__(self, state: AgentState) -> dict:
        plan_state = state.get("itinerary_plan", {})
        assembled  = plan_state.get("assembled", {})
        retries    = plan_state.get("observer_retries", 0)

        if not assembled:
            return {
                "itinerary_feasible": False,
                "itinerary_fallback_reason": "no_flights",
            }

        issues = _find_issues(assembled, state)

        if issues and retries < MAX_RETRIES:
            logger.info("Observer: issues=%s retry=%d — triggering re-plan", issues, retries + 1)
            return {
                "itinerary_plan": {
                    **plan_state,
                    "assembled": {},            # clear so planner re-runs cleanly
                    "observer_issues": issues,
                    "observer_retries": retries + 1,
                },
                "itinerary_feasible": False,
                "itinerary_fallback_reason": issues[0],
            }

        if issues and not assembled.get("within_budget", True):
            # Retries exhausted and still over budget → fallback
            return {
                "itinerary_feasible": False,
                "itinerary_fallback_reason": "budget_exceeded",
            }

        # ── All good: render ────────────────────────────────────────────
        md = _render(assembled)
        return {
            "itinerary_plan": {**plan_state, "final_markdown": md},
            "itinerary_feasible": True,
            "messages": [AIMessage(content=md)],
        }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _find_issues(assembled: dict, state: AgentState) -> list[str]:
    issues = []
    budget = state.get("total_budget", 0)
    if budget and not assembled.get("within_budget", True):
        issues.append("budget_exceeded")
    if len(assembled.get("days", [])) < assembled.get("total_days", 0):
        issues.append("missing_days")
    return issues


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------

def _render(plan: dict) -> str:
    if not plan:
        return "⚠️ Could not build your itinerary. Please try again."

    lines = []
    dest       = plan.get("destination", "")
    origin     = plan.get("origin", "")
    total_days = plan.get("total_days", 0)
    total_cost = plan.get("estimated_total_cost", 0)
    prefs      = plan.get("user_preferences_applied", [])

    lines.append(f"# ✈️ Your {total_days}-Day Trip to {dest}")
    lines.append(f"*From {origin} · Estimated total: **${total_cost:,.0f}***\n")

    if prefs:
        lines.append(" · ".join(f"✅ {p.capitalize()}" for p in prefs) + "\n")

    # Outbound flight
    f = plan.get("selected_flight", {})
    if f and f.get("flight_number"):
        lines.append("## ✈️ Outbound Flight")
        lines.append(
            f"**{f.get('airline', '')} {f.get('flight_number', '')}**"
            f" · ${f.get('price', 0):,.0f}  \n"
            f"🛫 Departure: `{f.get('departure_time') or 'N/A'}`"
            f" → 🛬 Arrival: `{f.get('arrival_time') or 'N/A'}`\n"
        )

    # Return flight
    rf = plan.get("selected_return_flight", {})
    if rf and rf.get("flight_number"):
        lines.append("## 🔄 Return Flight")
        lines.append(
            f"**{rf.get('airline', '')} {rf.get('flight_number', '')}**"
            f" · ${rf.get('price', 0):,.0f}  \n"
            f"🛫 Departure: `{rf.get('departure_time') or 'N/A'}`"
            f" → 🛬 Arrival: `{rf.get('arrival_time') or 'N/A'}`\n"
        )

    # Hotel
    h = plan.get("selected_hotel", {})
    if h and h.get("name"):
        stars = "⭐" * h.get("stars", 0)
        bk    = " · 🍳 Breakfast included" if h.get("breakfast_available") else ""
        lines.append("## 🏨 Hotel")
        lines.append(f"**{h.get('name', '')}** {stars} · ${h.get('price_per_night', 0):,.0f}/night{bk}\n")

    # Day-by-day
    for day in plan.get("days", []):
        lines.append(f"---\n## 📅 Day {day['day']}: {day.get('theme', '')}\n")
        for slot in day.get("slots", []):
            emoji    = TYPE_EMOJI.get(slot.get("slot_type", ""), "📍")
            cost_str = f"💰 ${slot['estimated_cost']:,.0f}" if slot.get("estimated_cost") else "🆓 Free"
            lines.append(f"### {emoji} `{slot.get('time', '')}` — {slot.get('name', '')}")
            if slot.get("description"):
                lines.append(slot["description"])
            lines.append(f"*⏱ {slot.get('duration_minutes', '')} min · {cost_str}*")
            if slot.get("notes"):
                lines.append(f"> {slot['notes']}")
            lines.append("")

    # Cost summary
    cb = plan.get("cost_breakdown", {})
    if cb:
        lines.append("---\n## 💳 Cost Summary\n")
        lines.append("| | Cost |")
        lines.append("|---|---|")
        lines.append(f"| ✈️ Outbound flight | ${cb.get('flight_cost', 0):,.0f} |")
        lines.append(f"| 🏨 Hotel ({total_days} nights) | ${cb.get('hotel_cost', 0):,.0f} |")
        lines.append(f"| 🎯 Activities & meals | ${cb.get('activity_cost', 0):,.0f} |")
        lines.append(f"| **Total** | **${cb.get('total_cost', 0):,.0f}** |")

    lines.append("\n*Prices are estimates. Verify before booking.*")
    return "\n".join(lines)