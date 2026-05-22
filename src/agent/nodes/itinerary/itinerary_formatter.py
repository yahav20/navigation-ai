"""
ItineraryFormatterNode — Plan & Execute, Step 3: FORMAT

Renders the enriched itinerary dict to a structured Markdown string
that is ready for display in the chat UI.

Two modes:
  1. FULL ITINERARY  — plan was feasible, renders the day-by-day schedule.
  2. ALTERNATIVES    — plan was not feasible, renders 2-3 alternative destinations
                       with a brief explanation why the original was rejected.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from agent.state import AgentState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Emoji map for slot types
# ---------------------------------------------------------------------------

TYPE_EMOJI = {
    "activity": "🎯",
    "meal": "🍽️",
    "rest": "🛌",
    "transport": "🚌",
    "hotel": "🏨",
}

FALLBACK_ACTION_LABELS = {
    "adjusted_days_1": "We shortened your trip by 1 day to fit your budget.",
    "adjusted_days_2": "We shortened your trip by 2 days to fit your budget.",
    "adjusted_days_3": "We shortened your trip by 3 days to fit your budget.",
    "adjusted_days_4": "We extended your trip by 1 day for a better experience.",
    "adjusted_days_5": "We extended your trip by 2 days for a better experience.",
    "adjusted_days_6": "We extended your trip by 3 days for a better experience.",
    "budget_bumped_500": "We increased the budget estimate by $500 to include the best options for you.",
    "relaxed_kosher_for_hotels": "No kosher-certified hotels were available; we've listed the closest alternatives.",
    "suggested_alternatives": "Unfortunately no direct route or matching options were found. Here are culturally similar destinations:",
}


# ---------------------------------------------------------------------------
# Formatter node
# ---------------------------------------------------------------------------

class ItineraryFormatterNode:
    """
    Pure rendering node — no LLM calls for the happy path.
    Uses LLM only to generate the alternatives explanation blurb (short).
    """

    ALT_SYSTEM_PROMPT = """You are a friendly travel assistant.
Write 2-3 sentences explaining why the original destination wasn't available
and introducing the alternative options. Be warm, brief, and helpful.
Do NOT use markdown headers. Just a short paragraph."""

    def __init__(self, response_model: BaseChatModel) -> None:
        """Store the response model used to format the fallback text."""
        self.response_model = response_model

    def __call__(self, state: AgentState) -> dict:
        plan = state.get("itinerary_plan", {})
        feasible = state.get("itinerary_feasible", False)
        fallback_action = state.get("itinerary_fallback_action", "")
        alternatives = state.get("itinerary_fallback_alternatives", [])
        reason = state.get("itinerary_fallback_reason", "")

        if not feasible and alternatives:
            md = self._render_alternatives(
                original_destination=state.get("destination_city", ""),
                reason=reason,
                alternatives=alternatives,
                fallback_action=fallback_action,
                state=state,
            )
        else:
            md = self._render_itinerary(plan, fallback_action)

        return {
            "messages": [AIMessage(content=md)],
            "itinerary_formatted": md,
        }

    # ------------------------------------------------------------------
    # Render: full itinerary
    # ------------------------------------------------------------------

    def _render_itinerary(self, plan: dict, fallback_action: str) -> str:
        if not plan or plan.get("error"):
            return "⚠️ Sorry, I wasn't able to build your itinerary. Please try again."

        lines = []

        # Header
        dest = plan.get("destination", "")
        origin = plan.get("origin", "")
        total_days = plan.get("total_days", 0)
        total_cost = plan.get("estimated_total_cost", 0)
        prefs_applied = plan.get("user_preferences_applied", [])

        lines.append(f"# ✈️ Your {total_days}-Day Trip to {dest}")
        lines.append(f"*From {origin} · Estimated total cost: **${total_cost:,.0f}***")

        if fallback_action and fallback_action in FALLBACK_ACTION_LABELS:
            lines.append(f"\n> 💡 {FALLBACK_ACTION_LABELS[fallback_action]}")

        if prefs_applied:
            prefs_str = " · ".join(f"✅ {p.capitalize()}" for p in prefs_applied)
            lines.append(f"\n{prefs_str}")

        lines.append("")

        # Flight summary
        flight = plan.get("selected_flight", {})
        if flight:
            lines.append("## ✈️ Flight")
            lines.append(
                f"**{flight.get('airline', '')} {flight.get('flight_number', '')}** · "
                f"${flight.get('price', 0):,.0f}  \n"
                f"🛫 Departure: `{flight.get('departure_time', '')}` → "
                f"🛬 Arrival: `{flight.get('arrival_time', '')}`"
            )
            lines.append("")

        # Hotel summary
        hotel = plan.get("selected_hotel", {})
        if hotel:
            stars = "⭐" * hotel.get("stars", 0)
            breakfast = "🍳 Breakfast included" if hotel.get("breakfast_available") else "🚫 No breakfast"
            lines.append("## 🏨 Hotel")
            lines.append(
                f"**{hotel.get('name', '')}** {stars}  \n"
                f"${hotel.get('price_per_night', 0):,.0f}/night · {breakfast}"
            )
            lines.append("")

        # Day-by-day schedule
        for day in plan.get("days", []):
            lines.append(f"---\n## 📅 Day {day['day']}: {day.get('theme', '')}")
            lines.append("")

            for slot in day.get("slots", []):
                emoji = TYPE_EMOJI.get(slot.get("type", "activity"), "📍")
                time_str = slot.get("time", "")
                name = slot.get("name", "")
                desc = slot.get("description", "")
                cost = slot.get("estimated_cost", 0)
                duration = slot.get("duration_minutes", 0)
                notes = slot.get("notes", "")

                cost_str = f"· 💰 ${cost:,.0f}" if cost else "· 🆓 Free"
                dur_str = f"· ⏱ {duration} min" if duration else ""

                lines.append(f"### {emoji} `{time_str}` — {name}")
                if desc:
                    lines.append(f"{desc}")
                lines.append(f"*{cost_str} {dur_str}*")
                if notes:
                    lines.append(f"> {notes}")
                lines.append("")

        # Footer: cost breakdown
        lines.append("---")
        lines.append("## 💳 Cost Summary")
        flight_cost = flight.get("price", 0)
        hotel_nights = plan.get("total_days", 1)
        hotel_cost = hotel.get("price_per_night", 0) * hotel_nights
        activity_cost = total_cost - flight_cost - hotel_cost

        lines.append(f"| Item | Cost |")
        lines.append(f"|------|------|")
        lines.append(f"| ✈️ Flight | ${flight_cost:,.0f} |")
        lines.append(f"| 🏨 Hotel ({hotel_nights} nights) | ${hotel_cost:,.0f} |")
        lines.append(f"| 🎯 Activities & meals | ${activity_cost:,.0f} |")
        lines.append(f"| **Total** | **${total_cost:,.0f}** |")

        lines.append("\n*Prices are estimates. Always verify before booking.*")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Render: alternatives
    # ------------------------------------------------------------------

    def _render_alternatives(
        self,
        original_destination: str,
        reason: str,
        alternatives: list[str],
        fallback_action: str,
        state: AgentState,
    ) -> str:
        lines = []

        # Generate blurb with LLM if available
        blurb = ""
        if self.response_model:
            try:
                prompt = (
                    f"Original destination: {original_destination}\n"
                    f"Reason unavailable: {reason}\n"
                    f"Alternatives offered: {', '.join(alternatives)}"
                )
                response = self.response_model.invoke([
                    SystemMessage(content=self.ALT_SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ])
                blurb = response.content.strip()
            except Exception as e:
                logger.warning("Alternatives blurb generation failed: %s", e)

        if not blurb:
            reason_msg = {
                "no_flights": f"no direct flights are currently available to {original_destination}",
                "budget_exceeded": f"the trip to {original_destination} exceeded your budget",
                "no_hotels": f"no suitable hotels were found in {original_destination}",
            }.get(reason, f"{original_destination} wasn't available for your dates")
            blurb = (
                f"Unfortunately, {reason_msg}. "
                f"Here are some culturally similar destinations you might love instead:"
            )

        lines.append(f"# 🗺️ Alternative Destinations")
        lines.append(f"\n{blurb}\n")

        for i, city in enumerate(alternatives, 1):
            lines.append(f"## {i}. 📍 {city}")
            lines.append(
                f"Similar culture and travel vibe to {original_destination}. "
                f"Direct flights available from your origin."
            )
            lines.append("")

        lines.append("---")
        lines.append(
            "Would you like me to plan a full itinerary for one of these destinations? "
            "Just let me know which one! 😊"
        )

        return "\n".join(lines)
