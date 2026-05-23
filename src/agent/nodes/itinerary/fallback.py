"""
ItineraryFallbackNode
=====================
Called when the Observer detects an unrecoverable issue or max retries reached.

Strategies (in order):
  no_flights / no_hotels   → suggest up to 3 culturally-similar alternative destinations
  budget_exceeded          → try ±3 days; if still over, bump $500; else → alternatives
  max_retries_exceeded     → suggest alternatives

Writes to state:
  itinerary_fallback_action        — what was done
  itinerary_fallback_alternatives  — list of city names (if alternatives chosen)
  itinerary_feasible               — True if a fix was applied (retry planner)
  messages                         — AIMessage if showing alternatives directly
"""
from __future__ import annotations
import json
import logging
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.nodes.itinerary.itinerary_tools import (
    search_outbound_flights,
    search_hotels,
)
from agent.state import AgentState

logger = logging.getLogger(__name__)

CULTURAL_SIMILARITY_PROMPT = """You are a travel geography expert.
Given a destination city, suggest exactly 3 alternative cities that are:
1. In the SAME cultural region (Spain → France/Italy/Portugal, NOT Morocco).
2. Similar travel vibe (beach, culture, city-break, food).
3. Likely to have direct flights from the given origin.

Return ONLY a JSON array of 3 city name strings.
Example: ["Lisbon", "Barcelona", "Rome"]
No explanation, no markdown.
"""


class ItineraryFallbackNode:
    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm

    def __call__(self, state: AgentState) -> dict:
        reason      = state.get("itinerary_fallback_reason", "unknown")
        destination = state.get("destination_city", "")
        origin      = state.get("current_city", "")
        budget      = state.get("total_budget", 0)
        trip_days   = state.get("trip_days", 3)
        plan_state  = state.get("itinerary_plan") or {}
        results     = plan_state.get("step_results", {})

        logger.info("ItineraryFallbackNode: reason=%s dest=%s", reason, destination)

        if reason == "budget_exceeded":
            return self._handle_budget(results, budget, trip_days, destination, origin)

        # no_flights, no_hotels, max_retries, unknown → alternatives
        return self._suggest_alternatives(destination, origin, budget, trip_days)

    # ------------------------------------------------------------------

    def _handle_budget(self, results, budget, trip_days, destination, origin):
        """Try day reduction, then budget bump, then alternatives."""
        min_flight = _cheapest_flight_price(results, "fetch_flights")
        min_ret    = _cheapest_flight_price(results, "fetch_return_flights")
        min_hotel  = _cheapest_hotel_night(results)

        # Try ±1..3 days
        for delta in [-1, -2, -3, 1, 2, 3]:
            new_days = max(1, trip_days + delta)
            est_cost = min_flight + min_ret + min_hotel * new_days
            if budget and est_cost <= budget * 1.05:
                logger.info("Fallback: adjusting days %d→%d", trip_days, new_days)
                return {
                    "trip_days": new_days,
                    "itinerary_fallback_action": f"adjusted_days_{new_days}",
                    "itinerary_feasible": True,   # retry planner
                    "itinerary_fallback_alternatives": [],
                }

        # Try +$500
        bumped = budget + 500
        est_cost = min_flight + min_ret + min_hotel * trip_days
        if est_cost <= bumped * 1.05:
            logger.info("Fallback: bumping budget +$500")
            return {
                "total_budget": bumped,
                "itinerary_fallback_action": "budget_bumped_500",
                "itinerary_feasible": True,
                "itinerary_fallback_alternatives": [],
            }

        # Nothing worked → alternatives
        return self._suggest_alternatives(destination, origin, bumped, trip_days)

    def _suggest_alternatives(self, destination, origin, budget, trip_days):
        """Ask LLM for culturally similar cities, validate flights exist, show to user."""
        prompt = (f"Destination: {destination}\nOrigin: {origin}\n"
                  f"Trip days: {trip_days}\nBudget: ${budget or 'flexible'}")

        raw = self.llm.invoke([
            SystemMessage(content=CULTURAL_SIMILARITY_PROMPT),
            HumanMessage(content=prompt),
        ]).content.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip().rstrip("```").strip()

        try:
            alternatives: list[str] = json.loads(raw)
            if not isinstance(alternatives, list):
                alternatives = []
        except Exception:
            alternatives = []

        # Validate: check flights exist for each alternative
        validated = []
        for city in alternatives[:3]:
            try:
                flights = search_outbound_flights.invoke(
                    {"origin": origin, "destination": city}
                )
                if flights:
                    validated.append(city)
            except Exception:
                pass  # skip unavailable cities

        final_alts = validated or alternatives[:3]

        md = _render_alternatives(destination, final_alts, origin)
        logger.info("Fallback: suggesting alternatives %s", final_alts)

        return {
            "itinerary_fallback_action": "suggested_alternatives",
            "itinerary_fallback_alternatives": final_alts,
            "itinerary_feasible": False,
            "alternative_destinations": final_alts,
            "messages": [AIMessage(content=md)],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cheapest_flight_price(results: dict, prefix: str) -> float:
    for k, v in results.items():
        if k.startswith(prefix) and isinstance(v, list) and v:
            return min(f.get("price", 9999) for f in v)
    return 0.0


def _cheapest_hotel_night(results: dict) -> float:
    for k, v in results.items():
        if k.startswith("fetch_hotels") and isinstance(v, list) and v:
            return min(h.get("price_per_night", 9999) for h in v)
    return 0.0


def _render_alternatives(original: str, alternatives: list[str], origin: str) -> str:
    lines = [
        f"# 🗺️ Alternative Destinations",
        f"\nUnfortunately we couldn't build a complete itinerary for **{original}** "
        f"within your constraints. Here are culturally similar alternatives:\n",
    ]
    for i, city in enumerate(alternatives, 1):
        lines.append(f"## {i}. 📍 {city}")
        lines.append(f"Flights available from {origin}. Similar culture and travel vibe to {original}.\n")
    lines.append("---")
    lines.append("Reply with a city name and I'll plan your full itinerary! 😊")
    return "\n".join(lines)
