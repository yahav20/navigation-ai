# src/agent/nodes/itinerary/fallback.py
"""
ItineraryFallbackNode — v2
==========================
Called when Observer detects an unrecoverable issue or MAX_RETRIES reached.

Strategy waterfall:
  budget_exceeded  → try ±1..3 days adjustment
                   → if still over, bump $500
                   → if still impossible, suggest alternatives
  no_flights       → suggest culturally-similar alternatives
  no_hotels        → suggest culturally-similar alternatives
  max_retries      → suggest alternatives
  unknown          → suggest alternatives

State updates:
  itinerary_fallback_action        — what was done
  itinerary_fallback_alternatives  — list[str] of city names
  itinerary_feasible               — True if a retry is warranted
  messages                         — AIMessage for user-facing output
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.nodes.itinerary.itinerary_tools import search_outbound_flights, search_hotels
from agent.state import AgentState

logger = logging.getLogger(__name__)

CULTURAL_SIMILARITY_PROMPT = """You are a travel geography expert.
Given a destination city, suggest exactly 3 alternative cities that are:
1. In the SAME cultural/geographic region.
2. Similar travel vibe (beach, culture, city-break, food, nature).
3. Likely to have direct or easy connecting flights from the given origin.
4. Within a similar price range.

Return ONLY a JSON array of 3 city name strings.
Example: ["Lisbon", "Barcelona", "Rome"]
No explanation, no markdown, no extra keys.
"""


class ItineraryFallbackNode:
    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm

    def __call__(self, state: AgentState) -> dict:
        reason = state.get("itinerary_fallback_reason", "unknown")
        destination = state.get("destination_city", "")
        origin = state.get("current_city", "")
        budget = state.get("total_budget", 0)
        trip_days = state.get("trip_days", 3)
        plan_state = state.get("itinerary_plan") or {}
        results = plan_state.get("step_results", {})

        logger.info("Fallback triggered: reason=%s dest=%s", reason, destination)
        print(f"\n--- 🛟 FALLBACK: {reason} ---")

        if "budget" in reason.lower() or "exceeded" in reason.lower():
            return self._handle_budget(results, budget, trip_days, destination, origin, plan_state)

        return self._suggest_alternatives(destination, origin, budget, trip_days)

    # ── Budget handler ─────────────────────────────────────────────────────


   # ── Budget handler ─────────────────────────────────────────────────────

    def _handle_budget(self, results, budget, trip_days, destination, origin, plan_state):
        min_flight = _cheapest_price(results, "fetch_flights")
        min_ret = _cheapest_price(results, "fetch_return_flights")
        
        # שימוש בפונקציה הריאליסטית שתיקנו מקודם
        realistic_hotel = _realistic_hotel_night(results)

        base_cost = min_flight + min_ret  # flights are fixed

        # Try ±1, ±2, ±3 days
        for delta in [-1, -2, -3, 1, 2, 3]:
            new_days = trip_days + delta
            
            if new_days < 1 or new_days == trip_days:
                continue
                
            est_cost = (base_cost + realistic_hotel * new_days) * 1.15
            
            if budget and est_cost <= budget * 1.05:
                print(f"✅ Fallback: adjusting trip from {trip_days} → {new_days} days")
                
                # 🔥 תיקון באג 2 (Caching): מוחקים מהזיכרון את טיסת החזור והמלון 
                # כדי להכריח את ה-Executor להביא אותם מחדש לפי הימים החדשים!
                cleared_results = {
                    k: v for k, v in results.items() 
                    if not k.startswith("fetch_return_flights") and not k.startswith("fetch_hotels")
                }

                return {
                    "trip_days": new_days,
                    "itinerary_fallback_action": f"days_adjusted_to_{new_days}",
                    "itinerary_feasible": True,
                    "itinerary_fallback_alternatives": [],
                    # 🔥 תיקון באג 1 (Retry Loop): מאפסים את הניסיונות ודוחפים את הקאש הנקי
                    "itinerary_plan": {
                        **plan_state,
                        "retry_count": 0,
                        "observer_reason": "",
                        "step_results": cleared_results
                    }
                }

        # Try +$500 budget bump
        bumped = budget + 500
        est_cost = (base_cost + realistic_hotel * trip_days) * 1.15
        
        if est_cost <= bumped * 1.05:
            print(f"✅ Fallback: bumping budget ${budget} → ${bumped}")
            return {
                "total_budget": bumped,
                "itinerary_fallback_action": "budget_bumped_500",
                "itinerary_feasible": True,
                "itinerary_fallback_alternatives": [],
                # 🔥 תיקון באג 1: מאפסים ניסיונות גם כאן
                "itinerary_plan": {
                    **plan_state,
                    "retry_count": 0,
                    "observer_reason": ""
                }
            }

        # Nothing worked → alternatives
        print("⚠️ Budget adjustment impossible even with bumps. Suggesting alternatives.")
        return self._suggest_alternatives(destination, origin, bumped, trip_days)
    # ── Alternative city suggester ─────────────────────────────────────────

    def _suggest_alternatives(self, destination, origin, budget, trip_days):
        prompt = (
            f"Destination: {destination}\nOrigin: {origin}\n"
            f"Trip days: {trip_days}\nBudget: ${budget or 'flexible'}"
        )
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

        # Validate: check flights available for each
        validated: list[str] = []
        for city in alternatives[:3]:
            try:
                flights = search_outbound_flights.invoke({"origin": origin, "destination": city})
                if flights:
                    validated.append(city)
            except Exception:
                pass  # skip unavailable

        final = validated or alternatives[:3]
        md = _render_alternatives(destination, final, origin, budget, trip_days)

        print(f"💡 Suggesting alternatives: {final}")
        return {
            "itinerary_fallback_action": "suggested_alternatives",
            "itinerary_fallback_alternatives": final,
            "itinerary_feasible": False,
            "alternative_destinations": final,
            "messages": [AIMessage(content=md)],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cheapest_price(results: dict, prefix: str) -> float:
    for k, v in results.items():
        if k.startswith(prefix):
            items = v if isinstance(v, list) else [v]
            prices = [f.get("price", 9999) for f in items if isinstance(f, dict)]
            if prices:
                return min(prices)
    return 0.0


def _cheapest_hotel_night(results: dict) -> float:
    for k, v in results.items():
        if k.startswith("fetch_hotels"):
            # v may be {"activities": ..., "hotels": ...} or a list
            hotels = v if isinstance(v, list) else []
            if isinstance(v, dict):
                hotels = v.get("hotels", [])
            prices = [h.get("price_per_night", 9999) for h in hotels if isinstance(h, dict)]
            if prices:
                return min(prices)
    return 0.0

def _realistic_hotel_night(results: dict) -> float:
    """
    במקום לקחת את המלון הכי זול (מה שגורם לאופטימיות יתר), 
    ניקח ממוצע של 3 המלונות הזולים ביותר, כדי לאפשר ל-Planner גמישות בחירה.
    """
    for k, v in results.items():
        if k.startswith("fetch_hotels"):
            hotels = v if isinstance(v, list) else []
            if isinstance(v, dict):
                hotels = v.get("hotels", [])
            
            prices = [float(h.get("price_per_night", 9999)) for h in hotels if isinstance(h, dict)]
            
            if prices:
                prices.sort()
                top_3 = prices[:3] # ניקח את 3 הזולים ביותר
                return sum(top_3) / len(top_3)
    return 0.0
def _render_alternatives(
    original: str, alternatives: list[str], origin: str,
    budget: float, trip_days: int,
) -> str:
    lines = [
        "# 🗺️ Alternative Destinations\n",
        f"Unfortunately we couldn't build a complete itinerary for **{original}** "
        f"within your constraints (${budget} · {trip_days} days).\n",
        "Here are culturally similar alternatives with available flights:\n",
    ]
    emojis = ["1️⃣", "2️⃣", "3️⃣"]
    for i, city in enumerate(alternatives):
        lines.append(f"## {emojis[i]} {city}")
        lines.append(f"Direct or connecting flights available from **{origin}**. "
                     f"Similar culture and travel vibe to {original}.\n")
    lines += [
        "---",
        "💬 Reply with a city name and I'll plan your full itinerary!",
    ]
    return "\n".join(lines)
