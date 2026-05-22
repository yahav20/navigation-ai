"""
ItineraryPlannerNode — Plan & Execute, Step 1: PLAN

Pulls ALL required data from the DB in one shot (deterministic, no LLM tool calls),
then uses the LLM purely as a *reasoner* to produce a structured day-by-day plan.

Data collected:
  - Flight options (already in state from enrichment/flight_search)
  - Hotels filtered by user preferences (kosher, accessibility, budget)
  - Activities filtered by user preferences, city, and day-schedule logic
  - Weather + best-time-to-visit for contextual planning
  - Hotel check-in time → determines Day-1 morning availability
  - Flight arrival time → determines Day-1 afternoon/evening schedule
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from agent.state import AgentState


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_amenities(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return [a.strip() for a in raw.split(",") if a.strip()]
    return []


def _matches_preferences(item: dict, prefs: dict) -> bool:
    """Return True when *item* satisfies every hard constraint in *prefs*."""
    # Kosher
    if prefs.get("kosher"):
        if not item.get("is_kosher") and not item.get("kosher"):
            return False
    # Wheelchair / accessibility
    if prefs.get("wheelchair") or prefs.get("accessibility"):
        amenities = _parse_amenities(item.get("amenities", []))
        features = item.get("features", [])
        accessible = any(
            "wheelchair" in str(a).lower() or "accessible" in str(a).lower()
            for a in amenities + features
        )
        if not accessible:
            return False
    # Vegan / vegetarian
    if prefs.get("vegan") or prefs.get("vegetarian"):
        cats = str(item.get("categories", "")).lower()
        food = str(item.get("food_available", "")).lower()
        if not ("vegan" in cats or "vegetarian" in cats or "vegan" in food):
            return False
    # Min-age guard (e.g. families with toddlers)
    if "min_age" in prefs:
        if item.get("min_age", 0) > prefs["min_age"]:
            return False
    return True


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class ItineraryPlannerNode:
    """
    Deterministic data-collection + LLM-based planning.

    Outputs written to state:
        itinerary_plan: dict  — structured day-by-day plan
        itinerary_feasible: bool
        itinerary_fallback_reason: str | None
    """

    SYSTEM_PROMPT = """You are an expert travel planner.
You receive raw travel data (flights, hotels, activities, weather) and produce
a realistic, hour-by-hour day-by-day itinerary in JSON.

RULES:
1. Be deterministic — choose specific items from the data provided, do NOT invent.
2. Respect user preferences strictly (kosher, accessibility, vegan, min_age, etc.).
3. Think logically about timing:
   - If the flight lands in the morning  → plan rest/lunch first, light afternoon activity, evening stroll.
   - If the flight lands in the afternoon → check-in, dinner, evening activity only.
   - If the flight lands at night         → check-in and sleep only.
   - Hotel breakfast available + check-out at 10:00 → first activity no earlier than 10:30 + walk time.
   - Last day: pick only activities close to the airport / that end before flight departure.
4. Estimate walking distances using hotel lat/lng vs activity lat/lng.
   Prefer nearby activities for morning slots, allow farther ones for afternoon.
5. Budget awareness: track cumulative cost (flight + hotel_total + activities).
6. Return ONLY valid JSON — no markdown, no explanation.

OUTPUT SCHEMA:
{
  "destination": "string",
  "origin": "string",
  "total_days": int,
  "estimated_total_cost": float,
  "selected_hotel": { "name": str, "stars": int, "price_per_night": float,
                       "breakfast_available": bool, "lat": float, "lng": float },
  "selected_flight": { "flight_number": str, "airline": str, "price": float,
                        "departure_time": str, "arrival_time": str },
  "days": [
    {
      "day": 1,
      "theme": "Arrival & first impressions",
      "slots": [
        {
          "time": "14:00",
          "duration_minutes": 60,
          "type": "activity|meal|rest|transport",
          "name": "string",
          "description": "string",
          "estimated_cost": float,
          "notes": "string (optional)"
        }
      ]
    }
  ],
  "user_preferences_applied": ["kosher", "wheelchair", ...]
}
"""

    def __init__(self, response_model: BaseChatModel) -> None:
        """Store the model used to reason and generate the structured itinerary."""
        self.response_model = response_model
    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def __call__(self, state: AgentState) -> dict:
        print("=== ITINERARY PLANNER START ===")
        print(f"bundle exists: {bool(state.get('itinerary_data_bundle'))}")
        print(f"bundle keys: {list(state.get('itinerary_data_bundle', {}).keys())}")
        destination = state.get("destination_city", "")
        origin = state.get("current_city", "")
        budget = state.get("total_budget", 0)
        trip_days = state.get("trip_days", 3)
        prefs = state.get("user_preferences", {})
        
        print(
            "ItineraryPlannerNode: destination=%s origin=%s days=%d budget=%.0f",
            destination, origin, trip_days, budget,
        )

        # 1. משיכת הנתונים מוכנים מתוך ה-State (במקום לשלוף מה-DB)
        # ה-DataPrepNode שעשינו קודם כבר ארז את זה בשבילנו!
        data_bundle = state.get("itinerary_data_bundle", {})
        
        # נוודא שהחבילה לא ריקה לחלוטין (למקרה של תקלה קודמת)
        if not data_bundle:
             print("❌ EMPTY ITINERARY DATA BUNDLE - PLANNER EXIT")
             return {
                 "itinerary_plan": {},
                 "itinerary_feasible": False,
                 "itinerary_fallback_reason": "missing_data",
             }

        # 2. Feasibility check BEFORE calling LLM
        feasibility = self._check_feasibility(data_bundle, budget, trip_days)
        if not feasibility["feasible"]:
            return {
                "itinerary_plan": {},
                "itinerary_feasible": False,
                "itinerary_fallback_reason": feasibility["reason"],
                "itinerary_data_bundle": data_bundle,  # keep for fallback node
            }

        # 3. LLM reasons over the data → structured itinerary
        plan = self._plan_with_llm(
            data_bundle=data_bundle,
            destination=destination,
            origin=origin,
            trip_days=trip_days,
            prefs=prefs,
        )

        return {
            "itinerary_plan": plan,
            "itinerary_feasible": True,
            "itinerary_fallback_reason": None,
            "itinerary_data_bundle": data_bundle,
        }
   
    # ------------------------------------------------------------------
    # Feasibility check
    # ------------------------------------------------------------------

    def _check_feasibility(self, bundle: dict, budget: float, trip_days: int) -> dict:
        if not bundle["flights"]:
            return {"feasible": False, "reason": "no_flights"}

        if not bundle["hotels"]:
            return {"feasible": False, "reason": "no_hotels"}

        # Rough cost estimate: cheapest flight + cheapest hotel × days
        min_flight = bundle["flights"][0].get("price", 0)
        min_hotel_night = bundle["hotels"][0].get("price_per_night", 0)
        min_cost = min_flight + min_hotel_night * trip_days

        if budget and min_cost > budget * 1.05:  # 5% tolerance
            return {
                "feasible": False,
                "reason": "budget_exceeded",
                "min_cost": min_cost,
            }

        return {"feasible": True}

    # ------------------------------------------------------------------
    # LLM planning
    # ------------------------------------------------------------------

    def _plan_with_llm(
        self,
        data_bundle: dict,
        destination: str,
        origin: str,
        trip_days: int,
        prefs: dict,
    ) -> dict:
        user_msg = f"""
Plan a {trip_days}-day trip from {origin} to {destination}.

USER PREFERENCES: {json.dumps(prefs, ensure_ascii=False)}

AVAILABLE FLIGHTS (pick one):
{json.dumps(data_bundle['flights'], ensure_ascii=False, indent=2)}

AVAILABLE HOTELS (pick one, must match preferences):
{json.dumps(data_bundle['hotels'], ensure_ascii=False, indent=2)}

AVAILABLE ACTIVITIES (choose the best mix per day — variety, proximity, preferences):
{json.dumps(data_bundle['activities'], ensure_ascii=False, indent=2)}

WEATHER INFO:
{json.dumps(data_bundle['weather'], ensure_ascii=False, indent=2)}

BEST TIME TO VISIT:
{json.dumps(data_bundle['best_time'], ensure_ascii=False, indent=2)}

BUDGET: ${data_bundle['budget'] or 'flexible'}

Remember:
- Day 1 schedule depends on flight arrival time.
- Account for hotel check-in (usually 15:00) and check-out (usually 10:00–11:00).
- If hotel has breakfast, first activity starts after 10:00.
- Estimate walk time between hotel and each activity using coordinates.
- Last day: only activities that end ≥2 hours before flight departure.

Return ONLY the JSON object described in your instructions.
"""
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ]

        response = self.response_model.invoke(messages)
        raw = response.content.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error("ItineraryPlannerNode: JSON parse error: %s\nRaw: %s", e, raw[:500])
            return {"error": "parse_failed", "raw": raw}
