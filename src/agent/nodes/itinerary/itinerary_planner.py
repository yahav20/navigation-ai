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


def _filter_flight(flight: dict, prefs: dict) -> bool:
    """מסנן טיסות לפי מחיר מקסימלי וחברת תעופה מועדפת"""
    if prefs.get("max_flight_price"):
        if flight.get("price", 9999) > prefs["max_flight_price"]:
            return False
            
    if prefs.get("preferred_airline"):
        pref_airline = str(prefs["preferred_airline"]).lower()
        flight_airline = str(flight.get("airline", "")).lower()
        if pref_airline not in flight_airline:
            return False
            
    return True

def _filter_hotel(hotel: dict, prefs: dict) -> bool:
    """מסנן מלונות לפי כוכבים, מחיר ללילה, כשרות ואבזור"""
    if prefs.get("min_hotel_stars"):
        if hotel.get("stars", 0) < prefs["min_hotel_stars"]:
            return False
            
    if prefs.get("max_hotel_price_per_night"):
        if hotel.get("price_per_night", 9999) > prefs["max_hotel_price_per_night"]:
            return False
            
    dietary = str(prefs.get("dietary_restrictions", "")).lower()
    if "kosher" in dietary:
        if not hotel.get("is_kosher"):
            return False
            
    if prefs.get("hotel_amenities"):
        req_amenities = str(prefs["hotel_amenities"]).lower()
        hotel_amenities = str(hotel.get("amenities", "")).lower()
        # בדיקה פשוטה אם האבזור המבוקש קיים בטקסט האבזור של המלון
        if req_amenities not in hotel_amenities:
            return False
            
    return True

def _filter_activity(activity: dict, prefs: dict) -> bool:
    """מסנן אטרקציות ומסעדות לפי תזונה (טבעוני/צמחוני/כשר)"""
    dietary = str(prefs.get("dietary_restrictions", "")).lower()
    
    if dietary and activity.get("food_available"):
        cats = str(activity.get("categories", "")).lower()
        features = str(activity.get("features", "")).lower()
        
        if "vegan" in dietary and "vegan" not in cats + features:
            return False
        if "vegetarian" in dietary and "vegetarian" not in cats + features:
            return False
        if "kosher" in dietary and "kosher" not in cats + features:
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
5. BUDGET ENFORCEMENT: 
   - Your total_days is {trip_days}. 
   - Your budget is {budget}. 
   - Calculation: (Flight Price + (Hotel Price * {trip_days}) + (Activities Budget)) MUST be <= {budget}.
   - IF your plan exceeds this, YOU MUST REDUCE days or switch to a cheaper hotel/activity IMMEDIATELY.
   - If you cannot meet the budget, RETURN ONLY: {"error": "budget_exceeded"}
6.Each activity must be selected from the provided activities list ONLY.
7.IF the total cost of your proposed itinerary exceeds the budget provided, 
DO NOT RETURN THE ITINERARY. Return a JSON object with: 
{ "error": "budget_exceeded", "reason": "Your itinerary is too expensive for the budget." }
Use exact name from dataset. Do NOT create new activities.
8. Return ONLY valid JSON — no markdown, no explanation.

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
        """Store the response model used to format the fallback text."""
        self.response_model = response_model

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def __call__(self, state: AgentState) -> dict:
        destination = state.get("destination_city", "")
        origin = state.get("current_city", "")
        budget = state.get("total_budget", 0)
        trip_days = state.get("trip_days", 3)
        prefs = state.get("user_preferences", {})
        flight_options = state.get("flight_options", [])

        raw_bundle = state.get("itinerary_data_bundle", {})
        print("\n================ ITINERARY PLANNER START ================")
        print(f"📍 Route: {origin} → {destination}")
        print(f"💰 Budget: {budget}")
        print(f"📆 Days: {trip_days}")
        print(f"🎯 Preferences: {prefs}")
        print(f"✈️ Flights in state: {len(flight_options)}")

        raw_bundle = state.get("itinerary_data_bundle", {})
        print(f"📦 Raw bundle exists: {bool(raw_bundle)}")

            # 1. Collect & Filter data deterministically from the raw bundle
        data_bundle = self._collect_data(
            raw_bundle=raw_bundle,
            prefs=prefs,
            budget=budget,
            trip_days=trip_days,
            flight_options=flight_options,
        )

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

        if "selected_flight" in plan and isinstance(plan["selected_flight"], dict):
            chosen_flight_num = plan["selected_flight"].get("flight_number")
            
            real_flight = next(
                (f for f in data_bundle["flights"] if f.get("flight_number") == chosen_flight_num), 
                None
            )
            
            if real_flight:
                plan["selected_flight"]["departure_time"] = real_flight.get("departure_time", "Not available")
                plan["selected_flight"]["arrival_time"] = real_flight.get("arrival_time", "Not available")
                plan["selected_flight"]["airline"] = real_flight.get("airline", "Unknown")
                plan["selected_flight"]["price"] = real_flight.get("price", 0)

        return {
            "itinerary_plan": plan,
            "itinerary_feasible": True,
            "itinerary_fallback_reason": None,
            "itinerary_data_bundle": data_bundle,
        }

    # ------------------------------------------------------------------
    # Data collection (deterministic — no LLM)
    # ------------------------------------------------------------------

    def _collect_data(
        self,
        raw_bundle: dict,
        prefs: dict,
        budget: float,
        trip_days: int,
        flight_options: list[dict],
    ) -> dict:
        """Filter raw data using explicit user preferences."""

        available_flights = [f for f in flight_options if str(f.get("availability", "")).lower() == "available"]
        filtered_flights = [f for f in available_flights if _filter_flight(f, prefs)]
        flights_sorted = sorted(filtered_flights, key=lambda f: f.get("price", 9999))
        return_flights = raw_bundle.get("return_flights", [])
        available_return_flights = [
            f for f in return_flights
            if str(f.get("availability", "")).lower() == "available"
        ]
        filtered_return_flights = [
            f for f in available_return_flights
            if _filter_flight(f, prefs)
        ]
        return_flights_sorted = sorted(filtered_return_flights, key=lambda f: f.get("price", 9999))
        raw_hotels = raw_bundle.get("hotels", [])
        filtered_hotels = [h for h in raw_hotels if _filter_hotel(h, prefs)]
        hotels_sorted = sorted(filtered_hotels, key=lambda h: (-h.get("stars", 0), h.get("price_per_night", 9999)))

        raw_activities = raw_bundle.get("activities", [])
        filtered_activities = [a for a in raw_activities if _filter_activity(a, prefs)]
        activities_sorted = sorted(filtered_activities, key=lambda a: -a.get("rating", 0))

        return {
            "flights": flights_sorted[:3],
            "return_flights": return_flights_sorted[:3],
            "hotels": hotels_sorted[:5],
            "activities": activities_sorted[:20],
            "weather": raw_bundle.get("weather", []),
            "best_time": raw_bundle.get("best_time", {}),
            "budget": budget,
            "trip_days": trip_days,
            "preferences": prefs,
        }
    # ------------------------------------------------------------------
    # Feasibility check
    # ------------------------------------------------------------------

    def _check_feasibility(self, bundle: dict, budget: float, trip_days: int) -> dict:
        if not bundle["flights"]:
            return {"feasible": False, "reason": "no_flights"}

        if not bundle["hotels"]:
            return {"feasible": False, "reason": "no_hotels"}
        print(f"DEBUG: Min cost calculated: {min_cost}, Budget: {budget}")
        # מציאת המחיר הזול ביותר האמיתי מתוך כל האופציות בחבילה
        min_flight = min((f.get("price", 9999) for f in bundle["flights"]), default=0)
        min_hotel_night = min((h.get("price_per_night", 9999) for h in bundle["hotels"]), default=0)

        min_cost = min_flight + (min_hotel_night * trip_days)

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
        return_flights = data_bundle.get("return_flights") or []

        return_flight = return_flights[0] if return_flights else {}
        return_departure = return_flight.get("departure_time", "Unknown")
        data_bundle["return_departure"] = return_departure
            
        user_msg = f"""
Plan a {trip_days}-day trip from {origin} to {destination}.

USER PREFERENCES: {json.dumps(prefs, ensure_ascii=False)}

AVAILABLE OUTBOUND FLIGHTS (pick one):
{json.dumps(data_bundle['flights'], ensure_ascii=False, indent=2)}

RETURN FLIGHT DEPARTURE TIME: {data_bundle['return_departure']}

AVAILABLE HOTELS (pick one, must match preferences):
{json.dumps(data_bundle['hotels'], ensure_ascii=False, indent=2)}

AVAILABLE ACTIVITIES (choose the best mix per day — variety, proximity, preferences):
{json.dumps(data_bundle['activities'], ensure_ascii=False, indent=2)}

BUDGET: ${data_bundle['budget'] or 'flexible'}

Remember:
- Day 1 schedule depends on the outbound flight arrival time.
- Account for hotel check-in (usually 15:00) and check-out (usually 10:00–11:00).
- Estimate walk time between hotel and each activity using coordinates.
- 🔴 CRITICAL: - Day {trip_days} MUST end with:
  "Transfer to airport"
- Use RETURN FLIGHT DEPARTURE TIME: {return_departure}
- Ensure this is the LAST activity of the day

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
