# src/agent/nodes/itinerary/executor.py
from __future__ import annotations
import json
import logging
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agent.nodes.itinerary.schemas import ExecutionPlan

# ייבוא הכלים של ה-Itinerary
from agent.nodes.itinerary.itinerary_tools import (
    search_outbound_flights, search_return_flights,
    search_hotels, search_activities,
    get_weather, calculate_trip_cost,
)
from agent.state import AgentState

logger = logging.getLogger(__name__)

DEFAULT_MEALS_PER_DAY = 60.0

DAY_SCHEDULE_SYSTEM = """
You are a highly precise Travel Schedule Engineer. Your goal is to build a 100% logical, non-overlapping day-by-day itinerary.

### ALGORITHMIC RULES:
1. **TRANSIT CALCULATION:** - You are provided with coordinates (lat/lng) for all locations. 
   - If distance is < 1.5km, set slot_type to 'transport', mode to 'walk', duration to 15 mins/km.
   - If distance >= 1.5km, set slot_type to 'transport', mode to 'car', duration to 30 mins, add $20 cost.
   - You MUST include a 'transport' slot between ANY two locations that are >0.5km apart.

2. **FOOD & DINING:**
   - DO NOT suggest standalone 'Lunch' or 'Dinner' slots if the activity you are visiting has `food_available: true`. 
   - Integrate meals into the activity experience whenever possible.
   - Only create a 'meal' slot if the user has been active for >4 hours without a food-enabled activity.

3. **TIMING & OVERLAP:**
   - Total Day Duration: Start from 08:00 to 22:00.
   - NO OVERLAP: Slot N+1 start_time MUST be >= Slot N end_time + Transit_Time.
   - If a slot ends at 10:00 and transit is 30 mins, the next activity can only start at 10:30.

4. **ANCHORS (IMMUTABLE):**
   - Day 1: First activity cannot start before (Arrival_Time + 90 mins).
   - Last Day: Last activity must end (Departure_Time - 150 mins). 
   - Everything between these anchors must be perfectly sequential.

5. **ALLOWED SLOTS:**
   - Only these types are allowed: 'activity', 'meal', 'rest', 'transport'.
   - 'rest' is only allowed once per day (30 mins after lunch).

OUTPUT FORMAT:
Return ONLY a valid JSON array of slot objects:
[{"time":"HH:MM","duration_minutes":int,"slot_type":"activity|meal|rest|transport", "name":"string", "estimated_cost":float}]
"""

class ItineraryExecutorNode:
    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm

    def __call__(self, state: AgentState) -> dict:
        plan_state  = state.get("itinerary_plan", {})
        plan        = ExecutionPlan(**plan_state["execution_plan"])
        results     = dict(plan_state.get("step_results", {}))
        
        # --- שליפת המשתנים מה-State ---
        destination = state.get("destination_city", "")
        origin      = state.get("current_city", "")
        trip_days   = state.get("trip_days", plan.total_days)
        budget      = state.get("total_budget", 0)
        prefs       = state.get("user_preferences", {})

        current_index = state.get("current_step_index", 0)
        
        # הגנה מפני חריגת אינדקס
        if current_index >= len(plan.steps):
            return {"skipped": True}

        step = plan.steps[current_index]
        key = f"{step.step_type}_{step.step_id}"
        cache_key = step.step_type 

        print(f"\n--- ⚙️ EXECUTING STEP {current_index + 1}/{len(plan.steps)}: {step.step_type} ---")
        
        # מנגנון Caching (למניעת קריאות כפולות לאותו כלי אם ה-Replanner הריץ שוב)
        cached_key = next((k for k in results if k.startswith(cache_key) and "error" not in results[k] and "skipped" not in results[k]), None)
        
        if cached_key and step.step_type not in ["build_day_schedule", "verify_budget"]:
             print(f"⏩ Skipping Tool Execution - Using cached actual data for '{step.step_type}'.")
             results[key] = results[cached_key] 
        else:
             try:
                 # 🔴 כאן אנחנו מעבירים את כל המשתנים בבטחה לפונקציית העזר _run
                 results[key] = self._run(step, results, destination, origin, trip_days, budget, prefs, state)
                 print(f"✅ Step {step.step_type} completed successfully.")
             except Exception as e:
                 print(f"❌ Step {step.step_type} failed: {e}")
                 results[key] = {"error": str(e), "step_type": step.step_type}

        return {
            "current_step_index": current_index + 1,
            "itinerary_plan": {**plan_state, "step_results": results}
        }

    def _run(self, step, results, destination, origin, trip_days, budget, prefs, state):
        dietary = str(prefs.get("dietary_restrictions", "")).lower()
        kosher  = "kosher"     in dietary
        veg     = "vegetarian" in dietary
        vegan   = "vegan"      in dietary

        if step.step_type == "fetch_flights":
            return search_outbound_flights.invoke({"origin": origin, "destination": destination})

        if step.step_type == "fetch_return_flights":
            return search_return_flights.invoke({"origin": destination, "destination": origin})

        if step.step_type == "fetch_hotels":
            return search_hotels.invoke({"city": destination, "kosher_only": kosher})

        if step.step_type == "fetch_activities":
            return search_activities.invoke({
                "city": destination, "kosher_only": kosher,
                "vegetarian_friendly": veg, "vegan_friendly": vegan,
            })

        if step.step_type == "fetch_weather":
            return get_weather.invoke({"city": destination})

        if step.step_type == "build_day_schedule":
            return self._build_day(step, results, destination, trip_days, prefs)

        if step.step_type == "verify_budget":
            return self._verify_budget(results, budget, trip_days)

        return {"skipped": True}

    # ------------------------------------------------------------------
    def _build_day(self, step, results, destination, trip_days, prefs):
        day_num = step.day or 1

        outbound = _first(results, "fetch_flights")
        ret      = _first(results, "fetch_return_flights")
        hotel    = _first(results, "fetch_hotels")
        acts     = _list(results, "fetch_activities")
        weather  = _list(results, "fetch_weather")

        used = _used_activities(results)
        available = [a for a in acts if isinstance(a, dict) and a.get("name") not in used]

        # שליפה בטוחה למקרה שהכלים חזרו ריקים
        arrival_time   = (outbound or {}).get("arrival_time", "12:00") if isinstance(outbound, dict) else "12:00"
        departure_time = (ret or {}).get("departure_time", "20:00") if isinstance(ret, dict) else "20:00"
        has_breakfast  = (hotel or {}).get("breakfast_available", False) if isinstance(hotel, dict) else False
        hotel_name     = (hotel or {}).get("name", "N/A") if isinstance(hotel, dict) else "N/A"
        hotel_lat      = (hotel or {}).get("latitude") or (hotel or {}).get("lat") if isinstance(hotel, dict) else None
        hotel_lng      = (hotel or {}).get("longitude") or (hotel or {}).get("lng") if isinstance(hotel, dict) else None

        context = (
            f"Day {day_num} of {trip_days} in {destination}.\n"
            f"Hotel: {hotel_name} (breakfast_available={has_breakfast}, "
            f"lat={hotel_lat}, lng={hotel_lng})\n"
            f"User preferences: {prefs}\n"
            f"Weather: {json.dumps(weather, ensure_ascii=False)}\n"
            + (f"ARRIVAL DAY: flight arrives {arrival_time}\n" if day_num == 1 else "")
            + (f"DEPARTURE DAY: return flight departs {departure_time}\n" if day_num == trip_days else "")
            + f"\nAvailable activities (not yet used, sorted by rating):\n"
            + json.dumps(available[:10], ensure_ascii=False, indent=2)
        )

        raw = self.llm.invoke([
            SystemMessage(content=DAY_SCHEDULE_SYSTEM),
            HumanMessage(content=context),
        ]).content.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip().rstrip("```").strip()

        try:
            slots = json.loads(raw)
            if not isinstance(slots, list):
                raise ValueError("Expected list")
        except Exception as e:
            logger.error("Day %d parse error: %s — using fallback", day_num, e)
            slots = _fallback_slots(day_num, arrival_time, departure_time,
                                     has_breakfast, available[:4])

        day_cost = round(sum(float(s.get("estimated_cost", 0)) for s in slots), 2)
        themes = {1: "Arrival & first impressions", trip_days: "Final day & departure"}
        return {
            "day": day_num,
            "theme": themes.get(day_num, f"Day {day_num} — Explore {destination}"),
            "slots": slots,
            "day_cost": day_cost,
        }

    def _verify_budget(self, results, budget, trip_days):
        outbound = _first(results, "fetch_flights")
        ret      = _first(results, "fetch_return_flights")
        hotel    = _first(results, "fetch_hotels")

        activity_cost = sum(
            v.get("day_cost", 0) for k, v in results.items()
            if k.startswith("build_day_schedule") and isinstance(v, dict)
        )
        return calculate_trip_cost.invoke({
            "flight_price":                  (outbound or {}).get("price", 0) if isinstance(outbound, dict) else 0,
            "return_flight_price":           (ret or {}).get("price", 0) if isinstance(ret, dict) else 0,
            "hotel_price_per_night":         (hotel or {}).get("price_per_night", 0) if isinstance(hotel, dict) else 0,
            "trip_days":                     trip_days,
            "estimated_activities_budget":   activity_cost,
            "estimated_meals_budget_per_day": DEFAULT_MEALS_PER_DAY,
        })


# ---------------------------------------------------------------------------
def _find_key(results: dict, prefix: str) -> Optional[str]:
    for k in results:
        if k.startswith(prefix):
            return k
    return None

def _first(results: dict, prefix: str) -> Optional[dict]:
    k = _find_key(results, prefix)
    if not k:
        return None
    v = results[k]
    if isinstance(v, list):
        return v[0] if v else None
    return v if isinstance(v, dict) and not v.get("error") else None

def _list(results: dict, prefix: str) -> list:
    k = _find_key(results, prefix)
    if not k:
        return []
    v = results[k]
    return v if isinstance(v, list) else []

def _used_activities(results: dict) -> set:
    used = set()
    for v in results.values():
        if isinstance(v, dict) and "slots" in v:
            for s in v["slots"]:
                if s.get("slot_type") == "activity":
                    used.add(s.get("name", ""))
    return used

def _fallback_slots(day_num, arrival, departure, has_breakfast, acts):
    slots = []
    if day_num == 1:
        slots += [
            {"time": arrival, "duration_minutes": 90, "slot_type": "rest",
             "name": "Arrival & check-in", "description": "Settle in.", "estimated_cost": 0},
            {"time": "13:00", "duration_minutes": 60, "slot_type": "meal",
             "name": "Lunch", "description": "Local restaurant.", "estimated_cost": 18},
            {"time": "19:30", "duration_minutes": 60, "slot_type": "meal",
             "name": "Welcome dinner", "description": "", "estimated_cost": 25},
        ]
    else:
        bk_cost = 0 if has_breakfast else 12
        bk_name = "Hotel breakfast" if has_breakfast else "Breakfast at café"
        slots.append({"time": "08:00", "duration_minutes": 60, "slot_type": "meal",
                       "name": bk_name, "description": "", "estimated_cost": bk_cost})
        for i, a in enumerate(acts[:2]):
            slots.append({"time": f"{10+i*2}:00", "duration_minutes": a.get("avg_duration_minutes",90) if isinstance(a, dict) else 90,
                           "slot_type": "activity", "name": a.get("name","Activity") if isinstance(a, dict) else "Activity",
                           "description": a.get("categories","") if isinstance(a, dict) else "", "estimated_cost": a.get("price",0) if isinstance(a, dict) else 0})
        slots += [
            {"time": "13:00", "duration_minutes": 60, "slot_type": "meal",
             "name": "Lunch", "description": "", "estimated_cost": 18},
            {"time": "14:00", "duration_minutes": 30, "slot_type": "rest",
             "name": "Afternoon rest", "description": "", "estimated_cost": 0},
        ]
        for i, a in enumerate(acts[2:4]):
            slots.append({"time": f"{15+i*2}:00", "duration_minutes": a.get("avg_duration_minutes",90) if isinstance(a, dict) else 90,
                           "slot_type": "activity", "name": a.get("name","Activity") if isinstance(a, dict) else "Activity",
                           "description": a.get("categories","") if isinstance(a, dict) else "", "estimated_cost": a.get("price",0) if isinstance(a, dict) else 0})
        slots.append({"time": "19:30", "duration_minutes": 60, "slot_type": "meal",
                       "name": "Dinner", "description": "", "estimated_cost": 25})
    return slots