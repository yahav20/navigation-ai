"""
ItineraryExecutorNode — executes one PlanStep at a time using tools.
See docstring in class for full design notes.
"""
from __future__ import annotations
import json
import logging
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agent.nodes.itinerary.schemas import ExecutionPlan, PlanStep
from agent.nodes.itinerary.itinerary_tools import (
    search_outbound_flights, search_return_flights,
    search_hotels, search_activities,
    get_weather, calculate_trip_cost,
)
from agent.state import AgentState

logger = logging.getLogger(__name__)

DEFAULT_MEALS_PER_DAY = 60.0

DAY_SCHEDULE_SYSTEM = """You are a single-day travel scheduler.
Build a realistic hour-by-hour schedule for ONE day of a trip.

MANDATORY:
- 3 meals every full day: breakfast (~08:00), lunch (~13:00), dinner (~19:30).
- Hotel breakfast is FREE if breakfast_available=true.
- Include a 30-min rest after lunch.
- Add transport slots between locations that are >0.5 km apart.
- Respect activity opening_time / closing_time.
- Day 1: first activity only AFTER flight arrival + 90 min check-in.
- Last day: no activity ending later than 2.5 h before return flight departure.

Return ONLY a valid JSON array of slot objects. No explanation.
Each slot: {"time":"HH:MM","duration_minutes":int,"slot_type":"activity|meal|rest|transport",
            "name":"string","description":"string","estimated_cost":float,"notes":"string or null"}
"""


class ItineraryExecutorNode:
    """
    Iterates through every PlanStep and executes it.
    - Data-fetch steps: call tools deterministically.
    - build_day_schedule: LLM builds the day using context from prior steps.
    - verify_budget: deterministic calculation.
    Tool errors are caught and stored as {"error": "..."} — never crash.
    """

    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm

    def __call__(self, state: AgentState) -> dict:
        plan_state  = state.get("itinerary_plan", {})
        plan        = ExecutionPlan(**plan_state["execution_plan"])
        results     = dict(plan_state.get("step_results", {}))
        destination = state.get("destination_city", "")
        origin      = state.get("current_city", "")
        trip_days   = state.get("trip_days", plan.total_days)
        budget      = state.get("total_budget", 0)
        prefs       = state.get("user_preferences", {})

        for step in plan.steps:
            key = f"{step.step_type}_{step.step_id}"
            logger.info("Executor: step %d [%s]", step.step_id, step.step_type)
            try:
                results[key] = self._run(step, results, destination, origin,
                                         trip_days, budget, prefs, state)
            except Exception as e:
                logger.warning("Step %d error: %s", step.step_id, e)
                results[key] = {"error": str(e), "step_type": step.step_type}
                
                
        

        return {"itinerary_plan": {**plan_state, "step_results": results}}

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
        available = [a for a in acts if a.get("name") not in used]

        arrival_time   = (outbound or {}).get("arrival_time", "12:00")
        departure_time = (ret or {}).get("departure_time", "20:00")
        has_breakfast  = (hotel or {}).get("breakfast_available", False)
        hotel_name     = (hotel or {}).get("name", "N/A")
        hotel_lat      = (hotel or {}).get("latitude") or (hotel or {}).get("lat")
        hotel_lng      = (hotel or {}).get("longitude") or (hotel or {}).get("lng")

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
            "flight_price":                  (outbound or {}).get("price", 0),
            "return_flight_price":           (ret or {}).get("price", 0),
            "hotel_price_per_night":         (hotel or {}).get("price_per_night", 0),
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
            slots.append({"time": f"{10+i*2}:00", "duration_minutes": a.get("avg_duration_minutes",90),
                           "slot_type": "activity", "name": a.get("name","Activity"),
                           "description": a.get("categories",""), "estimated_cost": a.get("price",0)})
        slots += [
            {"time": "13:00", "duration_minutes": 60, "slot_type": "meal",
             "name": "Lunch", "description": "", "estimated_cost": 18},
            {"time": "14:00", "duration_minutes": 30, "slot_type": "rest",
             "name": "Afternoon rest", "description": "", "estimated_cost": 0},
        ]
        for i, a in enumerate(acts[2:4]):
            slots.append({"time": f"{15+i*2}:00", "duration_minutes": a.get("avg_duration_minutes",90),
                           "slot_type": "activity", "name": a.get("name","Activity"),
                           "description": a.get("categories",""), "estimated_cost": a.get("price",0)})
        slots.append({"time": "19:30", "duration_minutes": 60, "slot_type": "meal",
                       "name": "Dinner", "description": "", "estimated_cost": 25})
    return slots
