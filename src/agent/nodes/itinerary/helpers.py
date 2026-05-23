"""
Shared helper functions for the itinerary sub-graph.
Pure functions only — no state, no LLM, no DB.
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timedelta
from typing import Optional

from agent.nodes.itinerary.schemas import DaySlot, ExecutionPlan, PlanStep


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def travel_slot(to_name: str, km: float, depart_time: datetime) -> DaySlot:
    if km <= 2.0:
        mode, mins, cost = "walking", max(5, int((km / 4.5) * 60)), 0.0
    else:
        mode, mins, cost = "taxi/rideshare", max(5, int((km / 30.0) * 60)), 8.0
    return DaySlot(
        time=fmt_time(depart_time),
        duration_minutes=mins,
        slot_type="transport",
        name=f"Travel to {to_name}",
        description=f"{mode.capitalize()} — {km:.1f} km",
        estimated_cost=cost,
        notes=f"~{mins} min",
    )


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def parse_dt(s: str) -> datetime:
    base = datetime(2000, 1, 1)
    for fmt in ("%H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M"):
        try:
            d = datetime.strptime(s, fmt)
            return d.replace(year=2000) if d.year == 1900 else d
        except ValueError:
            continue
    return base


def fmt_time(dt: datetime) -> str:
    return dt.strftime("%H:%M")


# ---------------------------------------------------------------------------
# Data filtering (called by planner before sending to LLM)
# ---------------------------------------------------------------------------

def filter_bundle(
    raw: dict,
    flight_options: list,
    prefs: dict,
    budget: float,
    trip_days: int,
) -> dict:
    """
    Filter raw DB data by user preferences.
    Returns a clean bundle with only valid, sorted options.
    """
    dietary = str(prefs.get("dietary_restrictions", "")).lower()

    # Outbound flights — available only, sorted cheapest first
    available = [f for f in flight_options
                 if str(f.get("availability", "")).lower() == "available"]
    flights = sorted(available, key=lambda f: f.get("price", 9999))

    # Return flights
    ret_avail = [f for f in raw.get("return_flights", [])
                 if str(f.get("availability", "")).lower() == "available"]
    return_flights = sorted(ret_avail, key=lambda f: f.get("price", 9999))

    # Hotels
    hotels_raw = raw.get("hotels", [])
    if "kosher" in dietary:
       
        hotels_raw = [h for h in hotels_raw if h.get("is_kosher")]
        
    if budget > 0:
        min_outbound = flights[0].get("price", 0) if flights else 0
        min_return = return_flights[0].get("price", 0) if return_flights else 0
        total_flights_cost = min_outbound + min_return
        
        budget_left_for_hotel = (budget - total_flights_cost) * 0.80
        nights = max(1, trip_days - 1)
        
        hotels_raw = [
            h for h in hotels_raw 
            if (h.get("price_per_night", 0) * nights) <= budget_left_for_hotel
        ]

    hotels = sorted(hotels_raw, 
                key=lambda h: (-h.get("stars", 0), h.get("price_per_night", 0)))

    # Activities — filter food venues by dietary, sort by rating
    def _activity_ok(a: dict) -> bool:
        if not dietary or not a.get("food_available"):
            return True
        cats = str(a.get("categories", "")).lower()
        if "vegan" in dietary and "vegan" not in cats:
            return False
        if "vegetarian" in dietary and "vegetarian" not in cats:
            return False
        if "kosher" in dietary and "kosher" not in cats:
            return False
        return True

    activities = sorted(
        [a for a in raw.get("activities", []) if _activity_ok(a)],
        key=lambda a: -a.get("rating", 0),
    )

    return {
        "flights": flights[:2],           
        "return_flights": return_flights[:2],
        "hotels": hotels[:3],           
        "activities": activities[:12],   
        "weather": raw.get("weather", []),
        "best_time": raw.get("best_time", {}),
    }


def check_feasibility(bundle: dict, budget: float, trip_days: int) -> dict:
    """Return {feasible: bool, reason: str|None}."""
    if not bundle["flights"]:
        return {"feasible": False, "reason": "no_flights"}
    if not bundle["hotels"]:
        return {"feasible": False, "reason": "no_hotels"}

    min_flight = min((f.get("price", 0) for f in bundle["flights"]), default=0)
    min_hotel  = min((h.get("price_per_night", 0) for h in bundle["hotels"]), default=0)
    min_cost   = min_flight + min_hotel * trip_days

    if budget and min_cost > budget * 1.05:
        return {"feasible": False, "reason": "budget_exceeded"}

    return {"feasible": True, "reason": None}


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def strip_fences(raw: str) -> str:
    """Remove markdown code fences from LLM output."""
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


def extract_day_number(description: str, fallback: int) -> int:
    m = re.search(r"[Dd]ay\s*(\d+)", description)
    return int(m.group(1)) if m else fallback


def default_plan(destination: str, origin: str, trip_days: int) -> ExecutionPlan:
    """Fallback plan when LLM output can't be parsed."""
    steps = [
        PlanStep(step_id=1, step_type="select_flight",
                 description="Select cheapest available outbound flight", depends_on=[]),
        PlanStep(step_id=2, step_type="select_hotel",
                 description="Select best hotel by stars then price", depends_on=[1]),
    ]
    for d in range(1, trip_days + 1):
        steps.append(PlanStep(
            step_id=2 + d,
            step_type="build_day",
            description=f"Day {d}: build schedule",
            depends_on=[1, 2],
        ))
    steps.append(PlanStep(
        step_id=3 + trip_days,
        step_type="verify_budget",
        description="Verify total cost against budget",
        depends_on=list(range(1, 3 + trip_days)),
    ))
    return ExecutionPlan(destination=destination, origin=origin,
                         total_days=trip_days, steps=steps)