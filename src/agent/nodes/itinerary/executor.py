"""
ItineraryExecutorNode — Plan & Execute, Step 2: EXECUTE

Runs each PlanStep from the ExecutionPlan deterministically.
NO LLM calls. Only real data from the filtered bundle.

Steps handled:
  select_flight  → cheapest available outbound flight (real departure/arrival times)
  select_hotel   → top hotel after filtering
  build_day      → hour-by-hour slots with Haversine travel times
  verify_budget  → total cost calculation
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

from agent.nodes.itinerary.helpers import (
    extract_day_number,
    fmt_time,
    haversine_km,
    parse_dt,
    travel_slot,
)
from agent.nodes.itinerary.schemas import DaySlot, ExecutionPlan
from agent.state import AgentState

logger = logging.getLogger(__name__)

DAY_THEMES = {1: "Arrival & first impressions"}


class ItineraryExecutorNode:
    """No LLM. Produces a fully assembled itinerary dict."""

    def __call__(self, state: AgentState) -> dict:
        plan_state = state.get("itinerary_plan", {})
        plan       = ExecutionPlan(**plan_state["execution_plan"])
        bundle     = plan_state["filtered_bundle"]
        trip_days  = state.get("trip_days", plan.total_days)
        budget     = state.get("total_budget", 0)

        execution_results: dict = {}

        for step in plan.steps:
            logger.info("Executing step %d [%s]: %s",
                        step.step_id, step.step_type, step.description[:60])

            if step.step_type == "select_flight":
                execution_results["select_flight"] = _exec_select_flight(bundle)

            elif step.step_type == "select_hotel":
                execution_results["select_hotel"] = _exec_select_hotel(bundle)

            elif step.step_type == "build_day":
                day_num = extract_day_number(step.description, len(execution_results) - 1)
                used    = _collect_used(execution_results)
                execution_results[f"build_day_{day_num}"] = _exec_build_day(
                    day_num=day_num,
                    total_days=trip_days,
                    activities=bundle.get("activities", []),
                    hotel=execution_results.get("select_hotel", {}),
                    outbound_flight=execution_results.get("select_flight", {}),
                    return_flights=bundle.get("return_flights", []),
                    already_used=used,
                )

            elif step.step_type == "verify_budget":
                execution_results["verify_budget"] = _exec_verify_budget(
                    execution_results, budget, trip_days, bundle
                )

        assembled = _assemble(execution_results, state, plan)
        return {
            "itinerary_plan": {
                **plan_state,
                "assembled": assembled,
            }
        }


# ---------------------------------------------------------------------------
# Step executors
# ---------------------------------------------------------------------------

def _exec_select_flight(bundle: dict) -> dict:
    flights = bundle.get("flights", [])
    if not flights:
        return {}
    chosen = flights[0]  # already sorted cheapest-first by filter_bundle
    return {
        "flight_number":  chosen.get("flight_number", ""),
        "airline":        chosen.get("airline", ""),
        "price":          chosen.get("price", 0),
        "departure_time": chosen.get("departure_time", ""),   # real DB value
        "arrival_time":   chosen.get("arrival_time", ""),     # real DB value
    }


def _exec_select_hotel(bundle: dict) -> dict:
    hotels = bundle.get("hotels", [])
    if not hotels:
        return {}
    chosen = hotels[0]  # already sorted stars-desc, price-asc
    return {
        "name":               chosen.get("name", ""),
        "stars":              chosen.get("stars", 0),
        "price_per_night":    chosen.get("price_per_night", 0),
        "breakfast_available":chosen.get("breakfast_available", False),
        "lat":  chosen.get("latitude") or chosen.get("lat"),
        "lng":  chosen.get("longitude") or chosen.get("lng"),
    }


def _exec_build_day(
    day_num: int,
    total_days: int,
    activities: list,
    hotel: dict,
    outbound_flight: dict,
    return_flights: list,
    already_used: set,
) -> dict:
    slots: list[DaySlot] = []
    hotel_lat = hotel.get("lat")
    hotel_lng = hotel.get("lng")
    has_breakfast = hotel.get("breakfast_available", False)

    # ── Day 1: arrival ─────────────────────────────────────────────────
    if day_num == 1:
        arrival_str = outbound_flight.get("arrival_time", "12:00")
        arrival_dt  = parse_dt(arrival_str)
        hour = arrival_dt.hour

        slots.append(DaySlot(
            time=fmt_time(arrival_dt), duration_minutes=90,
            slot_type="rest", name="Arrival & hotel check-in",
            description="Settle in and freshen up after the flight.",
            estimated_cost=0.0,
            notes="Standard check-in: 15:00. Early check-in may cost extra.",
        ))

        if hour < 12:
            # Morning arrival → rest, lunch at noon, 2 afternoon activities
            slots.append(DaySlot(
                time="12:00", duration_minutes=60, slot_type="meal",
                name="Welcome lunch",
                description="First taste of local cuisine near the hotel.",
                estimated_cost=18.0,
            ))
            _add_activities(slots, activities, already_used,
                            hotel_lat, hotel_lng,
                            start=parse_dt("13:30"), max_count=2)

        elif hour < 17:
            # Afternoon arrival → check-in, dinner, 1 evening activity
            dinner_start = arrival_dt + timedelta(hours=2)
            slots.append(DaySlot(
                time=fmt_time(dinner_start), duration_minutes=60, slot_type="meal",
                name="Welcome dinner",
                description="Enjoy dinner at a local restaurant.",
                estimated_cost=25.0,
            ))
            _add_activities(slots, activities, already_used,
                            hotel_lat, hotel_lng,
                            start=dinner_start + timedelta(hours=1, minutes=30),
                            max_count=1)
        else:
            # Late/night arrival → check-in only
            slots.append(DaySlot(
                time=fmt_time(arrival_dt + timedelta(hours=1)),
                duration_minutes=30, slot_type="rest",
                name="Rest & prepare for tomorrow",
                description="Early night before your adventures begin.",
                estimated_cost=0.0,
            ))

        theme = "Arrival & first impressions"

    # ── Last day: return flight ─────────────────────────────────────────
    elif day_num == total_days:
        if has_breakfast:
            slots.append(DaySlot(
                time="08:00", duration_minutes=60, slot_type="meal",
                name="Hotel breakfast & check-out",
                description="Final breakfast. Check out by 10:00.",
                estimated_cost=0.0,
            ))

        # Find return flight and compute airport cutoff
        return_dep_str = ""
        cutoff_dt: Optional[datetime] = None
        chosen_return: dict = {}

        if return_flights:
            chosen_return = return_flights[0]  # already sorted cheapest-first
            return_dep_str = chosen_return.get("departure_time", "")
            if return_dep_str:
                cutoff_dt = parse_dt(return_dep_str) - timedelta(hours=2, minutes=30)

        _add_activities(slots, activities, already_used,
                        hotel_lat, hotel_lng,
                        start=parse_dt("10:30"),
                        max_count=2,
                        cutoff=cutoff_dt)

        transfer_time = cutoff_dt or parse_dt("16:00")
        slots.append(DaySlot(
            time=fmt_time(transfer_time), duration_minutes=30,
            slot_type="transport", name="Transfer to airport",
            description="Head to the airport for your return flight.",
            estimated_cost=15.0,
            notes=f"Return flight departs: {return_dep_str}" if return_dep_str else None,
        ))

        theme = f"Final day & departure"

    # ── Middle days ─────────────────────────────────────────────────────
    else:
        if has_breakfast:
            slots.append(DaySlot(
                time="08:00", duration_minutes=60, slot_type="meal",
                name="Hotel breakfast",
                description="Start the day with the included breakfast.",
                estimated_cost=0.0,
            ))
        _add_activities(slots, activities, already_used,
                        hotel_lat, hotel_lng,
                        start=parse_dt("09:30" if has_breakfast else "09:00"),
                        max_count=4)
        theme = f"Day {day_num} — Explore"
    if not slots:
        slots.append(DaySlot(
            time="10:00", duration_minutes=480, slot_type="rest",
            name="Leisure time",
            description="Explore the city at your own pace or relax.",
            estimated_cost=0.0
        ))
    return {
        "day": day_num,
        "theme": theme,
        "slots": [s.model_dump() for s in slots],
        "day_cost": round(sum(s.estimated_cost for s in slots), 2),
    }


def _exec_verify_budget(results: dict, budget: float, trip_days: int, bundle: dict) -> dict:
    outbound_cost = results.get("select_flight", {}).get("price", 0)
    
    ret_flights = bundle.get("return_flights", [])
    return_cost = ret_flights[0].get("price", 0) if ret_flights else 0
    
    flight_cost   = outbound_cost + return_cost
    hotel_night   = results.get("select_hotel", {}).get("price_per_night", 0)
    hotel_cost    = hotel_night * max(1, (trip_days ))
    activity_cost = sum(
        v.get("day_cost", 0)
        for k, v in results.items()
        if k.startswith("build_day_")
    )
    total = round(flight_cost + hotel_cost + activity_cost, 2)
    return {
        "flight_cost":   flight_cost,
        "hotel_cost":    hotel_cost,
        "activity_cost": round(activity_cost, 2),
        "total_cost":    total,
        "within_budget": not budget or total <= budget * 1.10,
    }


# ---------------------------------------------------------------------------
# Activity scheduling helper
# ---------------------------------------------------------------------------

def _add_activities(
    slots: list,
    activities: list,
    already_used: set,
    hotel_lat: Optional[float],
    hotel_lng: Optional[float],
    start: datetime,
    max_count: int,
    cutoff: Optional[datetime] = None,
) -> None:
    current_time = start
    current_lat  = hotel_lat
    current_lng  = hotel_lng
    added = 0

    for act in activities:
        if added >= max_count:
            break
        name = act.get("name", "")
        if name in already_used:
            continue

        act_lat  = act.get("latitude") or act.get("lat")
        act_lng  = act.get("longitude") or act.get("lng")
        duration = act.get("avg_duration_minutes") or 90
        cost     = float(act.get("price", 0))

        # Travel slot
        travel_end = current_time
        if current_lat and current_lng and act_lat and act_lng:
            km = haversine_km(current_lat, current_lng, act_lat, act_lng)
            if km > 0.1:
                t_slot = travel_slot(name, km, current_time)
                travel_end = current_time + timedelta(minutes=t_slot.duration_minutes)
                slots.append(t_slot)

        # Respect opening hours
        open_str  = act.get("opening_time") or "09:00"
        open_dt   = parse_dt(open_str)
        if travel_end.replace(year=2000) < open_dt:
            travel_end = travel_end.replace(hour=open_dt.hour, minute=open_dt.minute)

        activity_end = travel_end + timedelta(minutes=duration)

        # Respect last-day airport cutoff
        if cutoff and activity_end > cutoff:
            continue

        notes_parts = []
        if act.get("requires_booking"):
            notes_parts.append("⚠️ Advance booking required.")
        if act.get("min_age", 0) > 0:
            notes_parts.append(f"Min age: {act['min_age']}.")

        slots.append(DaySlot(
            time=fmt_time(travel_end),
            duration_minutes=duration,
            slot_type="activity",
            name=name,
            description=act.get("categories", ""),
            estimated_cost=cost,
            lat=act_lat,
            lng=act_lng,
            notes=" ".join(notes_parts) or None,
        ))

        already_used.add(name)
        current_time = activity_end
        current_lat  = act_lat
        current_lng  = act_lng
        added += 1


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def _collect_used(results: dict) -> set:
    used = set()
    for v in results.values():
        if isinstance(v, dict) and "slots" in v:
            for s in v["slots"]:
                if s.get("slot_type") == "activity":
                    used.add(s.get("name", ""))
    return used


def _assemble(results: dict, state: AgentState, plan: ExecutionPlan) -> dict:
    flight  = results.get("select_flight", {})
    hotel   = results.get("select_hotel", {})
    budget_check = results.get("verify_budget", {})

    days = sorted(
        [v for k, v in results.items() if k.startswith("build_day_")],
        key=lambda d: d["day"],
    )

    # Best available return flight for display
    bundle = state.get("itinerary_plan", {}).get("filtered_bundle", {})
    ret_flights = bundle.get("return_flights", [])
    chosen_ret = ret_flights[0] if ret_flights else {}

    prefs = state.get("user_preferences", {})

    return {
        "destination":           plan.destination,
        "origin":                plan.origin,
        "total_days":            plan.total_days,
        "estimated_total_cost":  budget_check.get("total_cost", 0),
        "within_budget":         budget_check.get("within_budget", True),
        "selected_flight":       flight,
        "selected_return_flight": {
            "flight_number":  chosen_ret.get("flight_number", ""),
            "airline":        chosen_ret.get("airline", ""),
            "price":          chosen_ret.get("price", 0),
            "departure_time": chosen_ret.get("departure_time", ""),
            "arrival_time":   chosen_ret.get("arrival_time", ""),
        } if chosen_ret else {},
        "selected_hotel":        hotel,
        "days":                  days,
        "cost_breakdown":        budget_check,
        "user_preferences_applied": [k for k, v in prefs.items() if v],
    }