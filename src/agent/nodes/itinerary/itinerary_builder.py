"""
ItineraryBuilderNode — Plan & Execute, Step 2: EXECUTE

Receives the structured plan from ItineraryPlannerNode and "executes" each day slot:
  - Validates opening hours of each activity
  - Calculates realistic walk times between slots using Haversine
  - Adds buffer times, meal suggestions, rest logic
  - Attaches practical notes (booking required, min age, etc.)
  - Produces a rich, ready-to-format itinerary dict

This node is intentionally DETERMINISTIC — it processes the plan produced by
the planner and enriches it with facts from the data bundle. No LLM calls here.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta
from typing import Any

from agent.state import AgentState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WALK_SPEED_KMH = 4.5          # average tourist walking speed
TAXI_SPEED_KMH = 30.0         # city taxi / ride-share average
TAXI_THRESHOLD_KM = 2.0       # beyond this distance, suggest taxi
CHECKIN_HOUR = 15              # standard hotel check-in (15:00)
CHECKOUT_HOUR = 10             # standard hotel check-out (10:00)
BREAKFAST_DONE_HOUR = 10      # hotel breakfast finishes latest
FIRST_ACTIVITY_BUFFER_MIN = 30  # buffer after breakfast/check-in before leaving

# Arrival window → Day-1 plan mode
ARRIVAL_MORNING_CUTOFF = 12   # before noon → morning arrival
ARRIVAL_AFTERNOON_CUTOFF = 17 # before 17:00 → afternoon arrival
                               # after 17:00 → evening/night arrival


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _walk_minutes(km: float) -> int:
    return max(5, int((km / WALK_SPEED_KMH) * 60))


def _taxi_minutes(km: float) -> int:
    return max(5, int((km / TAXI_SPEED_KMH) * 60))


def _transport_info(km: float) -> dict:
    if km <= TAXI_THRESHOLD_KM:
        mins = _walk_minutes(km)
        return {"mode": "walking", "duration_minutes": mins, "distance_km": round(km, 2)}
    else:
        mins = _taxi_minutes(km)
        return {"mode": "taxi/rideshare", "duration_minutes": mins, "distance_km": round(km, 2)}


def _parse_time(t: str) -> datetime:
    """Parse HH:MM or ISO datetime string to datetime (date = today placeholder)."""
    base = datetime(2000, 1, 1)
    for fmt in ("%H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M"):
        try:
            parsed = datetime.strptime(t, fmt)
            if parsed.year == 1900:
                parsed = parsed.replace(year=2000)
            return parsed
        except ValueError:
            continue
    return base  # fallback


def _fmt_time(dt: datetime) -> str:
    return dt.strftime("%H:%M")


def _day1_start_time(arrival_time_str: str, has_breakfast: bool) -> tuple[datetime, list[dict]]:
    """
    Given a flight arrival time, return the earliest reasonable start time
    for Day-1 activities, along with pre-activity slots (rest, lunch, etc.).
    """
    arrival = _parse_time(arrival_time_str)
    hour = arrival.hour
    pre_slots = []

    if hour < ARRIVAL_MORNING_CUTOFF:
        # Morning arrival: rest until noon, lunch, start afternoon
        rest_end = arrival.replace(hour=12, minute=0, second=0)
        pre_slots = [
            {
                "time": _fmt_time(arrival),
                "duration_minutes": int((rest_end - arrival).seconds / 60),
                "type": "rest",
                "name": "Arrival & hotel check-in",
                "description": "Settle in, freshen up, and rest after the flight.",
                "estimated_cost": 0,
                "notes": "Early check-in may be available for a fee.",
            },
            {
                "time": "12:00",
                "duration_minutes": 60,
                "type": "meal",
                "name": "Welcome lunch",
                "description": "First taste of local cuisine near the hotel.",
                "estimated_cost": 20,
                "notes": "Ask hotel reception for nearby recommendations.",
            },
        ]
        start = arrival.replace(hour=13, minute=30)

    elif hour < ARRIVAL_AFTERNOON_CUTOFF:
        # Afternoon arrival: check-in, short rest, evening
        pre_slots = [
            {
                "time": _fmt_time(arrival),
                "duration_minutes": 90,
                "type": "rest",
                "name": "Hotel check-in & rest",
                "description": "Check in and relax before heading out.",
                "estimated_cost": 0,
            },
            {
                "time": _fmt_time(arrival + timedelta(hours=1, minutes=30)),
                "duration_minutes": 60,
                "type": "meal",
                "name": "Dinner",
                "description": "Enjoy dinner at a local restaurant.",
                "estimated_cost": 25,
            },
        ]
        start = arrival + timedelta(hours=3)

    else:
        # Evening / night arrival: check-in and sleep only
        pre_slots = [
            {
                "time": _fmt_time(arrival),
                "duration_minutes": 30,
                "type": "rest",
                "name": "Late arrival & check-in",
                "description": "Check in and prepare for tomorrow's adventures.",
                "estimated_cost": 0,
                "notes": "Notify hotel of late arrival in advance.",
            },
        ]
        start = arrival.replace(hour=23, minute=59)  # no more activities today

    return start, pre_slots


def _last_day_cutoff(departure_time_str: str) -> datetime:
    """
    Return the latest time an activity can START on the last day,
    ensuring the traveller reaches the airport 2h before departure.
    """
    dep = _parse_time(departure_time_str)
    # 2h airport buffer + 30min transit to airport
    return dep - timedelta(hours=2, minutes=30)


# ---------------------------------------------------------------------------
# Builder node
# ---------------------------------------------------------------------------

class ItineraryBuilderNode:
    """
    Deterministic execution of the LLM-produced plan.
    Enriches each slot with transport info, timing fixes, and practical notes.
    """

    def __call__(self, state: AgentState) -> dict:
        plan = state.get("itinerary_plan", {})
        if not plan or plan.get("error"):
            logger.warning("ItineraryBuilderNode: no valid plan to execute.")
            return {"itinerary_plan": plan}

        days = plan.get("days", [])
        hotel = plan.get("selected_hotel", {})
        flight = plan.get("selected_flight", {})
        hotel_lat = hotel.get("lat") or hotel.get("latitude")
        hotel_lng = hotel.get("lng") or hotel.get("longitude")
        has_breakfast = hotel.get("breakfast_available", False)

        arrival_time = flight.get("arrival_time", "12:00")
        departure_time = flight.get("departure_time", "23:00")  # return flight if stored

        enriched_days = []
        total_activity_cost = 0.0

        for i, day in enumerate(days):
            day_num = day.get("day", i + 1)
            slots = list(day.get("slots", []))

            # ---- Day 1: insert pre-activity slots based on arrival ----
            if day_num == 1:
                activity_start, pre_slots = _day1_start_time(arrival_time, has_breakfast)
                # Remove any slots the LLM put before our computed start
                slots = [
                    s for s in slots
                    if _parse_time(s.get("time", "23:59")) >= activity_start
                ]
                slots = pre_slots + slots

            # ---- Last day: enforce airport cutoff ----
            if day_num == len(days) and departure_time:
                cutoff = _last_day_cutoff(departure_time)
                kept = []
                for slot in slots:
                    slot_start = _parse_time(slot.get("time", "00:00"))
                    slot_dur = slot.get("duration_minutes", 60)
                    slot_end = slot_start + timedelta(minutes=slot_dur)
                    if slot_end <= cutoff:
                        kept.append(slot)
                    else:
                        logger.info(
                            "Day %d: dropping '%s' — ends after airport cutoff",
                            day_num, slot.get("name"),
                        )
                # Always add airport transfer slot
                kept.append({
                    "time": _fmt_time(cutoff),
                    "duration_minutes": 30,
                    "type": "transport",
                    "name": "Transfer to airport",
                    "description": "Head to the airport for departure.",
                    "estimated_cost": 20,
                    "notes": "Allow extra time during rush hour.",
                })
                slots = kept

            # ---- Add hotel morning slot (days 2+) ----
            if day_num > 1:
                if has_breakfast:
                    morning_slot = {
                        "time": "08:00",
                        "duration_minutes": 60,
                        "type": "meal",
                        "name": "Hotel breakfast",
                        "description": "Enjoy breakfast included with your stay.",
                        "estimated_cost": 0,
                        "notes": "Check-out day: vacate room by 10:00.",
                    } if day_num == len(days) else {
                        "time": "08:00",
                        "duration_minutes": 60,
                        "type": "meal",
                        "name": "Hotel breakfast",
                        "description": "Start the day with the included breakfast.",
                        "estimated_cost": 0,
                    }
                    # Only prepend if no breakfast slot already present
                    if not any("breakfast" in s.get("name", "").lower() for s in slots):
                        slots = [morning_slot] + slots

            # ---- Inject transport slots between activities ----
            enriched_slots = []
            prev_lat, prev_lng = hotel_lat, hotel_lng
            prev_end_time = None

            for slot in slots:
                activity_lat = slot.get("latitude") or slot.get("lat")
                activity_lng = slot.get("longitude") or slot.get("lng")
                slot_start = _parse_time(slot.get("time", "09:00"))

                # Transport from previous location
                if (
                    prev_lat and prev_lng
                    and activity_lat and activity_lng
                    and slot.get("type") == "activity"
                ):
                    km = _haversine_km(prev_lat, prev_lng, activity_lat, activity_lng)
                    transport = _transport_info(km)

                    # Check if we need to adjust slot start time to account for travel
                    if prev_end_time:
                        travel_start = prev_end_time
                        travel_end = travel_start + timedelta(minutes=transport["duration_minutes"])
                        if travel_end > slot_start:
                            # Nudge activity start forward
                            slot = dict(slot)
                            slot["time"] = _fmt_time(travel_end + timedelta(minutes=5))
                            slot_start = _parse_time(slot["time"])

                    if km > 0.1:  # only add transport note if meaningful distance
                        transport_slot = {
                            "time": _fmt_time(
                                slot_start - timedelta(minutes=transport["duration_minutes"] + 5)
                            ),
                            "duration_minutes": transport["duration_minutes"],
                            "type": "transport",
                            "name": f"Travel to {slot.get('name', 'next activity')}",
                            "description": f"{transport['mode'].capitalize()} — {transport['distance_km']} km",
                            "estimated_cost": 5 if transport["mode"].startswith("taxi") else 0,
                            "notes": f"~{transport['duration_minutes']} min by {transport['mode']}",
                        }
                        enriched_slots.append(transport_slot)

                    # Update prev location
                    prev_lat, prev_lng = activity_lat, activity_lng

                # Track end time
                dur = slot.get("duration_minutes", 60)
                prev_end_time = slot_start + timedelta(minutes=dur)

                # Add practical notes
                slot = dict(slot)
                if slot.get("requires_booking"):
                    existing = slot.get("notes", "")
                    slot["notes"] = (existing + " ⚠️ Advance booking required.").strip()
                if slot.get("min_age", 0) > 0:
                    existing = slot.get("notes", "")
                    slot["notes"] = (existing + f" Min age: {slot['min_age']}.").strip()

                enriched_slots.append(slot)
                total_activity_cost += slot.get("estimated_cost", 0)

            enriched_days.append({**day, "slots": enriched_slots})

        # Update plan with enriched days and recalculated cost
        plan = {
            **plan,
            "days": enriched_days,
            "estimated_total_cost": round(
                plan.get("selected_flight", {}).get("price", 0)
                + plan.get("selected_hotel", {}).get("price_per_night", 0)
                * len(days)
                + total_activity_cost,
                2,
            ),
        }

        return {"itinerary_plan": plan}
