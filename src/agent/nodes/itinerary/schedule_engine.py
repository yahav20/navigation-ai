# src/agent/nodes/itinerary/schedule_engine.py
"""
ScheduleEngine — Pure-Python deterministic day planner.
No LLM involved. The LLM only *selects* activities; this module *sequences* them.

Algorithm per day:
  1.  Build a Timeline (cursor starting at day_start)
  2.  Insert arrival anchor (Day 1) or hotel-breakfast (other days)
  3.  For each candidate activity (LLM-ranked):
        a. Compute transit from current_location → activity
        b. Check window fits (activity opening, day_end)
        c. If yes → insert [transit_slot, activity_slot]
        d. Inject meal if hunger_clock >= MEAL_INTERVAL and activity has no food
  4.  Insert dinner (if not covered)
  5.  Enforce departure anchor (last day)
  6.  Return typed list of TimeSlot

Key design decisions:
  - Haversine for real distances (no Google Maps API needed)
  - Walk threshold: ≤ 1.2 km (flat city) — configurable
  - Taxi speed: 25 km/h (urban), walk speed: 4 km/h
  - Taxi minimum: 10 min (pickup + boarding), flat $8 base + $1.5/km
  - Airport-to-hotel always taxi (regardless of distance)
  - Meal injection: every 4h of active time without food
  - Rest: once per day, 20 min, after first meal post-midday
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DAY_START_DEFAULT = "08:00"
DAY_END = "22:00"

WALK_MAX_KM = 1.2          # beyond this → taxi
WALK_SPEED_KMH = 4.0
TAXI_SPEED_KMH = 25.0
TAXI_BASE_COST = 8.0       # $ fixed base fare
TAXI_COST_PER_KM = 1.5
TAXI_MIN_MINUTES = 10      # minimum taxi trip duration (waiting + boarding)
AIRPORT_HOTEL_MIN_MINUTES = 45   # always taxi, at least 45 min

MEAL_INTERVAL_HOURS = 4.0  # inject meal if hungry for this long
BREAKFAST_DURATION = 40    # minutes
LUNCH_DURATION = 60
DINNER_DURATION = 75
REST_DURATION = 25

MEAL_COSTS = {
    "breakfast": 12.0,
    "lunch": 22.0,
    "dinner": 30.0,
}

HOTEL_CHECKIN_DURATION = 30    # minutes to settle in after arrival

SlotType = Literal["activity", "meal", "rest", "transport", "checkin"]

# ---------------------------------------------------------------------------
# Data Types
# ---------------------------------------------------------------------------

@dataclass
class GeoPoint:
    lat: float
    lng: float
    name: str = ""

@dataclass
class TimeSlot:
    start: datetime
    end: datetime
    slot_type: SlotType
    name: str
    description: str = ""
    estimated_cost: float = 0.0
    transport_mode: Optional[Literal["walk", "taxi"]] = None   # only for transport slots
    distance_km: Optional[float] = None

    @property
    def duration_minutes(self) -> int:
        return int((self.end - self.start).total_seconds() / 60)

    def to_dict(self) -> dict:
        d = {
            "time": self.start.strftime("%H:%M"),
            "end_time": self.end.strftime("%H:%M"),
            "duration_minutes": self.duration_minutes,
            "slot_type": self.slot_type,
            "name": self.name,
            "description": self.description,
            "estimated_cost": round(self.estimated_cost, 2),
        }
        if self.transport_mode:
            d["transport_mode"] = self.transport_mode
        if self.distance_km is not None:
            d["distance_km"] = round(self.distance_km, 2)
        return d


# Keywords in categories that mean the activity IS a meal/food venue,
# not just an attraction that happens to have a café inside.
_MEAL_CATEGORY_KEYWORDS = {
    "restaurant", "food", "market", "café", "cafe", "dining",
    "breakfast", "lunch", "dinner", "brunch", "bistro",
    "brasserie", "eatery", "culinary", "gastronomy",
}


def _is_meal_activity(categories: str) -> bool:
    """Return True when the activity IS a dining/food experience."""
    cats_lower = categories.lower()
    return any(kw in cats_lower for kw in _MEAL_CATEGORY_KEYWORDS)


@dataclass
class ActivityCandidate:
    """Normalized activity from DB, ready for scheduling."""
    name: str
    lat: float
    lng: float
    duration_minutes: int
    price: float
    opening_time: str     # "HH:MM"
    closing_time: str     # "HH:MM"
    food_available: bool
    categories: str
    rating: float
    requires_booking: bool = False
    operating_days: str = "Daily"

    @property
    def is_meal_venue(self) -> bool:
        """True when this activity IS the meal (restaurant, market, etc.).
        Distinct from food_available=True, which means an attraction that
        also has food (e.g. a theme park with a snack bar).
        """
        return _is_meal_activity(self.categories)


# ---------------------------------------------------------------------------
# Geo helpers
# ---------------------------------------------------------------------------

def haversine_km(a: GeoPoint, b: GeoPoint) -> float:
    """Real-world distance between two lat/lng points (km)."""
    R = 6371.0
    lat1, lon1 = math.radians(a.lat), math.radians(a.lng)
    lat2, lon2 = math.radians(b.lat), math.radians(b.lng)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(h))


def transit_plan(
    frm: GeoPoint,
    to: GeoPoint,
    force_taxi: bool = False,
) -> tuple[Literal["walk", "taxi"], float, float]:
    """
    Returns (mode, duration_minutes, cost).
    force_taxi: airport transfers, long-distance, etc.
    """
    dist = haversine_km(frm, to)

    if dist == 0.0:
        return "walk", 0, 0.0

    if not force_taxi and dist <= WALK_MAX_KM:
        minutes = max(5, (dist / WALK_SPEED_KMH) * 60)
        return "walk", round(minutes), 0.0
    else:
        minutes = max(TAXI_MIN_MINUTES, (dist / TAXI_SPEED_KMH) * 60)
        cost = TAXI_BASE_COST + dist * TAXI_COST_PER_KM
        return "taxi", round(minutes), round(cost, 2)


# ---------------------------------------------------------------------------
# Timeline cursor
# ---------------------------------------------------------------------------

def _hm(t: datetime) -> str:
    return t.strftime("%H:%M")

def _parse_time(base_date: datetime, hm: str) -> datetime:
    """Parse 'HH:MM' string into a datetime on the same calendar date."""
    h, m = map(int, hm.split(":"))
    return base_date.replace(hour=h, minute=m, second=0, microsecond=0)


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

class DayScheduleBuilder:
    """
    Builds a single day's schedule deterministically.

    Usage:
        builder = DayScheduleBuilder(day_config)
        slots = builder.build(activity_candidates)
    """

    def __init__(self, cfg: "DayConfig") -> None:
        self.cfg = cfg
        self._slots: list[TimeSlot] = []
        # Hunger clock: hours since last food event
        self._last_food_time: Optional[datetime] = None
        self._had_rest = False
        self._had_lunch = False
        self._had_dinner = False

        # Reference date (arbitrary; only HH:MM matters)
        self._date = datetime(2000, 1, 1)

    # ── Public entry point ────────────────────────────────────────────────

    def build(self, candidates: list[ActivityCandidate]) -> list[dict]:
        cfg = self.cfg

        day_end = _parse_time(self._date, DAY_END)
        departure_anchor = (
            _parse_time(self._date, cfg.departure_time) - timedelta(minutes=150)
            if cfg.is_last_day and cfg.departure_time
            else day_end
        )

        # ── Cursor: current time & location ──
        cursor = _parse_time(self._date, DAY_START_DEFAULT)
        location = GeoPoint(cfg.hotel_lat, cfg.hotel_lng, "Hotel")

        # ── Day 1 arrival sequence ──
        if cfg.is_first_day and cfg.arrival_time:
            cursor, location = self._insert_arrival_sequence(cfg, cursor, location)
        else:
            cursor = self._insert_breakfast(cfg, cursor)
            self._last_food_time = cursor  # breakfast just happened

        # ── 1. הפרדת המסעדות מהאטרקציות ──
        meal_candidates = [act for act in candidates if act.is_meal_venue]
        regular_candidates = [act for act in candidates if not act.is_meal_venue]

        # ── Main activity loop ──
        used_names: set[str] = set()

        for act in regular_candidates:
            if cursor >= departure_anchor:
                break

            # ─ Hunger check: inject meal before activity if needed ─
            if self._is_hungry(cursor) and not act.food_available:
                # מעבירים את המיקום ואת רשימת המסעדות הפנויות
                cursor, location = self._inject_meal(cursor, departure_anchor, location, meal_candidates)
                if cursor >= departure_anchor:
                    break

            # ─ Rest injection (once per day, around midday) ─
            if not self._had_rest and cursor.hour >= 13 and cursor.hour <= 14:
                cursor = self._inject_rest(cursor)
                self._had_rest = True

            # ─ Transit to activity ─
            act_point = GeoPoint(act.lat, act.lng, act.name)
            mode, transit_min, transit_cost = transit_plan(location, act_point)

            transit_end = cursor + timedelta(minutes=transit_min)
            act_start = transit_end

            # ─ Check activity opening window ─
            opening = _parse_time(self._date, act.opening_time)
            if act_start < opening:
                act_start = opening  # wait for opening
                transit_end = opening - timedelta(minutes=1)  # recalc transit start

            act_end = act_start + timedelta(minutes=act.duration_minutes)

            # ─ Fits in the day? ─
            if act_end > departure_anchor:
                # Try a shorter version (min 45 min)
                short_end = act_start + timedelta(minutes=45)
                if short_end > departure_anchor:
                    break  # skip entirely
                act_end = short_end

            # ─ Closing time check ─
            closing = _parse_time(self._date, act.closing_time)
            if act_end > closing:
                break

            # ─ Insert transit slot ─
            if transit_min > 0:
                label = f"Transit to {act.name}"
                self._push(TimeSlot(
                    start=cursor,
                    end=transit_end,
                    slot_type="transport",
                    name=label,
                    description=f"{mode} · {haversine_km(location, act_point):.1f} km",
                    estimated_cost=transit_cost,
                    transport_mode=mode,
                    distance_km=haversine_km(location, act_point),
                ))

            # ─ Insert activity slot ─
            self._push(TimeSlot(
                start=act_start,
                end=act_end,
                slot_type="activity",
                name=act.name,
                description=act.categories,
                estimated_cost=act.price,
            ))

            if act.food_available:
                self._last_food_time = act_end  # activity fed the user
                if act_start.hour >= 11 and act_start.hour <= 15:
                    self._had_lunch = True
                elif act_start.hour >= 18:
                    self._had_dinner = True

            cursor = act_end
            location = act_point
            used_names.add(act.name)

        # ── Final dinner ──
        # מוודאים שהשעה היא לפחות 18:00 לפני שמוסיפים ארוחת ערב
        if not self._had_dinner and cursor < departure_anchor and cursor.hour >= 18:
            cursor, location = self._inject_dinner(cursor, departure_anchor, location, cfg, meal_candidates)

        return [s.to_dict() for s in self._slots]

    # ── Arrival sequence (Day 1 only) ─────────────────────────────────────

    def _insert_arrival_sequence(
        self, cfg: "DayConfig", cursor: datetime, location: GeoPoint
    ) -> tuple[datetime, GeoPoint]:
        arrival = _parse_time(self._date, cfg.arrival_time)

        # Airport is a special location (no coordinates → force taxi)
        airport = GeoPoint(cfg.hotel_lat + 0.3, cfg.hotel_lng + 0.3, "Airport")
        # We approximate airport ~30km away; override with AIRPORT_HOTEL_MIN_MINUTES
        taxi_min = max(AIRPORT_HOTEL_MIN_MINUTES, 45)
        taxi_cost = TAXI_BASE_COST + 30 * TAXI_COST_PER_KM

        transfer_end = arrival + timedelta(minutes=taxi_min)
        self._push(TimeSlot(
            start=arrival,
            end=transfer_end,
            slot_type="transport",
            name="Airport → Hotel transfer",
            description="Taxi from airport to hotel",
            estimated_cost=taxi_cost,
            transport_mode="taxi",
        ))

        checkin_end = transfer_end + timedelta(minutes=HOTEL_CHECKIN_DURATION)
        self._push(TimeSlot(
            start=transfer_end,
            end=checkin_end,
            slot_type="checkin",
            name=f"Check-in · {cfg.hotel_name}",
            description="Settle in, freshen up",
            estimated_cost=0.0,
        ))

        hotel_loc = GeoPoint(cfg.hotel_lat, cfg.hotel_lng, cfg.hotel_name)

        # Early arrival (before 10:00) → light breakfast if not included
        if checkin_end.hour < 10:
            bk_end = self._insert_breakfast(cfg, checkin_end)
            return bk_end, hotel_loc

        # Late arrival (12:00+) → skip to lunch zone
        if checkin_end.hour >= 12:
            self._last_food_time = checkin_end  # will trigger lunch injection shortly

        return checkin_end, hotel_loc

    # ── Meal injectors ─────────────────────────────────────────────────────

    def _insert_breakfast(self, cfg: "DayConfig", cursor: datetime) -> datetime:
        cost = 0.0 if cfg.hotel_has_breakfast else MEAL_COSTS["breakfast"]
        name = f"Breakfast · {cfg.hotel_name}" if cfg.hotel_has_breakfast else "Breakfast at local café"
        end = cursor + timedelta(minutes=BREAKFAST_DURATION)
        self._push(TimeSlot(
            start=cursor, end=end,
            slot_type="meal", name=name,
            description="Start the day right",
            estimated_cost=cost,
        ))
        self._last_food_time = end
        return end

    def _inject_meal(self, cursor: datetime, hard_limit: datetime, location: GeoPoint, meal_candidates: list[ActivityCandidate]) -> tuple[datetime, GeoPoint]:
        """Inject lunch or dinner depending on time of day, using real DB venues if available."""
        is_dinner = cursor.hour >= 17 or self._had_lunch

        # 1. מנסים למצוא מסעדה אמיתית מה-DB
        for i, meal in enumerate(meal_candidates):
            opening_hr = int(meal.opening_time.split(":")[0])
            # סינון קל: לא ניקח קרוז ערב לצהריים, ולא ניקח בית קפה של בוקר לערב
            if not is_dinner and opening_hr >= 17: continue
            if is_dinner and meal.closing_time < "18:00": continue

            meal_candidates.pop(i)
            meal_pt = GeoPoint(meal.lat, meal.lng, meal.name)
            mode, t_min, t_cost = transit_plan(location, meal_pt)
            
            start = cursor + timedelta(minutes=t_min)
            opening = _parse_time(self._date, meal.opening_time)
            if start < opening:
                start = opening
            
            end = start + timedelta(minutes=meal.duration_minutes)
            if end > hard_limit:
                end = hard_limit
                
            if t_min > 0:
                self._push(TimeSlot(start=cursor, end=start, slot_type="transport", name=f"Transit to {meal.name}", estimated_cost=t_cost, transport_mode=mode, distance_km=haversine_km(location, meal_pt)))
                
            self._push(TimeSlot(start=start, end=end, slot_type="meal", name=meal.name, description=meal.categories, estimated_cost=meal.price))
            
            if is_dinner: self._had_dinner = True
            else: self._had_lunch = True
            
            self._last_food_time = end
            return end, meal_pt # מעדכן את המיקום למסעדה!

        # 2. גיבוי - אם נגמרו המסעדות ב-DB, שמים בלוק גנרי
        if cursor.hour < 15 and not self._had_lunch:
            name, cost, dur = "Lunch", MEAL_COSTS["lunch"], LUNCH_DURATION
            self._had_lunch = True
        elif cursor.hour >= 17 and not self._had_dinner:
            name, cost, dur = "Dinner", MEAL_COSTS["dinner"], DINNER_DURATION
            self._had_dinner = True
        else:
            name, cost, dur = "Snack / coffee break", 8.0, 20

        end = cursor + timedelta(minutes=dur)
        if end > hard_limit:
            end = hard_limit
        self._push(TimeSlot(start=cursor, end=end, slot_type="meal", name=name, description="Local restaurant", estimated_cost=cost))
        self._last_food_time = end
        return end, location


    def _inject_dinner(self, cursor: datetime, hard_limit: datetime, location: GeoPoint, cfg: "DayConfig", meal_candidates: list[ActivityCandidate]) -> tuple[datetime, GeoPoint]:
        # 1. מנסים למצוא מסעדה אמיתית מה-DB
        for i, meal in enumerate(meal_candidates):
            if meal.closing_time < "18:00": continue # מדלגים על מקומות שסגורים בערב
            
            meal_candidates.pop(i)
            meal_pt = GeoPoint(meal.lat, meal.lng, meal.name)
            mode, t_min, t_cost = transit_plan(location, meal_pt)
            
            start = cursor + timedelta(minutes=t_min)
            opening = _parse_time(self._date, meal.opening_time)
            if start < opening:
                start = opening
                
            end = min(start + timedelta(minutes=meal.duration_minutes), hard_limit)
            
            if t_min > 0:
                self._push(TimeSlot(start=cursor, end=start, slot_type="transport", name=f"Transit to {meal.name}", estimated_cost=t_cost, transport_mode=mode, distance_km=haversine_km(location, meal_pt)))
                
            self._push(TimeSlot(start=start, end=end, slot_type="meal", name=meal.name, description=meal.categories, estimated_cost=meal.price))
            self._had_dinner = True
            self._last_food_time = end
            return end, meal_pt

        # 2. גיבוי - מסעדה גנרית קרובה למלון
        hotel = GeoPoint(cfg.hotel_lat, cfg.hotel_lng, cfg.hotel_name)
        mode, t_min, t_cost = transit_plan(location, hotel)
        dinner_start = cursor + timedelta(minutes=t_min)

        if dinner_start >= hard_limit:
            return cursor, location

        if t_min > 0:
            self._push(TimeSlot(start=cursor, end=dinner_start, slot_type="transport", name=f"Return to hotel area", description=f"{mode} back", estimated_cost=t_cost, transport_mode=mode))

        dinner_end = min(dinner_start + timedelta(minutes=DINNER_DURATION), hard_limit)
        self._push(TimeSlot(start=dinner_start, end=dinner_end, slot_type="meal", name="Dinner", description="Local restaurant near hotel", estimated_cost=MEAL_COSTS["dinner"]))
        self._had_dinner = True
        self._last_food_time = dinner_end
        return dinner_end, hotel
    
    def _inject_rest(self, cursor: datetime) -> datetime:
        end = cursor + timedelta(minutes=REST_DURATION)
        self._push(TimeSlot(
            start=cursor, end=end,
            slot_type="rest", name="Afternoon rest",
            description="Recharge before the evening",
            estimated_cost=0.0,
        ))
        return end

    # ── Internal helpers ───────────────────────────────────────────────────

    def _is_hungry(self, cursor: datetime) -> bool:
        if self._last_food_time is None:
            return cursor.hour >= 12  # arrived with no breakfast → hungry at noon
        elapsed = (cursor - self._last_food_time).total_seconds() / 3600
        return elapsed >= MEAL_INTERVAL_HOURS

    def _push(self, slot: TimeSlot) -> None:
        self._slots.append(slot)


# ---------------------------------------------------------------------------
# DayConfig — input struct for the builder
# ---------------------------------------------------------------------------

@dataclass
class DayConfig:
    day_number: int
    total_days: int
    hotel_name: str
    hotel_lat: float
    hotel_lng: float
    hotel_has_breakfast: bool = False

    is_first_day: bool = False
    arrival_time: Optional[str] = None      # "HH:MM" — only Day 1

    is_last_day: bool = False
    departure_time: Optional[str] = None    # "HH:MM" — only last day