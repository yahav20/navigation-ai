"""
ScheduleEngine — Pure-Python deterministic day builder.
No LLM involved here. The LLM (ActivitySelector) decides *what* goes where;
this module decides *when* and produces the final timestamped timeline.

Consumed inputs
---------------
  candidates  : list[ActivityCandidate]   — sightseeing + pinned meal venues
  day_plan    : dict                      — from ActivitySelector, contains:
                  activities, lunch_restaurant, coffee_place, dinner_restaurant,
                  recommended_rest_blocks
  DayConfig   : weather_condition, blocked_times, hotel info, arrival/departure

Algorithm per day
-----------------
  1.  Cursor starts at day_start_time from hotel location
  2.  Day 1 → arrival taxi + check-in; other days → breakfast
  3.  Register blocked windows + LLM rest suggestions as no-schedule zones
  4.  Extreme heat → auto-add 14:00-16:00 hotel rest
  5.  Main sightseeing loop:
        a. Flush any pending rest block whose window is now due
        b. Insert pinned coffee ~10:00 if window reached and not yet done
        c. Insert pinned lunch ~12:00 if window reached and not yet done
        d. Hunger check → inject meal from candidate list or placeholder
        e. Legacy midday rest (13-14) if no LLM rest blocks provided
        f. Compute transit; skip activity if it overflows day / blocked
        g. Insert [transit, activity] pair; advance cursor
  6.  End of sightseeing:
        • Flush remaining rest blocks
        • Insert pinned dinner if cursor ≥ 18:00
        • "Free time" gap slot if finishing early
        • Fallback dinner from candidate list or near-hotel placeholder
  7.  Last-day departure taxi

Design principles
-----------------
  • Haversine for real distance (no API calls)
  • Walk ≤ 1.2 km / Taxi above that
  • Transit is never skipped — always a visible slot
  • Meal injection never leaves the user hungry >4 h
  • Rainy days: outdoor activities removed before the loop
  • All "human" logic is deterministic Python — no LLM in this file
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DAY_START_DEFAULT      = "08:00"
DAY_END                = "22:00"

WALK_MAX_KM            = 1.2          # farther → taxi
WALK_SPEED_KMH         = 4.0
TAXI_SPEED_KMH         = 25.0
TAXI_BASE_COST         = 8.0          # USD fixed base fare
TAXI_COST_PER_KM       = 1.5
TAXI_MIN_MINUTES       = 10           # min taxi trip (wait + board)
AIRPORT_HOTEL_MINUTES  = 45           # always taxi, at least 45 min

MEAL_INTERVAL_HOURS    = 4.0          # inject meal if hungry this long
BREAKFAST_DURATION     = 40
LUNCH_DURATION         = 60
DINNER_DURATION        = 75
COFFEE_DURATION        = 30
REST_DURATION          = 25           # legacy quick rest
HOTEL_CHECKIN_DURATION = 30

MEAL_COSTS = {
    "breakfast": 12.0,
    "lunch":     22.0,
    "dinner":    30.0,
    "coffee":     8.0,
}

SlotType = Literal["activity", "meal", "rest", "transport", "checkin"]

# ---------------------------------------------------------------------------
# Keyword sets
# ---------------------------------------------------------------------------

# Outdoor activities skipped on rainy / stormy days
_OUTDOOR_KW = {
    "beach", "park", "garden", "promenade", "walk", "hike", "trail",
    "viewpoint", "outdoor", "open-air", "open air", "market", "bazaar",
    "boardwalk", "boat", "kayak", "surf", "cycling", "bike", "tour",
    "rooftop", "terrace", "square", "plaza",
}

# Keywords that make an activity a DINING venue (not just a place with food)
_MEAL_KW = {
    "restaurant", "food", "café", "cafe", "dining", "eatery",
    "breakfast", "lunch", "dinner", "brunch", "bistro",
    "brasserie", "culinary", "gastronomy", "coffee", "bakery",
    "bar", "pub", "tavern", "trattoria", "pizzeria",
}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class GeoPoint:
    lat:  float
    lng:  float
    name: str = ""


@dataclass
class TimeSlot:
    start:          datetime
    end:            datetime
    slot_type:      SlotType
    name:           str
    description:    str   = ""
    estimated_cost: float = 0.0
    transport_mode: Optional[Literal["walk", "taxi"]] = None
    distance_km:    Optional[float] = None

    @property
    def duration_minutes(self) -> int:
        return int((self.end - self.start).total_seconds() / 60)

    def to_dict(self) -> dict:
        d: dict = {
            "time":             self.start.strftime("%H:%M"),
            "end_time":         self.end.strftime("%H:%M"),
            "duration_minutes": self.duration_minutes,
            "slot_type":        self.slot_type,
            "name":             self.name,
            "description":      self.description,
            "estimated_cost":   round(self.estimated_cost, 2),
        }
        if self.transport_mode:
            d["transport_mode"] = self.transport_mode
        if self.distance_km is not None:
            d["distance_km"] = round(self.distance_km, 2)
        return d


def _is_meal_venue(categories: str) -> bool:
    cl = categories.lower()
    return any(kw in cl for kw in _MEAL_KW)


def _is_outdoor(categories: str, name: str) -> bool:
    text = (categories + " " + name).lower()
    return any(kw in text for kw in _OUTDOOR_KW)


@dataclass
class ActivityCandidate:
    """Normalised activity / restaurant ready for scheduling."""
    name:             str
    lat:              float
    lng:              float
    duration_minutes: int
    price:            float
    opening_time:     str    # "HH:MM"
    closing_time:     str    # "HH:MM"
    food_available:   bool
    categories:       str
    rating:           float
    requires_booking: bool = False
    operating_days:   str  = "Daily"

    @property
    def is_meal_venue(self) -> bool:
        return _is_meal_venue(self.categories)

    @property
    def is_outdoor(self) -> bool:
        return _is_outdoor(self.categories, self.name)


# ---------------------------------------------------------------------------
# DayConfig  (all inputs the builder needs)
# ---------------------------------------------------------------------------

@dataclass
class DayConfig:
    day_number: int
    total_days: int

    hotel_name:          str   = "Hotel"
    hotel_lat:           float = 0.0
    hotel_lng:           float = 0.0
  

    is_first_day:   bool         = False
    arrival_time:   Optional[str] = None   # "HH:MM" — Day 1 only
    is_last_day:    bool         = False
    departure_time: Optional[str] = None   # "HH:MM" — last day only

    # Schedule window (from user_preferences or defaults)
    day_start_time: str = DAY_START_DEFAULT
    day_end_time:   str = DAY_END

    # Planning context
    weather_condition:     str        = ""
    blocked_times:         list[dict] = field(default_factory=list)
    suggested_rest_blocks: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Geo / transit helpers
# ---------------------------------------------------------------------------

def haversine_km(a: GeoPoint, b: GeoPoint) -> float:
    """Straight-line distance between two lat/lng points in km."""
    R = 6371.0
    lat1, lon1 = math.radians(a.lat), math.radians(a.lng)
    lat2, lon2 = math.radians(b.lat), math.radians(b.lng)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(h))


def transit_plan(
    frm: GeoPoint,
    to:  GeoPoint,
    force_taxi: bool = False,
) -> tuple[Literal["walk", "taxi"], int, float]:
    """Return (mode, duration_minutes, cost_usd)."""
    dist = haversine_km(frm, to)
    if dist == 0.0:
        return "walk", 0, 0.0
    if not force_taxi and dist <= WALK_MAX_KM:
        mins = max(5, (dist / WALK_SPEED_KMH) * 60)
        return "walk", round(mins), 0.0
    mins = max(TAXI_MIN_MINUTES, (dist / TAXI_SPEED_KMH) * 60)
    cost = TAXI_BASE_COST + dist * TAXI_COST_PER_KM
    return "taxi", round(mins), round(cost, 2)


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def _pt(base: datetime, hm: str) -> datetime:
    """Parse 'HH:MM' into a datetime on the given base date."""
    h, m = map(int, hm.split(":"))
    return base.replace(hour=h, minute=m, second=0, microsecond=0)


def _overlaps(
    start: datetime, end: datetime,
    blocked: list[tuple[datetime, datetime]],
) -> bool:
    return any(start < be and end > bs for bs, be in blocked)


def _next_free(
    cursor: datetime,
    dur:    int,
    blocked: list[tuple[datetime, datetime]],
    limit:  datetime,
) -> Optional[datetime]:
    """First slot ≥ cursor of length `dur` that avoids all blocked windows."""
    t = cursor
    for _ in range(30):
        end = t + timedelta(minutes=dur)
        if end > limit:
            return None
        if not _overlaps(t, end, blocked):
            return t
        for bs, be in blocked:
            if t < be and end > bs:
                t = be
                break
    return None


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class DayScheduleBuilder:
    """
    Build a single day's timestamped schedule.

    Usage
    -----
        cfg     = DayConfig(...)
        builder = DayScheduleBuilder(cfg)
        slots   = builder.build(candidates, day_plan=day_plan_dict)
        # slots → list[dict]  (each dict is a TimeSlot.to_dict())
    """

    def __init__(self, cfg: DayConfig) -> None:
        self.cfg             = cfg
        self._slots:          list[TimeSlot]      = []
        self._last_food_time: Optional[datetime]  = None
        self._had_rest   = False
        self._had_lunch  = False
        self._had_dinner = False
        self._had_coffee = False
        self._date       = datetime(2000, 1, 1)   # arbitrary anchor date

    # ── Entry point ──────────────────────────────────────────────────────────

    def build(
        self,
        candidates: list[ActivityCandidate],
        day_plan:   dict | None = None,
    ) -> list[dict]:
        cfg      = self.cfg
        day_plan = day_plan or {}

        day_end         = _pt(self._date, cfg.day_end_time)
        _dep_dt         = (
            _pt(self._date, cfg.departure_time)
            if cfg.is_last_day and cfg.departure_time
            else None
        )
        # Guard: if the 2h30 buffer lands at or before day-start the flight is
        # early-morning (e.g. 02:05) — user has the whole day free for sightseeing.
        _dep_anchor_raw  = (_dep_dt - timedelta(minutes=150)) if _dep_dt else None
        _dep_after_start = (
            _dep_anchor_raw is not None
            and _dep_anchor_raw > _pt(self._date, cfg.day_start_time)
        )
        departure_anchor = _dep_anchor_raw if _dep_after_start else day_end

        # ── Weather flags ────────────────────────────────────────────────────
        cond     = cfg.weather_condition.lower()
        is_rainy = any(w in cond for w in ("rain", "storm", "thunder", "shower"))
        is_v_hot = "extreme_heat" in cond or "extreme heat" in cond

        # ── Blocked windows ──────────────────────────────────────────────────
        blocked: list[tuple[datetime, datetime]] = []
        for b in cfg.blocked_times:
            if not isinstance(b, dict):
                continue
            try:
                blocked.append((_pt(self._date, b["start"]),
                                 _pt(self._date, b["end"])))
            except (KeyError, ValueError):
                pass

        # ── LLM-suggested rest blocks → also soft-block so activities avoid them
        pending_rests: list[tuple[datetime, datetime, str]] = []
        rest_sources = (
            day_plan.get("recommended_rest_blocks") or
            cfg.suggested_rest_blocks or []
        )
        for rb in rest_sources:
            if not isinstance(rb, dict):
                continue
            try:
                rs     = _pt(self._date, rb["start"])
                re_    = _pt(self._date, rb["end"])
                reason = str(rb.get("reason", "Rest"))
                pending_rests.append((rs, re_, reason))
                blocked.append((rs, re_))
            except (KeyError, ValueError):
                pass

        # Extreme heat: force 14-16 rest block if not already present
        if is_v_hot:
            hs, he = _pt(self._date, "14:00"), _pt(self._date, "16:00")
            if not any(bs == hs for bs, _ in blocked):
                pending_rests.append((hs, he, "Afternoon rest — extreme heat"))
                blocked.append((hs, he))

        # ── Cursor & location init ───────────────────────────────────────────
        cursor   = _pt(self._date, cfg.day_start_time)
        location = GeoPoint(cfg.hotel_lat, cfg.hotel_lng, "Hotel")

        if cfg.is_first_day and cfg.arrival_time:
            cursor, location = self._arrival_sequence(cfg, blocked)
        else:
            breakfast_name = (
                day_plan.get("breakfast_place")
                or day_plan.get("coffee_place")
            )
            cursor = self._insert_breakfast(cfg, cursor, breakfast_name=breakfast_name)
            self._last_food_time = cursor

        # ── Split candidates into sightseeing vs meal venues ─────────────────
        pinned_lunch   = day_plan.get("lunch_restaurant")
        pinned_coffee  = day_plan.get("coffee_place")
        pinned_dinner  = day_plan.get("dinner_restaurant")
        pinned_names   = {n for n in [pinned_lunch, pinned_coffee, pinned_dinner] if n}

        meal_cands = [c for c in candidates if c.is_meal_venue or c.name in pinned_names]
        sights     = [c for c in candidates if not c.is_meal_venue and c.name not in pinned_names]
        meal_idx   = {c.name: c for c in meal_cands}

        # Rainy day: remove outdoor sightseeing candidates
        if is_rainy:
            sights = [c for c in sights if not c.is_outdoor]

        # ── Main sightseeing loop ────────────────────────────────────────────
        for act in sights:
            if cursor >= departure_anchor:
                break

            # — Flush rest blocks that are now due —
            cursor = self._flush_rests(cursor, pending_rests, departure_anchor)

            # — Pinned coffee (10:00–11:00 window) —
            if not self._had_coffee and pinned_coffee and 10 <= cursor.hour < 11:
                cursor, location = self._insert_pinned_meal(
                    pinned_coffee, "coffee", meal_idx,
                    cursor, departure_anchor, location, blocked)
                meal_cands = [c for c in meal_cands if c.name != pinned_coffee]

            # — Pinned lunch (12:00–14:30 window) —
            if not self._had_lunch and pinned_lunch and 12 <= cursor.hour < 15:
                cursor, location = self._insert_pinned_meal(
                    pinned_lunch, "lunch", meal_idx,
                    cursor, departure_anchor, location, blocked)
                meal_cands = [c for c in meal_cands if c.name != pinned_lunch]

            # — Generic hunger check —
            if self._is_hungry(cursor):
                cursor, location = self._inject_meal(
                    cursor, departure_anchor, location, meal_cands,
                    pinned_lunch, pinned_dinner, meal_idx)
                if cursor >= departure_anchor:
                    break

            # — Legacy rest: once per day around 14-16 if no LLM rest given —
            if not self._had_rest and not pending_rests and 14 <= cursor.hour <= 16:
                cursor = self._inject_rest(cursor, "Afternoon rest — recharge")
                if cursor >= departure_anchor:
                    break

            # — Compute transit to activity —
            act_pt              = GeoPoint(act.lat, act.lng, act.name)
            mode, t_min, t_cost = transit_plan(location, act_pt)

            # transit_start defaults to now; bumped to `free` in the blocked-window path
            # so the transit slot shows only actual travel time, not waiting.
            transit_start = cursor
            transit_end   = cursor + timedelta(minutes=t_min)
            act_start     = transit_end

            # Respect opening time (silent wait — not shown as "transit")
            opening = _pt(self._date, act.opening_time)
            if act_start < opening:
                act_start = opening

            act_end = act_start + timedelta(minutes=act.duration_minutes)

            # Fits within the day?
            if act_end > departure_anchor:
                short = act_start + timedelta(minutes=45)
                if short > departure_anchor:
                    continue      # skip this activity entirely
                act_end = short

            # Closing time
            closing = _pt(self._date, act.closing_time)
            if act_end > closing:
                continue

            # Blocked window? Shift the whole block past the obstruction.
            if _overlaps(cursor, act_end, blocked):
                free = _next_free(cursor, t_min + act.duration_minutes, blocked, departure_anchor)
                if free is None:
                    continue
                # Transit starts AFTER the blocked window clears, not at cursor.
                transit_start = free
                transit_end   = free + timedelta(minutes=t_min)
                act_start     = transit_end
                if act_start < opening:
                    act_start = opening
                act_end = act_start + timedelta(minutes=act.duration_minutes)
                if act_end > departure_anchor or act_end > closing:
                    continue

            # — Gap-lunch guard ─────────────────────────────────────────────────
            # When an opening-time wait or blocked window pushes act_start past 14:00
            # and we're still in the lunch window (cursor < 14:00), inject lunch now
            # rather than leaving a hours-long hungry gap in the schedule.
            if not self._had_lunch and cursor.hour < 14 and act_start.hour >= 14:
                gap_limit = min(act_start, _pt(self._date, "14:00"))
                if pinned_lunch and meal_idx.get(pinned_lunch):
                    cursor, location = self._insert_pinned_meal(
                        pinned_lunch, "lunch", meal_idx,
                        cursor, gap_limit, location, blocked)
                    meal_cands = [c for c in meal_cands if c.name != pinned_lunch]
                elif not (pinned_lunch and pinned_lunch in meal_idx):
                    cursor, location = self._inject_meal(
                        cursor, gap_limit, location,
                        meal_cands, None, pinned_dinner, meal_idx)
                if cursor >= departure_anchor:
                    break
                # Re-derive transit from updated position after the meal.
                mode, t_min, t_cost = transit_plan(location, act_pt)
                transit_start = cursor
                transit_end   = cursor + timedelta(minutes=t_min)
                act_start     = max(transit_end, opening)
                act_end       = act_start + timedelta(minutes=act.duration_minutes)
                if act_end > departure_anchor or act_end > closing:
                    continue

            # — Insert transit slot (actual travel time only) —
            if t_min > 0:
                self._push(TimeSlot(
                    start=transit_start, end=transit_end,
                    slot_type="transport",
                    name=f"Transit to {act.name}",
                    description=f"{mode} · {haversine_km(location, act_pt):.1f} km",
                    estimated_cost=t_cost,
                    transport_mode=mode,
                    distance_km=haversine_km(location, act_pt),
                ))

            # — Insert activity slot —
            self._push(TimeSlot(
                start=act_start, end=act_end,
                slot_type="activity",
                name=act.name,
                description=act.categories,
                estimated_cost=act.price,
            ))

            # Update food clock if this activity includes food
            if act.food_available:
                self._last_food_time = act_end
                if 11 <= act_start.hour <= 15:
                    self._had_lunch  = True
                elif act_start.hour >= 18:
                    self._had_dinner = True

            cursor   = act_end
            location = act_pt

        # ── End-of-day sequence ──────────────────────────────────────────────

        # Flush any remaining rest blocks
        cursor = self._flush_rests(cursor, pending_rests, departure_anchor)

        # Pinned dinner (18:00+ window)
        if not self._had_dinner and pinned_dinner and cursor.hour >= 18:
            cursor, location = self._insert_pinned_meal(
                pinned_dinner, "dinner", meal_idx,
                cursor, departure_anchor, location, blocked)
            meal_cands = [c for c in meal_cands if c.name != pinned_dinner]

        # "Free time" gap before 18:00 if ending early
        if not self._had_dinner and cursor < departure_anchor:
            dinner_earliest = _pt(self._date, "18:00")
            if cursor < dinner_earliest < departure_anchor:
                self._push(TimeSlot(
                    start=cursor, end=dinner_earliest,
                    slot_type="rest",
                    name="Free time — explore at your own pace",
                    description="Browse local shops, relax at a café, or head back to the hotel",
                    estimated_cost=0.0,
                ))
                cursor = dinner_earliest

            if cursor < departure_anchor and cursor.hour >= 17:
                cursor, location = self._inject_dinner(
                    cursor, departure_anchor, location, cfg, meal_cands)

        # Departure taxi (last day) — skip for overnight/early-morning flights;
        # those depart after the schedule window and the user arranges them separately.
        if _dep_after_start:
            t_min  = max(AIRPORT_HOTEL_MINUTES, 45)
            t_cost = TAXI_BASE_COST + 30 * TAXI_COST_PER_KM
            ts     = max(cursor, departure_anchor)
            self._push(TimeSlot(
                start=ts, end=ts + timedelta(minutes=t_min),
                slot_type="transport",
                name="City → Airport transfer",
                description=f"Taxi for your {cfg.departure_time} departure",
                estimated_cost=t_cost,
                transport_mode="taxi",
            ))
        return [s.to_dict() for s in self._slots]

    # ── Day 1 arrival ────────────────────────────────────────────────────────

    def _arrival_sequence(
        self, cfg: DayConfig, blocked: list
    ) -> tuple[datetime, GeoPoint]:
        arrival      = _pt(self._date, cfg.arrival_time)
        t_min        = max(AIRPORT_HOTEL_MINUTES, 45)
        t_cost       = TAXI_BASE_COST + 30 * TAXI_COST_PER_KM
        transfer_end = arrival + timedelta(minutes=t_min)

        self._push(TimeSlot(
            start=arrival, end=transfer_end,
            slot_type="transport",
            name="Airport → Hotel transfer",
            description="Taxi from airport to hotel",
            estimated_cost=t_cost,
            transport_mode="taxi",
        ))

        checkin_end = transfer_end + timedelta(minutes=HOTEL_CHECKIN_DURATION)
        self._push(TimeSlot(
            start=transfer_end, end=checkin_end,
            slot_type="checkin",
            name=f"Check-in · {cfg.hotel_name}",
            description="Settle in, freshen up",
            estimated_cost=0.0,
        ))

        hotel_loc = GeoPoint(cfg.hotel_lat, cfg.hotel_lng, cfg.hotel_name)

        if checkin_end.hour < 10:
            bk_end = self._insert_breakfast(cfg, checkin_end)
            return bk_end, hotel_loc
        if checkin_end.hour >= 12:
            self._last_food_time = checkin_end   # will trigger lunch soon
        return checkin_end, hotel_loc

    # ── Breakfast ────────────────────────────────────────────────────────────

    def _insert_breakfast(self, cfg: DayConfig, cursor: datetime, breakfast_name: str | None = None) -> datetime:
        cost = MEAL_COSTS["breakfast"]
        
        # מוודאים שהשם קיים, והוא לא המילה "null" או "none" שמגיעה מה-LLM
        if breakfast_name and str(breakfast_name).lower().strip() not in ("null", "none"):
            name = breakfast_name
        else:
            name = "Morning Coffee & Pastry"
            
        end  = cursor + timedelta(minutes=BREAKFAST_DURATION)
        self._push(TimeSlot(
            start=cursor, end=end,
            slot_type="meal", name=name,
            description="Start the day right",
            estimated_cost=cost,
        ))
        self._last_food_time = end
        return end
    # ── Pinned venue insertion ───────────────────────────────────────────────

    def _insert_pinned_meal(
        self,
        venue_name: str,
        meal_type:  str,           # "lunch" | "coffee" | "dinner"
        meal_idx:   dict[str, ActivityCandidate],
        cursor:     datetime,
        limit:      datetime,
        location:   GeoPoint,
        blocked:    list[tuple[datetime, datetime]],
    ) -> tuple[datetime, GeoPoint]:
        """Insert a named venue at the current cursor position."""
        venue = meal_idx.get(venue_name)
        if not venue:
            return cursor, location

        vpt             = GeoPoint(venue.lat, venue.lng, venue_name)
        mode, t_min, tc = transit_plan(location, vpt)

        # FIX: separate transit_end from start (opening-time adjustment)
        transit_end = cursor + timedelta(minutes=t_min)
        start       = transit_end
        opening     = _pt(self._date, venue.opening_time)
        if start < opening:
            start = opening

        dur = {"lunch": LUNCH_DURATION, "coffee": COFFEE_DURATION,
               "dinner": DINNER_DURATION}.get(meal_type, LUNCH_DURATION)
        end = start + timedelta(minutes=max(dur, venue.duration_minutes))

        if end > limit or start >= limit:
            return cursor, location
        if _overlaps(start, end, blocked):
            return cursor, location

        if t_min > 0:
            self._push(TimeSlot(
                start=cursor, end=transit_end,
                slot_type="transport",
                name=f"Transit to {venue_name}",
                estimated_cost=tc, transport_mode=mode,
                distance_km=haversine_km(location, vpt),
            ))
        self._push(TimeSlot(
            start=start, end=end,
            slot_type="meal",
            name=venue_name,
            description=venue.categories,
            estimated_cost=venue.price or MEAL_COSTS.get(meal_type, 15.0),
        ))
        self._last_food_time = end
        getattr(self, f"_set_{meal_type}", lambda: None)()
        meal_idx.pop(venue_name, None)
        return end, vpt

    def _set_lunch(self):  self._had_lunch  = True
    def _set_coffee(self): self._had_coffee = True
    def _set_dinner(self): self._had_dinner = True

    # ── Generic meal injection (hunger-triggered) ────────────────────────────

    def _inject_meal(
        self,
        cursor:       datetime,
        limit:        datetime,
        location:     GeoPoint,
        meal_cands:   list[ActivityCandidate],
        pinned_lunch: Optional[str],
        pinned_dinner: Optional[str],
        meal_idx:     dict[str, ActivityCandidate],
    ) -> tuple[datetime, GeoPoint]:
        """
        Insert the nearest available meal venue.
        Skips injection if a pinned venue for this meal-type is still pending —
        avoids double meals.
        """
        is_dinner = cursor.hour >= 17 or self._had_lunch

        # Don't inject generic meal if pinned one is coming soon
        if not is_dinner and pinned_lunch and pinned_lunch in meal_idx:
            return cursor, location
        if is_dinner and pinned_dinner and pinned_dinner in meal_idx:
            return cursor, location

        for i, meal in enumerate(meal_cands):
            oh = int(meal.opening_time.split(":")[0])
            if not is_dinner and oh >= 17:
                continue
            if is_dinner and meal.closing_time < "18:00":
                continue

            meal_cands.pop(i)
            mpt             = GeoPoint(meal.lat, meal.lng, meal.name)
            mode, t, tc     = transit_plan(location, mpt)
            transit_end     = cursor + timedelta(minutes=t)
            start           = transit_end
            opening         = _pt(self._date, meal.opening_time)
            if start < opening:
                start = opening
            end = start + timedelta(minutes=meal.duration_minutes)
            if end > limit:
                end = limit
            if t > 0:
                self._push(TimeSlot(
                    start=cursor, end=transit_end,
                    slot_type="transport",
                    name=f"Transit to {meal.name}",
                    estimated_cost=tc, transport_mode=mode,
                    distance_km=haversine_km(location, mpt),
                ))
            self._push(TimeSlot(
                start=start, end=end,
                slot_type="meal",
                name=meal.name, description=meal.categories,
                estimated_cost=meal.price,
            ))
            if is_dinner: self._had_dinner = True
            else:         self._had_lunch  = True
            self._last_food_time = end
            return end, mpt

        # Placeholder when no venue available
        if cursor.hour < 15 and not self._had_lunch:
            nm, cost, dur, tag = "Lunch",              MEAL_COSTS["lunch"],  LUNCH_DURATION,  "L"
        elif cursor.hour >= 17 and not self._had_dinner:
            nm, cost, dur, tag = "Dinner",             MEAL_COSTS["dinner"], DINNER_DURATION, "D"
        else:
            nm, cost, dur, tag = "Snack / coffee break", 8.0,                20,              None
        end = min(cursor + timedelta(minutes=dur), limit)
        self._push(TimeSlot(
            start=cursor, end=end,
            slot_type="meal", name=nm,
            description="Local restaurant", estimated_cost=cost,
        ))
        if tag == "L": self._had_lunch  = True
        if tag == "D": self._had_dinner = True
        self._last_food_time = end
        return end, location

    # ── End-of-day dinner ────────────────────────────────────────────────────

    def _inject_dinner(
        self,
        cursor:    datetime,
        limit:     datetime,
        location:  GeoPoint,
        cfg:       DayConfig,
        meal_cands: list[ActivityCandidate],
    ) -> tuple[datetime, GeoPoint]:
        for i, meal in enumerate(meal_cands):
            if meal.closing_time < "18:00":
                continue
            meal_cands.pop(i)
            mpt             = GeoPoint(meal.lat, meal.lng, meal.name)
            mode, t, tc     = transit_plan(location, mpt)
            transit_end     = cursor + timedelta(minutes=t)
            start           = transit_end
            opening         = _pt(self._date, meal.opening_time)
            if start < opening:
                start = opening
            end = min(start + timedelta(minutes=meal.duration_minutes), limit)
            if t > 0:
                self._push(TimeSlot(
                    start=cursor, end=transit_end,
                    slot_type="transport",
                    name=f"Transit to {meal.name}",
                    estimated_cost=tc, transport_mode=mode,
                    distance_km=haversine_km(location, mpt),
                ))
            self._push(TimeSlot(
                start=start, end=end,
                slot_type="meal",
                name=meal.name, description=meal.categories,
                estimated_cost=meal.price,
            ))
            self._had_dinner     = True
            self._last_food_time = end
            return end, mpt

        # Near-hotel fallback
        hpt         = GeoPoint(cfg.hotel_lat, cfg.hotel_lng, cfg.hotel_name)
        mode, t, tc = transit_plan(location, hpt)
        transit_end = cursor + timedelta(minutes=t)
        ds          = transit_end
        if ds >= limit:
            return cursor, location
        if t > 0:
            self._push(TimeSlot(
                start=cursor, end=transit_end,
                slot_type="transport",
                name="Return to hotel area",
                description=f"{mode} back",
                estimated_cost=tc, transport_mode=mode,
            ))
        de = min(ds + timedelta(minutes=DINNER_DURATION), limit)
        self._push(TimeSlot(
            start=ds, end=de,
            slot_type="meal",
            name="Dinner",
            description="Local restaurant near hotel",
            estimated_cost=MEAL_COSTS["dinner"],
        ))
        self._had_dinner     = True
        self._last_food_time = de
        return de, hpt

    # ── Rest helpers ─────────────────────────────────────────────────────────

    def _inject_rest(self, cursor: datetime, reason: str = "Rest") -> datetime:
        end = cursor + timedelta(minutes=REST_DURATION)
        self._push(TimeSlot(
            start=cursor, end=end,
            slot_type="rest", name="Rest break",
            description=reason, estimated_cost=0.0,
        ))
        self._had_rest = True
        return end

    def _flush_rests(
        self,
        cursor:  datetime,
        pending: list[tuple[datetime, datetime, str]],
        limit:   datetime,
    ) -> datetime:
        """Insert rest blocks whose window is now due (cursor ≥ start − 10 min)."""
        to_remove = []
        for i, (rs, re_, reason) in enumerate(pending):
            if cursor >= rs - timedelta(minutes=10):
                actual_s = max(cursor, rs)
                actual_e = min(re_, limit)
                if actual_s < actual_e and not self._had_rest:
                    self._push(TimeSlot(
                        start=actual_s, end=actual_e,
                        slot_type="rest", name="Rest break",
                        description=reason, estimated_cost=0.0,
                    ))
                    cursor         = actual_e
                    self._had_rest = True
                to_remove.append(i)
        for i in reversed(to_remove):
            pending.pop(i)
        return cursor

    # ── Misc ─────────────────────────────────────────────────────────────────

    def _is_hungry(self, cursor: datetime) -> bool:
        if self._last_food_time is None:
            return cursor.hour >= 12
        elapsed = (cursor - self._last_food_time).total_seconds() / 3600
        return elapsed >= MEAL_INTERVAL_HOURS

    def _push(self, slot: TimeSlot) -> None:
        self._slots.append(slot)