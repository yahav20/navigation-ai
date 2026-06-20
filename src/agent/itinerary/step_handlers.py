"""
step_handlers.py — one function per itinerary step type.

Each handler receives the arguments it needs and returns a wrapped result dict
(see _wrap_result). The executor calls these via a dispatch map — it never
contains step logic itself.

Handlers:
  handle_fetch_weather     — simple tool call with cache
  handle_fetch_activities  — Google Maps fetch + dietary filter + ActivitySelector
  handle_build_day         — pure-Python ScheduleEngine, mode-aware DayConfig
  handle_verify_budget     — mode-aware cost rollup + budget gate

Utility functions used by all handlers live at the bottom of this file.
"""
from __future__ import annotations

import datetime
import json
from typing import Optional

from langchain_core.language_models import BaseChatModel

from agent.itinerary.itinerary_tools import (
    calculate_trip_cost,
    get_average_location_cost,
    get_min_location_cost,
    get_weather,
)
from agent.itinerary.schedule_engine import DayConfig, DayScheduleBuilder, haversine_km, GeoPoint
from agent.itinerary.activity_selector import select_activities_per_day, resolve_candidates
from agent.itinerary.schemas import PlanStep
from agent.shared.travelers import compute_default_rooms
from tools.activities import fetch_attractions, fetch_restaurants
from tools.dependencies import data_provider
from tools.weather import get_average_weather
from providers.google_maps import search_places

# ---------------------------------------------------------------------------
# Tool registry — used by handle_fetch_weather
# ---------------------------------------------------------------------------

def _args_fetch_weather(destination: str, **_) -> dict:
    return {"city": destination}

TOOL_REGISTRY: dict[str, tuple] = {
    "fetch_weather": (get_weather, _args_fetch_weather),
}

BUILD_PREREQUISITES = {"fetch_activities"}


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def handle_fetch_weather(step: PlanStep, results: dict, history: list, ctx: dict) -> dict:
    """Generic handler for registry-based fetch steps (currently fetch_weather)."""
    tool_fn, arg_builder = TOOL_REGISTRY[step.step_type]
    args = _canonical_args(arg_builder(**ctx))

    cached = _find_cached_result(results, history, step.step_type, args)
    if cached is not None:
        return _wrap_result(
            status="success", data=cached, error=None, replan_hint="",
            trace=_minimal_trace(step.step_type, args, "Cache hit.", ""),
        )

    try:
        raw = tool_fn.invoke(args)
        observation = json.dumps(raw, ensure_ascii=False)[:600]
    except Exception as exc:
        raw = None
        observation = f"Tool exception: {exc}"

    if _is_empty(raw):
        return _wrap_result(
            status="failed",
            data=None,
            error=observation,
            replan_hint=(
                f"`{step.step_type}` returned no data for destination "
                f"'{ctx.get('destination')}'. Check city name spelling."
            ),
            trace=_minimal_trace(step.step_type, args, observation, ""),
        )

    return _wrap_result(
        status="success", data=raw, error=None, replan_hint="",
        trace=_minimal_trace(step.step_type, args, observation, ""),
    )


def _extract_activity_costs(results: dict, trip_days: int) -> tuple[float, float, int]:
    """Return (activities_cost, meals_cost, days_built) from built day schedules.
    Used by the Critic to compare overage against what cheaper activities could save.
    """
    activities_cost = 0.0
    meals_cost      = 0.0
    days_built      = 0
    for key, wrapped in results.items():
        if not key.startswith("build_day_schedule"):
            continue
        day_data = wrapped.get("data") if isinstance(wrapped, dict) else None
        if not isinstance(day_data, dict) or not day_data.get("slots"):
            continue
        if day_data.get("day", 0) > trip_days:
            continue
        days_built += 1
        for slot in day_data["slots"]:
            cost = float(slot.get("estimated_cost", 0))
            if slot.get("slot_type") == "meal":
                meals_cost += cost
            else:
                activities_cost += cost
    return activities_cost, meals_cost, days_built


def handle_switch_travel_options(step: PlanStep, results: dict, history: list, ctx: dict, state: dict) -> dict:
    """Select the cheapest available flight + hotel alternative from state.

    Updates itinerary_selected_outbound_flight and itinerary_selected_hotel via
    state_updates so that subsequent build_day_schedule steps use the new values.
    """
    current_outbound = (state or {}).get("itinerary_selected_outbound_flight") or {}
    current_hotel    = (state or {}).get("itinerary_selected_hotel") or {}
    travel_plan      = (state or {}).get("travel_plan") or {}

    # Cheapest alternative outbound flight (different flight number)
    all_flights = [
        f for f in ((state or {}).get("flight_options") or [])
        if isinstance(f, dict) and not f.get("message")
    ]
    current_fn  = current_outbound.get("flight_number", "")
    alt_flights = sorted(
        [f for f in all_flights if f.get("flight_number") != current_fn],
        key=lambda f: float(f.get("price", 9999)),
    )
    new_flight = alt_flights[0] if alt_flights else None

    # Cheapest alternative hotel (different name)
    all_hotels = [h for h in travel_plan.get("hotels", []) if isinstance(h, dict)]
    current_hn = current_hotel.get("name", "")
    alt_hotels = sorted(
        [h for h in all_hotels if h.get("name", "") != current_hn],
        key=lambda h: float(h.get("price_per_night", 9999)),
    )
    new_hotel = alt_hotels[0] if alt_hotels else None

    switched_flight = new_flight is not None
    switched_hotel  = new_hotel  is not None

    observation = (
        f"Switched flight: {switched_flight} "
        f"({'$' + str(new_flight.get('price', '?')) if switched_flight else 'no change'}), "
        f"hotel: {switched_hotel} "
        f"({'$' + str(new_hotel.get('price_per_night', '?')) + '/night' if switched_hotel else 'no change'})"
    )

    result = _wrap_result(
        status="success",
        data={
            "switched_flight": switched_flight,
            "switched_hotel":  switched_hotel,
            "new_flight":      new_flight,
            "new_hotel":       new_hotel,
        },
        error=None,
        replan_hint="",
        trace=_minimal_trace("switch_travel_options", {}, observation, ""),
    )

    # Propagate new selections to state so build_day_schedule uses them
    state_updates: dict = {}
    if switched_flight:
        state_updates["itinerary_selected_outbound_flight"] = new_flight
    if switched_hotel:
        state_updates["itinerary_selected_hotel"] = new_hotel
    if state_updates:
        result["state_updates"] = state_updates

    return result


def handle_fetch_special_events(
    step: PlanStep,
    results: dict,
    history: list,
    ctx: dict,
    state: dict,
    llm,
) -> dict:
    """Fetch and extract special events for the destination via Tavily + LLM."""
    destination = ctx["destination"]
    trip_start  = state.get("trip_start", "")

    cache_args = _canonical_args({"city": destination, "trip_start": trip_start})
    cached = _find_cached_result(results, history, "fetch_special_events", cache_args)
    if cached is not None:
        return _wrap_result(
            status="success", data=cached, error=None, replan_hint="",
            trace=_minimal_trace("fetch_special_events", cache_args, "Cache hit.", ""),
        )

    try:
        from agent.itinerary.special_events_agent import SpecialEventsSubAgent  # noqa: PLC0415
        events = SpecialEventsSubAgent(llm)(state)
    except Exception:  # noqa: BLE001
        events = []

    data   = {"events": events}
    result = _wrap_result(
        status="success", data=data, error=None, replan_hint="",
        trace=_minimal_trace(
            "fetch_special_events", cache_args,
            f"Found {len(events)} special event(s).", "",
        ),
    )
    if events:
        result["state_updates"] = {"special_events_data": events}
    return result


def handle_fetch_avg_prices(step: PlanStep, results: dict, history: list, ctx: dict) -> dict:
    """Fetch and cache average flight + hotel prices for standalone budget estimation."""
    destination = ctx["destination"]
    origin      = ctx["origin"]
    trip_days   = ctx["trip_days"]

    args   = _canonical_args({"city": destination, "origin": origin})
    cached = _find_cached_result(results, history, "fetch_avg_prices", args)
    if cached is not None:
        return _wrap_result(
            status="success", data=cached, error=None, replan_hint="",
            trace=_minimal_trace("fetch_avg_prices", args, "Cache hit.", ""),
        )

    try:
        raw = get_average_location_cost.invoke({
            "destination": destination,
            "origin":      origin,
            "trip_days":   trip_days,
        })
    except Exception as exc:
        raw = None

    if _is_empty(raw):
        raw = {
            "avg_flight_price":        400.0,
            "avg_return_flight_price": 400.0,
            "avg_hotel_per_night":     120.0,
            "note": "estimated fallback",
        }

    return _wrap_result(
        status="success", data=raw, error=None, replan_hint="",
        trace=_minimal_trace("fetch_avg_prices", args,
                             f"avg_flight={raw.get('avg_flight_price')}, "
                             f"avg_hotel={raw.get('avg_hotel_per_night')}", ""),
    )


def handle_fetch_min_prices(step: PlanStep, results: dict, history: list, ctx: dict) -> dict:
    """Fetch and cache minimum available flight + hotel prices for standalone budget check."""
    destination = ctx["destination"]
    origin      = ctx["origin"]
    trip_days   = ctx["trip_days"]

    args   = _canonical_args({"city": destination, "origin": origin})
    cached = _find_cached_result(results, history, "fetch_min_prices", args)
    if cached is not None:
        return _wrap_result(
            status="success", data=cached, error=None, replan_hint="",
            trace=_minimal_trace("fetch_min_prices", args, "Cache hit.", ""),
        )

    try:
        raw = get_min_location_cost.invoke({
            "destination": destination,
            "origin":      origin,
            "trip_days":   trip_days,
        })
    except Exception as exc:
        raw = None

    if _is_empty(raw):
        raw = {
            "min_flight_price":        400.0,
            "min_return_flight_price": 400.0,
            "min_hotel_per_night":     120.0,
            "note": "estimated fallback",
        }

    return _wrap_result(
        status="success", data=raw, error=None, replan_hint="",
        trace=_minimal_trace("fetch_min_prices", args,
                             f"min_flight={raw.get('min_flight_price')}, "
                             f"min_hotel={raw.get('min_hotel_per_night')}", ""),
    )


def handle_fetch_activities(
    step: PlanStep,
    results: dict,
    history: list,
    ctx: dict,
    trip_days: int,
    llm: BaseChatModel,
) -> dict:
    """Fetch attractions + restaurants from Google Maps, then run ActivitySelector."""
    destination = ctx["destination"]
    prefs       = ctx["prefs"] or {}
    dietary     = str(prefs.get("dietary_restrictions") or "").lower().strip()
    pref_loc    = str(prefs.get("preferred_location") or "").strip()

    cache_args = _canonical_args({"city": destination})
    cached = _find_cached_result(results, history, "fetch_activities", cache_args)
    if cached is not None:
        return _wrap_result(
            status="success", data=cached, error=None, replan_hint="",
            trace=_minimal_trace("fetch_activities", cache_args,
                                 "Cache hit.", "Using cached data."),
        )

    # ── 1. Fetch attractions ──────────────────────────────────────────────
    attraction_query = (
        f"attractions near {pref_loc} in {destination}"
        if pref_loc else
        f"tourist attractions in {destination}"
    )
    raw_attractions: list[dict] = []
    try:
        raw_attractions = fetch_attractions.invoke({
            "city":  destination,
            "query": attraction_query,
        }) or []
    except Exception as exc:
        pass

    if _is_empty(raw_attractions):
        # Google Maps returned nothing — fall back to local SQLite activities so
        # the itinerary can still be built without a working Maps API key.
        raw_attractions = data_provider.fetch_activities(destination, approach="local")

    if _is_empty(raw_attractions):
        return _wrap_result(
            status="failed",
            data=None,
            error=f"fetch_attractions returned nothing for '{destination}'.",
            replan_hint=(
                f"No activities found for '{destination}' in Google Maps or local DB. "
                "Check the city name and GOOGLE_MAPS_API_KEY."
            ),
            trace=_minimal_trace("fetch_activities", cache_args,
                                 "Empty attractions list (Maps + local DB).",
                                 "Cannot build schedule without activities."),
        )

    # ── 2. Fetch restaurants ──────────────────────────────────────────────
    rest_invoke_args: dict = {"city": destination}
    if dietary in ("kosher", "vegetarian", "vegan"):
        rest_invoke_args["cuisine"] = dietary

    raw_restaurants: list[dict] = []
    try:
        raw_restaurants = fetch_restaurants.invoke(rest_invoke_args) or []
    except Exception:
        pass

    if raw_restaurants and dietary:
        filtered = _filter_restaurants_by_dietary(raw_restaurants, dietary)
        if filtered:
            raw_restaurants = filtered

    # ── 3. Activity selection for all days ────────────────────────────────
    weather_cond   = _resolve_weather(results, destination)
    planner_prefs  = _extract_planner_prefs(prefs)
    blocked_times  = list(planner_prefs.get("blocked_times") or [])

    try:
        selection = select_activities_per_day(
            llm=llm,
            activities=raw_attractions,
            trip_days=trip_days,
            prefs=planner_prefs,
            destination=destination,
            restaurants=raw_restaurants,
            weather=weather_cond or None,
            blocked_times=blocked_times or None,
        )
    except Exception:
        sorted_acts = sorted(raw_attractions, key=lambda a: -a.get("rating", 0))
        selection = {
            f"day_{d}": {
                "theme": f"Day {d} — Explore {destination}", "area": "",
                "activities": [a["name"] for a in sorted_acts[(d - 1) * 5: d * 5]],
                "lunch_restaurant": None, "breakfast_place": None, "dinner_restaurant": None,
                "recommended_rest_blocks": [],
            }
            for d in range(1, trip_days + 1)
        }

    data = {
        "activities":  raw_attractions,
        "restaurants": raw_restaurants,
        "selection":   selection,
    }
    observation = (
        f"Fetched {len(raw_attractions)} attractions + {len(raw_restaurants)} restaurants. "
        f"Weather: {weather_cond or 'unknown'}. Selection built for {len(selection)} days."
    )
    return _wrap_result(
        status="success", data=data, error=None, replan_hint="",
        trace=_minimal_trace("fetch_activities", cache_args,
                             observation, "Activities fetched and assigned to days."),
    )


def handle_build_day(
    step: PlanStep,
    results: dict,
    destination: str,
    trip_days: int,
    current_plan_keys: set[str],
    mode: str,
    state: dict,
) -> dict:
    """Build a deterministic hour-by-hour schedule for one day via ScheduleEngine."""
    day_num = step.day or 1
    action  = f"build_day_schedule (Day {day_num})"

    missing = _missing_prerequisites(results)
    if missing:
        return _wrap_result(
            status="failed",
            data=None,
            error=f"Day {day_num}: prerequisites missing: {missing}",
            replan_hint=(
                f"Day {day_num} cannot be built: {missing} have no successful result. "
                "Re-include them in the plan before build_day_schedule."
            ),
            trace=_minimal_trace(action, {"day": day_num},
                                 f"Prerequisites missing: {missing}", ""),
        )

    acts_data = _unwrap_data(results, "fetch_activities")
    all_activities:  list[dict] = []
    all_restaurants: list[dict] = []
    selection:       dict       = {}
    if isinstance(acts_data, dict):
        all_activities  = acts_data.get("activities",  [])
        all_restaurants = acts_data.get("restaurants", [])
        selection       = acts_data.get("selection",   {})

    # Retrieve pre-selected day plan from ActivitySelector output
    day_plan = selection.get(f"day_{day_num}", {})
    if not isinstance(day_plan, dict):
        day_plan = {"activities": list(day_plan) if isinstance(day_plan, list) else []}

    # Deduplicate: exclude activities already scheduled on previous days.
    # Also exclude meal venues already used — prevents the same restaurant from
    # appearing in the backup pool and being picked again via generic injection.
    used              = _used_activities(results, current_plan_keys)
    used_meals        = _used_meal_venues(results, current_plan_keys)
    available_acts    = [a for a in all_activities  if a.get("name") not in used]
    available_rests   = [r for r in all_restaurants
                         if r.get("name") not in used and r.get("name") not in used_meals]
    day_act_names     = [n for n in day_plan.get("activities", []) if n not in used]
    day_plan_filtered = {**day_plan, "activities": day_act_names}

    # Also nullify pinned meal slots that point to already-used venues.
    for meal_key in ("breakfast_place", "coffee_place", "lunch_restaurant", "dinner_restaurant"):
        if day_plan_filtered.get(meal_key) in used_meals:
            day_plan_filtered = {**day_plan_filtered, meal_key: None}

    candidates = resolve_candidates(available_acts, day_plan_filtered, available_rests)

    # Fallback: backup meal pool means candidates is never empty, so check for
    # sightseeing candidates specifically — not just any candidate.
    has_sights = any(not c.is_meal_venue for c in candidates)
    if not candidates or not has_sights:
        # ActivitySelector returned empty/unrecognised names — fall back to top-rated available acts
        sorted_avail = sorted(available_acts, key=lambda a: -(a.get("rating") or 0))
        fallback_plan = {**day_plan, "activities": [a["name"] for a in sorted_avail[:5]]}
        candidates = resolve_candidates(available_acts, fallback_plan, available_rests)
        if candidates:
            day_plan_filtered = fallback_plan

    if not candidates:
        return _wrap_result(
            status="failed",
            data=None,
            error=f"Day {day_num}: no activity candidates after deduplication.",
            replan_hint=(
                f"Day {day_num} has no unscheduled candidates. "
                "Re-run fetch_activities so ActivitySelector can reassign."
            ),
            trace=_minimal_trace(action, {"day": day_num},
                                 "resolve_candidates returned [].", ""),
        )

    # Build DayConfig with weather + blocked times
    weather_cond  = _resolve_weather(results, destination)
    prefs         = state.get("user_preferences") or {}
    blocked_times = list(prefs.get("blocked_times") or [])
    day_blocked   = [b for b in blocked_times
                     if isinstance(b, dict) and b.get("day") == day_num]
    rest_blocks   = day_plan.get("recommended_rest_blocks", [])

    cfg = (
        _build_config_from_travel_plan(state, day_num, trip_days, candidates,
                                       weather_cond, day_blocked, rest_blocks)
        if mode == "with_travel_data"
        else _build_config_standalone(state, day_num, trip_days, candidates,
                                      weather_cond, day_blocked, rest_blocks)
    )

    # Retrieve and geocode special events for this calendar day
    day_events = _enrich_event_coords(_get_day_events(state, day_num, results), destination)

    try:
        slots = DayScheduleBuilder(cfg).build(
            candidates,
            day_plan=day_plan_filtered,
            special_events=day_events or None,
        )
    except Exception as exc:
        return _wrap_result(
            status="failed",
            data=None,
            error=f"ScheduleEngine exception on Day {day_num}: {exc}",
            replan_hint=(
                f"ScheduleEngine crashed on Day {day_num}: {exc}. "
                "Check activity opening/closing times and coordinates."
            ),
            trace=_minimal_trace(action, {"day": day_num}, str(exc), ""),
        )

    if not slots:
        return _wrap_result(
            status="failed",
            data=None,
            error=f"Day {day_num}: ScheduleEngine produced no slots.",
            replan_hint=(
                f"Day {day_num}: all candidates may be outside operating hours. "
                "Try different activities or broaden the candidates pool."
            ),
            trace=_minimal_trace(action, {"day": day_num, "candidates": len(candidates)},
                                 "build() returned [].", ""),
        )

    day_cost = round(sum(float(s.get("estimated_cost", 0)) for s in slots), 2)
    day_themes = {
        1:         f"Arrival & first impressions of {destination}",
        trip_days: f"Final day & farewell to {destination}",
    }
    theme = day_plan.get("theme") or day_themes.get(day_num) or f"Day {day_num} — Explore {destination}"
    area  = day_plan.get("area", "")

    return _wrap_result(
        status="success",
        data={
            "day":      day_num,
            "theme":    theme,
            "area":     area,
            "slots":    slots,
            "day_cost": day_cost,
            "hotel":    cfg.hotel_name,
            "hotel_price_per_night": 0.0 if mode == "standalone" else _hotel_price(state),
        },
        error=None,
        replan_hint="",
        trace=_minimal_trace(action, {"day": day_num, "candidates": len(candidates)},
                             f"Built {len(slots)} slots. Day cost: ${day_cost}."
                             + (f" {len(day_events)} special event(s) injected." if day_events else ""),
                             f"Day {day_num} complete."),
    )


def handle_update_day_schedule(
    step: PlanStep,
    results: dict,
    destination: str,
    trip_days: int,
    current_plan_keys: set[str],
    mode: str,
    state: dict,
    llm=None,
) -> dict:
    """Rebuild one day with exclusions/replacements from itinerary_update_request.

    Quality path (when llm + replacement present):
      Phase 2 — targeted single-day LLM re-plan for full geographic/energy-curve quality.

    Deterministic fallback (always used for exclusions, or when Phase 2 fails):
      Phase 1A — backfill to original activity count by geographic proximity
      Phase 1B — geo-sort activities by nearest-neighbor chain from hotel
      Phase 1C — re-pick meal venues nearest to updated activity centroid
      Phase 1D — skip outdoor backfill candidates when weather is bad
    """
    day_num    = step.day or 1
    action     = f"update_day_schedule (Day {day_num})"
    update_req = (state or {}).get("itinerary_update_request") or {}
    target      = update_req.get("target_name")
    replacement = update_req.get("replacement_name")

    missing = _missing_prerequisites(results)
    if missing:
        return _wrap_result(
            status="failed", data=None,
            error=f"Day {day_num}: prerequisites missing: {missing}",
            replan_hint=f"Re-include {missing} before update_day_schedule.",
            trace=_minimal_trace(action, {"day": day_num}, f"Missing: {missing}", ""),
        )

    acts_data       = _unwrap_data(results, "fetch_activities")
    all_activities:  list[dict] = []
    all_restaurants: list[dict] = []
    selection:       dict       = {}
    if isinstance(acts_data, dict):
        all_activities  = acts_data.get("activities",  [])
        all_restaurants = acts_data.get("restaurants", [])
        selection       = acts_data.get("selection",   {})

    # Exclude the target from candidate pools
    filtered_acts  = [a for a in all_activities  if a.get("name") != target]
    filtered_rests = [r for r in all_restaurants if r.get("name") != target]

    # Deduplicate against other days
    used = _used_activities(results, current_plan_keys)
    used.discard(target or "")
    available_acts  = [a for a in filtered_acts  if a.get("name") not in used]
    available_rests = [r for r in filtered_rests if r.get("name") not in used]

    # Start from the original LLM day plan, stripped of excluded/used activities
    day_plan      = selection.get(f"day_{day_num}", {})
    original_count = len(day_plan.get("activities", []))
    day_act_names = [n for n in day_plan.get("activities", [])
                     if n not in used and n != target]

    # ── Replacement resolution + weather guard (Phase 1D) ─────────────────────
    weather_cond = _resolve_weather(results, destination)
    pin: Optional[dict] = None

    if replacement:
        pin = next(
            (a for a in available_acts if a.get("name", "").lower() == replacement.lower()),
            next((a for a in available_acts if replacement.lower() in a.get("name", "").lower()), None),
        )
        if pin is None:
            fetched = _search_activity(replacement, destination)
            # Phase 1D: if searched result is outdoor and weather is bad, warn but keep it;
            # the ScheduleEngine will handle skipping + backfill covers the gap.
            available_acts = [*available_acts, fetched]
            pin = fetched
        if pin["name"] not in day_act_names:
            day_act_names.insert(0, pin["name"])

    # ── Phase 2: LLM single-day re-plan ───────────────────────────────────────
    # Only when a replacement is specified and an LLM is available.
    # Gives full geographic clustering, meal proximity, and energy-curve ordering.
    day_plan_filtered: Optional[dict] = None
    pin_name = pin["name"] if pin else None

    if replacement and pin_name and llm:
        try:
            from agent.itinerary.activity_selector import select_single_day_update
            prefs = (state or {}).get("user_preferences") or {}
            fresh = select_single_day_update(
                llm=llm,
                available_acts=available_acts,
                available_rests=available_rests,
                pinned_activity=pin_name,
                excluded_names={target} if target else set(),
                day_num=day_num,
                weather=weather_cond,
                prefs=prefs,
                destination=destination,
            )
            # Accept the LLM plan only if it actually includes the requested activity
            if fresh.get("activities") and pin_name in fresh["activities"]:
                day_act_names     = fresh["activities"]
                day_plan_filtered = {**day_plan, **fresh}
        except Exception:
            pass  # fall through to deterministic path

    # ── Phase 1: Deterministic quality (when no LLM or Phase 2 failed) ────────
    if day_plan_filtered is None:
        act_idx      = {a["name"]: a for a in available_acts}
        hotel_lat, hotel_lng = _hotel_latlng(state, mode, available_acts)

        # 1A: Backfill to original activity count
        target_count  = max(original_count, 3)
        day_act_names = _backfill_activities(
            day_act_names, available_acts, target_count, act_idx, weather_cond
        )

        # 1B: Geo-sort for efficient routing
        day_act_names = _geo_sort_activities(day_act_names, act_idx, hotel_lat, hotel_lng)

        # 1C: Re-pick meal venues to match updated activity cluster
        updated_plan      = _repick_meals(day_plan, day_act_names, act_idx, available_rests)
        day_plan_filtered = {**updated_plan, "activities": day_act_names}

    # ── Resolve candidates and build schedule ──────────────────────────────────
    candidates = resolve_candidates(available_acts, day_plan_filtered, available_rests)

    if not candidates:
        sorted_avail  = sorted(available_acts, key=lambda a: -(a.get("rating") or 0))
        fallback_plan = {**day_plan_filtered, "activities": [a["name"] for a in sorted_avail[:5]]}
        candidates    = resolve_candidates(available_acts, fallback_plan, available_rests)
        if candidates:
            day_plan_filtered = fallback_plan

    if not candidates:
        return _wrap_result(
            status="failed", data=None,
            error=f"Day {day_num}: no candidates after exclusion.",
            replan_hint=f"All activities for Day {day_num} used or excluded.",
            trace=_minimal_trace(action, {"day": day_num}, "No candidates.", ""),
        )

    prefs         = (state or {}).get("user_preferences") or {}
    blocked_times = list(prefs.get("blocked_times") or [])
    day_blocked   = [b for b in blocked_times if isinstance(b, dict) and b.get("day") == day_num]
    rest_blocks   = day_plan_filtered.get("recommended_rest_blocks", [])

    cfg = (
        _build_config_from_travel_plan(state, day_num, trip_days, candidates,
                                       weather_cond, day_blocked, rest_blocks)
        if mode == "with_travel_data"
        else _build_config_standalone(state, day_num, trip_days, candidates,
                                      weather_cond, day_blocked, rest_blocks)
    )

    try:
        slots = DayScheduleBuilder(cfg).build(candidates, day_plan=day_plan_filtered)
    except Exception as exc:
        return _wrap_result(
            status="failed", data=None,
            error=f"ScheduleEngine error on Day {day_num}: {exc}",
            replan_hint=str(exc),
            trace=_minimal_trace(action, {"day": day_num}, str(exc), ""),
        )

    if not slots:
        return _wrap_result(
            status="failed", data=None,
            error=f"Day {day_num}: no slots produced.",
            replan_hint="Try a different replacement activity.",
            trace=_minimal_trace(action, {"day": day_num}, "build() returned [].", ""),
        )

    day_cost = round(sum(float(s.get("estimated_cost", 0)) for s in slots), 2)
    theme    = day_plan_filtered.get("theme") or day_plan.get("theme") or f"Day {day_num} — Explore {destination}"

    # Persist coordinates of all candidates (including newly-searched activities not in
    # fetch_activities) so formatter.py can resolve map pins for replacement activities.
    coord_index: dict[str, list[float]] = {}
    for a in available_acts:
        name = a.get("name", "")
        lat  = float(a.get("lat") or a.get("latitude") or 0)
        lng  = float(a.get("lng") or a.get("longitude") or 0)
        if name and (lat or lng):
            coord_index[name] = [lat, lng]

    return _wrap_result(
        status="success",
        data={
            "day":         day_num,
            "theme":       theme,
            "area":        day_plan_filtered.get("area") or day_plan.get("area", ""),
            "slots":       slots,
            "day_cost":    day_cost,
            "hotel":       cfg.hotel_name,
            "hotel_price_per_night": 0.0 if mode == "standalone" else _hotel_price(state),
            "coord_index": coord_index,
        },
        error=None, replan_hint="",
        trace=_minimal_trace(action, {"day": day_num, "excluded": target, "added": replacement},
                             f"Updated Day {day_num}. Cost: ${day_cost}.", ""),
    )


def handle_apply_global_preference(
    step: PlanStep,
    results: dict,
    state: dict,
) -> dict:
    """Apply a preference change to ALL built days in-place (no rebuild needed).

    Returns modified day results in data["modified_results"] so the executor
    can merge them into the results dict before committing.
    """
    update_req           = (state or {}).get("itinerary_update_request") or {}
    excluded_slot_types  = update_req.get("excluded_slot_types", [])

    modified: dict = {}
    days_modified  = 0

    for key, wrapped in results.items():
        if not key.startswith("build_day_schedule") or not isinstance(wrapped, dict):
            continue
        day_data = wrapped.get("data", {})
        if not isinstance(day_data, dict):
            continue

        new_slots  = _filter_slots(day_data.get("slots", []), excluded_slot_types)
        new_cost   = round(sum(float(s.get("estimated_cost", 0)) for s in new_slots), 2)
        modified[key] = {**wrapped, "data": {**day_data, "slots": new_slots, "day_cost": new_cost}}
        days_modified += 1

    return _wrap_result(
        status="success",
        data={"days_modified": days_modified, "modified_results": modified},
        error=None, replan_hint="",
        trace=_minimal_trace(
            "apply_global_preference", {},
            f"Removed {excluded_slot_types} from {days_modified} days.", "",
        ),
    )


def _filter_slots(slots: list, excluded_slot_types: list) -> list:
    """Remove slots matching excluded_slot_types; 'breakfast' removes morning meals."""
    remove_breakfast = "breakfast" in excluded_slot_types
    out = []
    for slot in slots:
        st = slot.get("slot_type", "")
        if st in excluded_slot_types:
            continue
        if remove_breakfast and st == "meal":
            try:
                if int(slot.get("time", "12:00").split(":")[0]) < 10:
                    continue
            except (ValueError, AttributeError):
                pass
        out.append(slot)
    return out


def handle_verify_budget(
    results: dict,
    budget: float,
    trip_days: int,
    destination: str,
    origin: str,
    mode: str,
    state: dict,
) -> dict:
    """Roll up all day costs, read flight/hotel prices, call calculate_trip_cost."""
    missing = _missing_prerequisites(results)
    if missing:
        return _wrap_result(
            status="failed",
            data=None,
            error=f"verify_budget prerequisites missing: {missing}",
            replan_hint=f"Cannot calculate budget: {missing} have no result yet.",
            trace=_minimal_trace("verify_budget", {"budget": budget},
                                 f"Missing: {missing}", ""),
        )

    activities_cost = 0.0
    meals_cost      = 0.0
    days_built      = 0
    for key, wrapped in results.items():
        if not key.startswith("build_day_schedule"):
            continue
        day_data = _inner_data(wrapped)
        if not isinstance(day_data, dict) or not day_data.get("slots"):
            continue
        if day_data.get("day", 0) > trip_days:
            continue
        days_built += 1
        for slot in day_data["slots"]:
            cost = float(slot.get("estimated_cost", 0))
            if slot.get("slot_type") == "meal":
                meals_cost += cost
            else:
                activities_cost += cost

    meal_per_day = meals_cost / days_built if days_built > 0 else 60.0
    avg_prices: Optional[dict] = None

    if mode == "with_travel_data":
        outbound_fl  = (state or {}).get("itinerary_selected_outbound_flight") or {}
        return_fl    = (state or {}).get("itinerary_selected_return_flight") or {}
        flight_price = float(outbound_fl.get("price", 0) or 0)
        ret_price    = float(return_fl.get("price", 0) or return_fl.get("total_price", 0) or 0)

        hotel_per_night = 0.0
        for key, wrapped in results.items():
            if key.startswith("build_day_schedule"):
                dd = _inner_data(wrapped)
                if isinstance(dd, dict) and dd.get("hotel_price_per_night"):
                    hotel_per_night = float(dd["hotel_price_per_night"])
                    break
    else:
        use_min  = bool((state or {}).get("use_min_prices_for_budget"))
        prefix   = "fetch_min_prices" if use_min else "fetch_avg_prices"
        prices   = _unwrap_data(results, prefix)

        if prices is not None:
            if use_min:
                hotel_per_night = float(prices.get("min_hotel_per_night", 120))
                flight_price    = float(prices.get("min_flight_price", 400))
                ret_price       = float(prices.get("min_return_flight_price", 400))
            else:
                hotel_per_night = float(prices.get("avg_hotel_per_night", 120))
                flight_price    = float(prices.get("avg_flight_price", 400))
                ret_price       = float(prices.get("avg_return_flight_price", 400))
            # Always store under avg_* keys so formatter reads consistently;
            # the note field distinguishes average vs minimum.
            avg_prices = {
                "avg_flight_price":        flight_price,
                "avg_return_flight_price": ret_price,
                "avg_hotel_per_night":     hotel_per_night,
                "note": "minimum available prices" if use_min else "estimated averages — no booking confirmed",
            }
        else:
            # Fallback: call the tool inline (handles runs without the new plan steps)
            try:
                tool_fn = get_min_location_cost if use_min else get_average_location_cost
                prices_live = tool_fn.invoke({
                    "destination": destination,
                    "origin":      origin,
                    "trip_days":   trip_days,
                })
                if use_min:
                    hotel_per_night = float(prices_live.get("min_hotel_per_night", 120))
                    flight_price    = float(prices_live.get("min_flight_price", 400))
                    ret_price       = float(prices_live.get("min_return_flight_price", 400))
                else:
                    hotel_per_night = float(prices_live.get("avg_hotel_per_night", 120))
                    flight_price    = float(prices_live.get("avg_flight_price", 400))
                    ret_price       = float(prices_live.get("avg_return_flight_price", 400))
                avg_prices = {
                    "avg_flight_price":        flight_price,
                    "avg_return_flight_price": ret_price,
                    "avg_hotel_per_night":     hotel_per_night,
                    "note": "minimum available prices" if use_min else "estimated averages — no booking confirmed",
                }
            except Exception:
                hotel_per_night = 120.0
                flight_price    = 400.0
                ret_price       = 400.0
                avg_prices = {
                    "avg_flight_price":        400.0,
                    "avg_return_flight_price": 400.0,
                    "avg_hotel_per_night":     120.0,
                    "note": "estimated fallback",
                }

    # Extract group size from state; defaults keep backward-compatibility.
    num_adults   = int((state or {}).get("num_adults")   or 1)
    num_children = int((state or {}).get("num_children") or 0)
    num_rooms    = int((state or {}).get("num_rooms")    or compute_default_rooms(num_adults, num_children))

    try:
        data = calculate_trip_cost.invoke({
            "flight_price":                   flight_price,
            "return_flight_price":            ret_price,
            "hotel_price_per_night":          hotel_per_night,
            "trip_days":                      trip_days,
            "estimated_activities_budget":    activities_cost,
            "estimated_meals_budget_per_day": meal_per_day,
            "num_adults":                     num_adults,
            "num_children":                   num_children,
            "num_rooms":                      num_rooms,
        })
    except Exception as exc:
        return _wrap_result(
            status="failed",
            data=None,
            error=f"calculate_trip_cost raised: {exc}",
            replan_hint=f"Budget tool failed: {exc}.",
            trace=_minimal_trace("verify_budget", {}, str(exc), ""),
        )

    grand_total       = float(data.get("grand_total", 0))
    group_grand_total = float(data.get("group_grand_total", grand_total))
    # Budget is the group's total budget — compare against the group cost.
    over_budget = bool(budget) and group_grand_total > budget * 1.05
    observation = (
        f"Grand total (solo): ${grand_total:.0f} | Group total: ${group_grand_total:.0f} | "
        f"Budget: ${budget or 'flexible'} | "
        + ("⚠️ OVER BUDGET" if over_budget else "✅ Within budget")
    )

    if mode == "standalone":
        data.pop("outbound_flight", None)
        data.pop("return_flight", None)
    if avg_prices is not None:
        data["avg_prices"] = avg_prices

    if over_budget:
        overage = group_grand_total - budget
        replan_hint = (
            f"Group budget exceeded by ${overage:.0f} "
            f"(group total ${group_grand_total:.0f} vs budget ${budget:.0f}). "
            "Options: cheaper activities, reduce trip_days by 1-2."
        )
        return _wrap_result(
            status="over_budget",
            data=data,
            error=f"Group budget exceeded by ${overage:.0f}.",
            replan_hint=replan_hint,
            trace=_minimal_trace("verify_budget",
                                 {"trip_days": trip_days, "budget": budget},
                                 observation, "Over budget."),
        )

    return _wrap_result(
        status="success",
        data=data,
        error=None,
        replan_hint="",
        trace=_minimal_trace("verify_budget",
                             {"trip_days": trip_days, "budget": budget},
                             observation, "Within budget."),
    )


# ---------------------------------------------------------------------------
# DayConfig builders
# ---------------------------------------------------------------------------

def _build_config_from_travel_plan(
    state: dict,
    day_num: int,
    trip_days: int,
    candidates,
    weather_cond: str = "",
    blocked_times: list | None = None,
    rest_blocks: list | None = None,
) -> DayConfig:
    hotel      = state.get("itinerary_selected_hotel") or {}
    flight_out = state.get("itinerary_selected_outbound_flight") or {}
    flight_ret = state.get("itinerary_selected_return_flight") or {}

    arrival_raw   = flight_out.get("arrival_time",   "14:00") if day_num == 1        else None
    departure_raw = flight_ret.get("departure_time", "20:00") if day_num == trip_days else None

    prefs = state.get("user_preferences") or {}
    return DayConfig(
        day_number=day_num,
        total_days=trip_days,
        hotel_name=hotel.get("name", "Hotel"),
        hotel_lat=float(hotel.get("latitude") or hotel.get("lat") or 48.85),
        hotel_lng=float(hotel.get("longitude") or hotel.get("lng") or 2.35),
        is_first_day=(day_num == 1),
        arrival_time=_normalize_time(str(arrival_raw)) if arrival_raw else None,
        is_last_day=(day_num == trip_days),
        departure_time=_normalize_time(str(departure_raw)) if departure_raw else None,
        day_start_time=prefs.get("day_start_time", "09:00"),
        day_end_time=prefs.get("day_end_time",   "21:00"),
        weather_condition=weather_cond,
        blocked_times=blocked_times or [],
        suggested_rest_blocks=rest_blocks or [],
    )


def _build_config_standalone(
    state: dict,
    day_num: int,
    trip_days: int,
    candidates,
    weather_cond: str = "",
    blocked_times: list | None = None,
    rest_blocks: list | None = None,
) -> DayConfig:
    lats = [c.lat for c in candidates if c.lat]
    lngs = [c.lng for c in candidates if c.lng]
    prefs = state.get("user_preferences") or {}
    return DayConfig(
        day_number=day_num,
        total_days=trip_days,
        hotel_name="",
        hotel_lat=sum(lats) / len(lats) if lats else 0.0,
        hotel_lng=sum(lngs) / len(lngs) if lngs else 0.0,
        is_first_day=(day_num == 1),
        arrival_time=None,
        is_last_day=(day_num == trip_days),
        departure_time=None,
        day_start_time=prefs.get("day_start_time", "09:00"),
        day_end_time=prefs.get("day_end_time",   "21:00"),
        weather_condition=weather_cond,
        blocked_times=blocked_times or [],
        suggested_rest_blocks=rest_blocks or [],
    )


def _hotel_price(state: dict) -> float:
    selected = state.get("itinerary_selected_hotel") or {}
    if selected.get("price_per_night"):
        return float(selected["price_per_night"])
    tp = state.get("travel_plan") or {}
    hotels = tp.get("hotels", [])
    return float(hotels[0].get("price_per_night", 0)) if hotels else 0.0


# ---------------------------------------------------------------------------
# fetch_activities helpers
# ---------------------------------------------------------------------------

_DIETARY_KEYWORDS: dict[str, list[str]] = {
    "kosher":      ["kosher"],
    "vegetarian":  ["vegetarian", "veggie", "plant"],
    "vegan":       ["vegan", "plant-based", "plant based"],
    "halal":       ["halal"],
    "gluten_free": ["gluten-free", "gluten free"],
}


def _filter_restaurants_by_dietary(
    restaurants: list[dict],
    dietary: str,
) -> list[dict]:
    """Soft filter: keep restaurants matching the dietary restriction.
    Returns empty list if nothing matches (caller decides whether to fall back)."""
    dietary_lower = dietary.lower()
    keywords = next(
        (kws for key, kws in _DIETARY_KEYWORDS.items() if key in dietary_lower),
        [dietary_lower],
    )
    matched = []
    for r in restaurants:
        text = " ".join([
            str(r.get("name") or ""),
            str(r.get("categories") or ""),
            " ".join(r.get("types") or []),
        ]).lower()
        if any(kw in text for kw in keywords):
            matched.append(r)
    return matched


def _extract_planner_prefs(prefs: dict) -> dict:
    """Return only the fields ActivitySelector uses; strip hotel/flight/price fields."""
    return {
        k: v for k, v in {
            "dietary_restrictions": prefs.get("dietary_restrictions"),
            "preferred_location":   prefs.get("preferred_location"),
            "blocked_times":        prefs.get("blocked_times"),
        }.items()
        if v is not None
    }


def _resolve_weather(results: dict, destination: str) -> str:
    """Return weather condition string for the day planner.
    Priority: fetch_weather result → seasonal average → empty string."""
    weather_data = _unwrap_data(results, "fetch_weather")
    if isinstance(weather_data, dict):
        cond = str(
            weather_data.get("condition") or
            weather_data.get("summary") or
            weather_data.get("description") or ""
        ).strip()
        if cond:
            return cond

    try:
        import datetime as _dt
        season_map = {
            12: "Winter", 1: "Winter",  2: "Winter",
             3: "Spring", 4: "Spring",  5: "Spring",
             6: "Summer", 7: "Summer",  8: "Summer",
             9: "Autumn", 10: "Autumn", 11: "Autumn",
        }
        season = season_map[_dt.date.today().month]
        avg_w  = get_average_weather.invoke({"city": destination, "season": season})
        cond   = str(avg_w.get("condition") or avg_w.get("summary") or "").strip()
        if cond:
            return cond
    except Exception:
        pass

    return ""


# ---------------------------------------------------------------------------
# Utility functions shared across handlers
# ---------------------------------------------------------------------------

def _is_empty(val) -> bool:
    if val is None:
        return True
    if isinstance(val, list):
        return len(val) == 0
    if isinstance(val, dict):
        if val.get("error"):
            return True
        return not any(v for k, v in val.items() if k != "error")
    return False


def _canonical_args(args: dict) -> dict:
    return {k: v for k, v in sorted(args.items())}


def _find_cached_result(results: dict, history: list, step_type: str, args: dict) -> Optional[dict]:
    never_cache = {"build_day_schedule"}
    if step_type in never_cache:
        return None
    for h in history:
        if (h.get("step_type") == step_type
                and h.get("status") == "success"
                and h.get("args") == args):
            k = next(
                (k for k in results
                 if k.startswith(step_type) and _inner_data(results[k]) is not None),
                None,
            )
            if k:
                return _inner_data(results[k])
    return None


def _inner_data(wrapped) -> Optional[dict]:
    if isinstance(wrapped, dict) and "status" in wrapped:
        return wrapped.get("data") if wrapped.get("status") == "success" else None
    if isinstance(wrapped, dict) and not wrapped.get("error"):
        return wrapped
    if isinstance(wrapped, list):
        return wrapped  # type: ignore[return-value]
    return None


def _unwrap_data(results: dict, prefix: str) -> Optional[dict]:
    for key in results:
        if key.startswith(prefix):
            return _inner_data(results[key])
    return None


# ---------------------------------------------------------------------------
# update_day_schedule helpers — Phase 1 deterministic quality
# ---------------------------------------------------------------------------

_BAD_WEATHER_KW = {"rain", "storm", "thunder", "shower"}
_OUTDOOR_KW_LOCAL = {
    "beach", "park", "garden", "promenade", "walk", "hike", "trail",
    "viewpoint", "outdoor", "open-air", "open air", "market", "bazaar",
    "boardwalk", "rooftop", "terrace", "square", "plaza",
}


def _is_outdoor_act(act: dict) -> bool:
    cats = str(act.get("categories") or act.get("types") or "").lower()
    name = str(act.get("name") or "").lower()
    return any(kw in cats + " " + name for kw in _OUTDOOR_KW_LOCAL)


def _is_bad_weather(cond: str) -> bool:
    return any(kw in cond.lower() for kw in _BAD_WEATHER_KW)


def _hotel_latlng(state: dict, mode: str, acts: list[dict]) -> tuple[float, float]:
    """Return hotel coordinates for geo-routing; falls back to activity centroid."""
    if mode == "with_travel_data":
        hotel = (state or {}).get("itinerary_selected_hotel") or {}
        lat = float(hotel.get("latitude") or hotel.get("lat") or 0)
        lng = float(hotel.get("longitude") or hotel.get("lng") or 0)
        if lat or lng:
            return lat, lng
    coords = [
        (float(a.get("lat") or a.get("latitude") or 0),
         float(a.get("lng") or a.get("longitude") or 0))
        for a in acts
        if float(a.get("lat") or a.get("latitude") or 0)
           or float(a.get("lng") or a.get("longitude") or 0)
    ]
    if coords:
        return sum(c[0] for c in coords) / len(coords), sum(c[1] for c in coords) / len(coords)
    return 0.0, 0.0


def _backfill_activities(
    day_act_names: list[str],
    available_acts: list[dict],
    target_count: int,
    act_idx: dict[str, dict],
    weather_cond: str,
) -> list[str]:
    """Fill day_act_names up to target_count with activities nearest the day's centroid.

    Skips outdoor activities when weather is bad (ScheduleEngine would skip them anyway).
    """
    if len(day_act_names) >= target_count:
        return day_act_names

    bad_weather = _is_bad_weather(weather_cond)

    coords = []
    for name in day_act_names:
        a = act_idx.get(name)
        if a:
            lat = float(a.get("lat") or a.get("latitude") or 0)
            lng = float(a.get("lng") or a.get("longitude") or 0)
            if lat or lng:
                coords.append((lat, lng))

    if coords:
        clat = sum(c[0] for c in coords) / len(coords)
        clng = sum(c[1] for c in coords) / len(coords)
    else:
        clat, clng = 0.0, 0.0

    anchor  = GeoPoint(lat=clat, lng=clng)
    current = set(day_act_names)

    scored: list[tuple[float, dict]] = []
    for a in available_acts:
        name = a.get("name", "")
        if not name or name in current:
            continue
        if bad_weather and _is_outdoor_act(a):
            continue
        lat = float(a.get("lat") or a.get("latitude") or 0)
        lng = float(a.get("lng") or a.get("longitude") or 0)
        dist = haversine_km(anchor, GeoPoint(lat=lat, lng=lng)) if (lat or lng) else float("inf")
        scored.append((dist, a))

    scored.sort(key=lambda x: x[0])

    result = list(day_act_names)
    for _, act in scored:
        if len(result) >= target_count:
            break
        result.append(act["name"])
    return result


def _geo_sort_activities(
    names: list[str],
    act_idx: dict[str, dict],
    hotel_lat: float,
    hotel_lng: float,
) -> list[str]:
    """Re-order activities by nearest-neighbor chain starting from the hotel.

    Produces the most geographically efficient routing for the ScheduleEngine.
    Activities without valid coordinates are appended at the end in original order.
    """
    if len(names) <= 1:
        return names

    remaining   = list(names)
    ordered:    list[str] = []
    curr_lat    = hotel_lat
    curr_lng    = hotel_lng

    while remaining:
        best_name: Optional[str] = None
        best_dist = float("inf")

        for name in remaining:
            a = act_idx.get(name)
            if not a:
                continue
            lat = float(a.get("lat") or a.get("latitude") or 0)
            lng = float(a.get("lng") or a.get("longitude") or 0)
            if not (lat or lng):
                continue
            dist = haversine_km(GeoPoint(lat=curr_lat, lng=curr_lng), GeoPoint(lat=lat, lng=lng))
            if dist < best_dist:
                best_dist = dist
                best_name = name

        if best_name is None:
            ordered.extend(remaining)
            break

        remaining.remove(best_name)
        ordered.append(best_name)
        a        = act_idx[best_name]
        curr_lat = float(a.get("lat") or a.get("latitude") or 0)
        curr_lng = float(a.get("lng") or a.get("longitude") or 0)

    return ordered


def _repick_meals(
    day_plan: dict,
    day_act_names: list[str],
    act_idx: dict[str, dict],
    available_rests: list[dict],
) -> dict:
    """Re-pick meal venues nearest to the updated activity cluster centroid.

    Only replaces meal slots that were already filled in the original plan.
    Keeps the slot empty (None) if the LLM didn't pick one to begin with.
    """
    if not available_rests or not day_act_names:
        return day_plan

    coords = []
    for name in day_act_names:
        a = act_idx.get(name)
        if a:
            lat = float(a.get("lat") or a.get("latitude") or 0)
            lng = float(a.get("lng") or a.get("longitude") or 0)
            if lat or lng:
                coords.append((lat, lng))

    if not coords:
        return day_plan

    clat   = sum(c[0] for c in coords) / len(coords)
    clng   = sum(c[1] for c in coords) / len(coords)
    anchor = GeoPoint(lat=clat, lng=clng)

    def _dist(r: dict) -> float:
        lat = float(r.get("lat") or r.get("latitude") or 0)
        lng = float(r.get("lng") or r.get("longitude") or 0)
        return haversine_km(anchor, GeoPoint(lat=lat, lng=lng)) if (lat or lng) else float("inf")

    sorted_rests = sorted(available_rests, key=_dist)
    used: set[str] = set()

    def _pick(prefer_cafe: bool = False) -> Optional[str]:
        for r in sorted_rests:
            name = r.get("name", "")
            if not name or name in used:
                continue
            if prefer_cafe:
                cats = str(r.get("categories") or r.get("types") or "").lower()
                if not any(k in cats for k in ("café", "cafe", "coffee", "bakery", "patisserie", "breakfast")):
                    continue
            used.add(name)
            return name
        if prefer_cafe:
            for r in sorted_rests:
                name = r.get("name", "")
                if name and name not in used:
                    used.add(name)
                    return name
        return None

    updated = {**day_plan}
    if day_plan.get("lunch_restaurant"):
        new = _pick()
        if new:
            updated["lunch_restaurant"] = new
    if day_plan.get("dinner_restaurant"):
        new = _pick()
        if new:
            updated["dinner_restaurant"] = new
    if day_plan.get("breakfast_place"):
        new = _pick(prefer_cafe=True)
        if new:
            updated["breakfast_place"] = new
    return updated


def _search_activity(name: str, destination: str) -> dict:
    """Look up an activity by name via Google Maps; returns a stub with unknown cost on failure."""
    try:
        results = search_places(f"{name} in {destination}")
        if results:
            match = next((r for r in results if name.lower() in r.get("name", "").lower()), results[0])
            return match
    except Exception:
        pass
    return {
        "name": name,
        "lat": 0, "lng": 0,
        "avg_duration_minutes": 120,
        "price": 0,
        "opening_time": "09:00",
        "closing_time": "21:00",
        "food_available": False,
        "categories": "attraction",
        "rating": 4.0,
        "requires_booking": False,
        "operating_days": "Daily",
    }


def _missing_prerequisites(results: dict) -> list[str]:
    return [
        p for p in BUILD_PREREQUISITES
        if not any(
            k.startswith(p) and _inner_data(results[k]) is not None
            for k in results
        )
    ]


def _used_activities(results: dict, current_plan_keys: set) -> set:
    used: set[str] = set()
    for key, wrapped in results.items():
        if key not in current_plan_keys:
            continue
        day_data = _inner_data(wrapped)
        if not isinstance(day_data, dict):
            continue
        for slot in day_data.get("slots", []):
            if slot.get("slot_type") == "activity" and slot.get("name"):
                used.add(slot["name"])
    return used


_GENERIC_MEAL_NAMES = frozenset({
    "Morning Coffee & Pastry", "Lunch", "Dinner", "Snack / coffee break",
})


def _used_meal_venues(results: dict, current_plan_keys: set) -> set:
    """Collect named restaurant/venue names already used in meal slots on prior days."""
    used: set[str] = set()
    for key, wrapped in results.items():
        if key not in current_plan_keys:
            continue
        day_data = _inner_data(wrapped)
        if not isinstance(day_data, dict):
            continue
        for slot in day_data.get("slots", []):
            if slot.get("slot_type") == "meal":
                name = slot.get("name", "")
                if name and name not in _GENERIC_MEAL_NAMES:
                    used.add(name)
    return used


def _normalize_time(raw: str) -> str:
    if not raw:
        return "12:00"
    if "T" in raw:
        raw = raw.split("T")[1]
    if " " in raw:
        raw = raw.split(" ")[1]
    return raw[:5]


def _minimal_trace(action, args, observation, reflection) -> dict:
    return {"thought": "", "action": action, "args": args,
            "observation": observation, "reflection": reflection}


def _drop_stale_budget(results: dict) -> dict:
    """Remove any verify_budget entries so the next run gets a fresh calculation."""
    return {k: v for k, v in results.items() if not k.startswith("verify_budget")}


def _wrap_result(status, data, error, replan_hint, trace) -> dict:
    return {"status": status, "data": data, "error": error,
            "replan_hint": replan_hint, "trace": trace}


# ---------------------------------------------------------------------------
# Special-events helpers
# ---------------------------------------------------------------------------

def _already_placed_event_names(
    results: dict, up_to_day: int, candidate_names: set[str]
) -> set[str]:
    """Return which candidate event names were placed as activity slots in days < up_to_day."""
    placed: set[str] = set()
    for key, wrapped in results.items():
        if not key.startswith("build_day_schedule") or not isinstance(wrapped, dict):
            continue
        day_data = wrapped.get("data", {})
        if not isinstance(day_data, dict) or day_data.get("day", 0) >= up_to_day:
            continue
        for slot in day_data.get("slots", []):
            if slot.get("slot_type") == "activity" and slot.get("name") in candidate_names:
                placed.add(slot["name"])
    return placed


def _get_day_events(state: dict, day_num: int, results: dict | None = None) -> list[dict]:
    """Return special events that fall on the calendar date for *day_num*.

    Uses state["trip_start"] to compute which calendar date corresponds to this day number,
    then filters from special_events_data. Deduplicates against prior built days so an
    all-month event (e.g. Christmas market) only appears once in the itinerary.
    """
    events     = (state or {}).get("special_events_data") or []
    trip_start = (state or {}).get("trip_start", "")
    if not events:
        return []
    # Fallback: when the user never stated a start date but flights are booked, use the
    # outbound departure date so events still get scheduled.
    if not trip_start:
        fl = (state or {}).get("itinerary_selected_outbound_flight") or {}
        trip_start = str(fl.get("departure_date") or fl.get("departure_time") or "")
    if not trip_start:
        return []
    try:
        s = trip_start.strip()
        # "YYYY-MM" (month-only) → assume the 1st; keeps generic itineraries working
        if len(s) == 7 and s[4] == "-":
            s = s + "-01"
        base      = datetime.date.fromisoformat(s[:10])
        trip_date = (base + datetime.timedelta(days=day_num - 1)).isoformat()
    except (ValueError, TypeError):
        return []

    matched = [e for e in events if trip_date in (e.get("applicable_dates") or [])]

    # Dedup: skip events already placed as activity slots on a prior day.
    # If the ScheduleEngine couldn't fit the event on day N, it won't appear in day N's
    # slots, so it remains eligible for day N+1 — giving it a second chance.
    if results and matched:
        candidate_names = {e.get("name", "") for e in matched}
        already = _already_placed_event_names(results, day_num, candidate_names)
        matched = [e for e in matched if e.get("name") not in already]

    return matched


def _enrich_event_coords(events: list[dict], destination: str) -> list[dict]:
    """Attempt to geocode events that have a location_hint but no lat/lng.

    Uses the existing _search_activity helper (Google Maps) so the schedule engine
    can compute realistic transit times. Mutates each event dict in-place and
    returns the list unchanged (for chaining).
    """
    for ev in events:
        if ev.get("lat") and ev.get("lng"):
            continue
        hint = str(ev.get("location_hint") or "").strip()
        if not hint:
            continue
        try:
            query  = f"{hint} {destination}" if destination.lower() not in hint.lower() else hint
            result = _search_activity(hint, destination)
            lat = float(result.get("lat") or result.get("latitude") or 0)
            lng = float(result.get("lng") or result.get("longitude") or 0)
            if lat or lng:
                ev["lat"] = lat
                ev["lng"] = lng
        except Exception:  # noqa: BLE001
            pass
    return events
