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

import json
from typing import Optional

from langchain_core.language_models import BaseChatModel

from agent.nodes.itinerary.itinerary_tools import (
    calculate_trip_cost,
    get_average_location_cost,
    get_min_location_cost,
    get_weather,
)
from agent.nodes.itinerary.schedule_engine import DayConfig, DayScheduleBuilder
from agent.nodes.itinerary.activity_selector import select_activities_per_day, resolve_candidates
from agent.nodes.itinerary.schemas import PlanStep
from tools.google_maps_attractions import fetch_attractions, fetch_restaurants
from tools.weather_and_time import get_average_weather

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
        return _wrap_result(
            status="failed",
            data=None,
            error=f"fetch_attractions returned nothing for '{destination}'.",
            replan_hint=(
                f"Google Maps returned no attractions for '{destination}'. "
                "Check the city name and GOOGLE_MAPS_API_KEY."
            ),
            trace=_minimal_trace("fetch_activities", cache_args,
                                 "Empty attractions list.",
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

    # Deduplicate: exclude activities already scheduled on previous days
    used              = _used_activities(results, current_plan_keys)
    available_acts    = [a for a in all_activities  if a.get("name") not in used]
    available_rests   = [r for r in all_restaurants if r.get("name") not in used]
    day_act_names     = [n for n in day_plan.get("activities", []) if n not in used]
    day_plan_filtered = {**day_plan, "activities": day_act_names}

    candidates = resolve_candidates(available_acts, day_plan_filtered, available_rests)

    if not candidates:
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

    try:
        slots = DayScheduleBuilder(cfg).build(candidates, day_plan=day_plan_filtered)
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
                             f"Built {len(slots)} slots. Day cost: ${day_cost}.",
                             f"Day {day_num} complete."),
    )


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

    try:
        data = calculate_trip_cost.invoke({
            "flight_price":                   flight_price,
            "return_flight_price":            ret_price,
            "hotel_price_per_night":          hotel_per_night,
            "trip_days":                      trip_days,
            "estimated_activities_budget":    activities_cost,
            "estimated_meals_budget_per_day": meal_per_day,
        })
    except Exception as exc:
        return _wrap_result(
            status="failed",
            data=None,
            error=f"calculate_trip_cost raised: {exc}",
            replan_hint=f"Budget tool failed: {exc}.",
            trace=_minimal_trace("verify_budget", {}, str(exc), ""),
        )

    grand_total = float(data.get("grand_total", 0))
    over_budget = bool(budget) and grand_total > budget * 1.05
    observation = (
        f"Grand total: ${grand_total:.0f} | Budget: ${budget or 'flexible'} | "
        + ("⚠️ OVER BUDGET" if over_budget else "✅ Within budget")
    )

    if mode == "standalone":
        data.pop("outbound_flight", None)
        data.pop("return_flight", None)
    if avg_prices is not None:
        data["avg_prices"] = avg_prices

    if over_budget:
        overage = grand_total - budget
        replan_hint = (
            f"Budget exceeded by ${overage:.0f}. "
            "Options: cheaper activities, reduce trip_days by 1-2."
        )
        return _wrap_result(
            status="over_budget",
            data=data,
            error=f"Budget exceeded by ${overage:.0f}.",
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
