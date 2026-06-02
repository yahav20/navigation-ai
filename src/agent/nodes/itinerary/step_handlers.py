"""
step_handlers.py — one function per itinerary step type.

Each handler receives the arguments it needs and returns a wrapped result dict
(see _wrap_result). The executor calls these via a dispatch map — it never
contains step logic itself.

Handlers:
  handle_fetch_weather     — simple tool call
  handle_fetch_activities  — tool call + ActivitySelector (LLM)
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
    get_weather,
    search_activities,
)
from agent.nodes.itinerary.schedule_engine import DayConfig, DayScheduleBuilder
from agent.nodes.itinerary.activity_selector import select_activities_per_day, resolve_candidates
from agent.nodes.itinerary.schemas import PlanStep

# ---------------------------------------------------------------------------
# Tool registry — used by handle_fetch_weather (and future simple fetch steps)
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


def handle_fetch_activities(
    step: PlanStep,
    results: dict,
    history: list,
    ctx: dict,
    trip_days: int,
    llm: BaseChatModel,
) -> dict:
    """Fetch activities for the destination then run ActivitySelector to assign days."""
    destination = ctx["destination"]
    prefs       = ctx["prefs"]
    dietary     = str(prefs.get("dietary_restrictions", "")).lower()

    args = _canonical_args({
        "city":                destination,
        "kosher_only":         "kosher" in dietary,
        "vegetarian_friendly": "vegetarian" in dietary,
        "vegan_friendly":      "vegan" in dietary,
    })

    cached = _find_cached_result(results, history, "fetch_activities", args)
    if cached is not None:
        return _wrap_result(
            status="success", data=cached, error=None, replan_hint="",
            trace=_minimal_trace("fetch_activities", args,
                                 "Cache hit.", "Using cached activities."),
        )

    try:
        raw: list[dict] = search_activities.invoke(args)
        observation = f"Fetched {len(raw) if isinstance(raw, list) else 0} activities."
    except Exception as exc:
        raw = None
        observation = f"search_activities exception: {exc}"

    if _is_empty(raw):
        return _wrap_result(
            status="failed",
            data=None,
            error=observation,
            replan_hint=(
                f"No activities found for '{destination}'. "
                "Verify city name or check that the DB has activities for this city."
            ),
            trace=_minimal_trace("fetch_activities", args, observation,
                                 "Empty activity list — cannot build schedule."),
        )

    try:
        selection = select_activities_per_day(
            llm=llm,
            activities=raw,
            trip_days=trip_days,
            prefs=prefs,
            destination=destination,
        )
        selector_obs = f"ActivitySelector assigned activities to {len(selection)} days."
    except Exception as exc:
        sorted_acts = sorted(raw, key=lambda a: -a.get("rating", 0))
        selection = {
            f"day_{d}": [a["name"] for a in sorted_acts[(d - 1) * 5 : d * 5]]
            for d in range(1, trip_days + 1)
        }
        selector_obs = f"ActivitySelector failed ({exc}); used rating-sorted fallback."

    return _wrap_result(
        status="success",
        data={"activities": raw, "selection": selection},
        error=None,
        replan_hint="",
        trace=_minimal_trace("fetch_activities", args,
                             observation + " | " + selector_obs,
                             "Activities fetched and assigned to days."),
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
    all_activities: list[dict] = []
    selection: dict[str, list[str]] = {}
    if isinstance(acts_data, dict):
        all_activities = acts_data.get("activities", [])
        selection      = acts_data.get("selection", {})

    day_names  = selection.get(f"day_{day_num}", [])
    used       = _used_activities(results, current_plan_keys)
    day_names  = [n for n in day_names if n not in used]
    candidates = resolve_candidates(all_activities, day_names)

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

    cfg = (
        _build_config_from_travel_plan(state, day_num, trip_days, candidates)
        if mode == "with_travel_data"
        else _build_config_standalone(state, day_num, trip_days, candidates)
    )

    try:
        slots = DayScheduleBuilder(cfg).build(candidates)
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
    return _wrap_result(
        status="success",
        data={
            "day":   day_num,
            "theme": day_themes.get(day_num, f"Day {day_num} — Explore {destination}"),
            "slots": slots,
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
        # Exclude days beyond current trip_days (can happen after a trip_days reduction replan)
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
        try:
            avg = get_average_location_cost.invoke({
                "destination": destination,
                "origin":      origin,
                "trip_days":   trip_days,
            })
            hotel_per_night = float(avg.get("avg_hotel_per_night", 120))
            flight_price    = float(avg.get("avg_flight_price", 400))
            ret_price       = float(avg.get("avg_return_flight_price", 400))
            avg_prices = avg
        except Exception:
            hotel_per_night = 120.0
            flight_price    = 400.0
            ret_price       = 400.0
            avg_prices = {
                "avg_flight_price":        400.0,
                "avg_return_flight_price": 400.0,
                "avg_hotel_per_night":     hotel_per_night,
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
# DayConfig builders (used only by handle_build_day)
# ---------------------------------------------------------------------------

def _build_config_from_travel_plan(state: dict, day_num: int, trip_days: int, candidates) -> DayConfig:
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
        hotel_has_breakfast=bool(hotel.get("breakfast_included") or hotel.get("breakfast_available", False)),
        is_first_day=(day_num == 1),
        arrival_time=_normalize_time(str(arrival_raw)) if arrival_raw else None,
        is_last_day=(day_num == trip_days),
        departure_time=_normalize_time(str(departure_raw)) if departure_raw else None,
        day_start_time=prefs.get("day_start_time", "09:00"),
        day_end_time=prefs.get("day_end_time",   "21:00"),
    )


def _build_config_standalone(state: dict, day_num: int, trip_days: int, candidates) -> DayConfig:
    lats = [c.lat for c in candidates if c.lat]
    lngs = [c.lng for c in candidates if c.lng]
    prefs = state.get("user_preferences") or {}
    return DayConfig(
        day_number=day_num,
        total_days=trip_days,
        hotel_name="",
        hotel_lat=sum(lats) / len(lats) if lats else 0.0,
        hotel_lng=sum(lngs) / len(lngs) if lngs else 0.0,
        hotel_has_breakfast=False,
        is_first_day=(day_num == 1),
        arrival_time=None,
        is_last_day=(day_num == trip_days),
        departure_time=None,
        day_start_time=prefs.get("day_start_time", "09:00"),
        day_end_time=prefs.get("day_end_time",   "21:00"),
    )


def _hotel_price(state: dict) -> float:
    tp = state.get("travel_plan") or {}
    hotels = tp.get("hotels", [])
    return float(hotels[0].get("price_per_night", 0)) if hotels else 0.0


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
    never_cache = {"build_day_schedule", "verify_budget"}
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


def _wrap_result(status, data, error, replan_hint, trace) -> dict:
    return {"status": status, "data": data, "error": error,
            "replan_hint": replan_hint, "trace": trace}
