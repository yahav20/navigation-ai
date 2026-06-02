"""
ItineraryExecutorNode — v5
==========================

Runs ONE step per graph invocation. After each step it writes
`itinerary_feasible` into state so the Replanner can gate the next step.

Flow per invocation:
  1. Load plan, results, history from state
  2. Dispatch to the correct handler based on step_type
  3. Write structured result to step_results
  4. Advance current_step_index
  5. If result.status == "failed" → set itinerary_feasible=False

Two modes (read from state["itinerary_mode"]):
  with_travel_data — DayConfig uses real arrival/departure/hotel from travel_plan.
                     verify_budget uses real flight + hotel prices.
  standalone       — DayConfig has no flight anchors, all days start at 08:00.
                     verify_budget calls get_average_location_cost for estimates.

Step handlers:
  fetch_activities → _run_fetch_activities  (fetch + ActivitySelector)
  fetch_weather    → _run_fetch_step        (simple tool call)
  build_day_*      → _run_build_day         (pure Python, ScheduleEngine)
  verify_budget    → _run_verify_budget     (pure Python, mode-aware)
"""
from __future__ import annotations

import json
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.nodes.itinerary.schemas import ExecutionPlan
from agent.nodes.itinerary.itinerary_tools import (
    get_weather,
    calculate_trip_cost,
    get_average_location_cost,
)
from tools.google_maps_attractions import fetch_attractions, fetch_restaurants
from tools.weather_and_time import get_average_weather
from agent.nodes.itinerary.schedule_engine import DayConfig, DayScheduleBuilder
from agent.nodes.itinerary.activity_selector import select_activities_per_day, resolve_candidates
from agent.state import AgentState

# ── Tool registry — simple fetch_* steps only ─────────────────────────────
def _args_fetch_weather(destination: str, **_) -> dict:
    return {"city": destination}

TOOL_REGISTRY: dict[str, tuple] = {
    "fetch_weather": (get_weather, _args_fetch_weather),
}

# ── Build prerequisites — must have successful results before build_day ───
BUILD_PREREQUISITES = {"fetch_activities"}

# ── ReAct prompts ──────────────────────────────────────────────────────────

REACT_THOUGHT_SYSTEM = """
You are a travel data-fetching agent about to execute one step.
Given the step type and its arguments, produce a brief reasoning trace.

Respond ONLY as JSON (no markdown fences):
{
  "thought": "<why this step is needed and any constraints to watch>",
  "replan_hint": "<what should change if this step returns empty or errors>"
}
"""

REACT_REFLECT_SYSTEM = """
You are evaluating the result of one travel data-fetching step.

Rules:
- A non-empty list or dict with real data → "success"
- An empty list [], empty dict {}, or None → "failed"
- A dict containing an "error" key with a value → "failed"

Respond ONLY as JSON (no markdown fences):
{
  "status": "success" | "failed",
  "reflection": "<one-sentence evaluation>",
  "replan_hint": "<specific corrective action for the Planner; empty string on success>"
}
"""


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class ItineraryExecutorNode:
    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm

    def __call__(self, state: AgentState) -> dict:
        plan_state = state.get("itinerary_plan", {})
        plan       = ExecutionPlan(**plan_state["execution_plan"])
        results    = dict(plan_state.get("step_results", {}))
        history: list[dict] = list(plan_state.get("execution_history", []))
        mode       = state.get("itinerary_mode", "standalone")

        destination = state.get("destination_city", "")
        origin      = state.get("current_city", "")
        trip_days   = state.get("trip_days", plan.total_days)
        budget      = state.get("total_budget", 0)
        prefs       = state.get("user_preferences", {})

        current_index = state.get("current_step_index", 0)

        if current_index >= len(plan.steps):
            return _state_update(current_index, plan_state, results, history,
                                 ["✅ All steps complete — passing to Replanner."],
                                 feasible=True)

        step     = plan.steps[current_index]
        step_key = f"{step.step_type}_{step.step_id}"

        log_lines = [
            f"⚙️ **Step {current_index + 1}/{len(plan.steps)}:** `{step.step_type}`"
            + (f" (Day {step.day})" if step.day else "")
        ]

        ctx = dict(destination=destination, origin=origin,
                   trip_days=trip_days, budget=budget, prefs=prefs)

        # ── Dispatch ───────────────────────────────────────────────────────
        if step.step_type in TOOL_REGISTRY:
            result = self._run_fetch_step(step, results, history, ctx)

        elif step.step_type == "fetch_activities":
            result = self._run_fetch_activities(step, results, history, ctx, trip_days)

        elif step.step_type == "build_day_schedule":
            current_plan_keys = {f"{s.step_type}_{s.step_id}" for s in plan.steps}
            result = self._run_build_day(
                step, results, destination, trip_days, current_plan_keys, mode, state,
            )

        elif step.step_type == "verify_budget":
            result = self._run_verify_budget(results, budget, trip_days, destination, origin, mode, state)

        else:
            result = _wrap_result(
                status="failed",
                data=None,
                error=f"Unknown step_type: `{step.step_type}`",
                replan_hint=f"Remove unknown step `{step.step_type}` from the plan.",
                trace=_minimal_trace(step.step_type, {}, "No handler registered.", ""),
            )

        return self._commit(step, step_key, result, current_index,
                            plan_state, results, history, log_lines)

    # ── Commit ─────────────────────────────────────────────────────────────

    def _commit(self, step, step_key, result, current_index,
                plan_state, results, history, log_lines) -> dict:
        results[step_key] = result
        history.append(_history_entry(step, result))

        status = result.get("status", "unknown")
        icon   = {"success": "✅", "failed": "❌", "fallback_used": "🚫"}.get(status, "•")
        log_lines.append(
            f"{icon} **{status}**"
            + (f" — {result['error']}" if result.get("error") else "")
        )
        if result.get("replan_hint"):
            log_lines.append(f"💡 *Hint:* {result['replan_hint']}")

        feasible = status == "success"
        return _state_update(current_index, plan_state, results, history,
                             log_lines, feasible=feasible)

    # ── fetch_weather handler (ReAct) ──────────────────────────────────────

    def _run_fetch_step(self, step, results, history, ctx) -> dict:
        tool_fn, arg_builder = TOOL_REGISTRY[step.step_type]
        args = _canonical_args(arg_builder(**ctx))

        cached = _find_cached_result(results, history, step.step_type, args)
        if cached is not None:
            return _wrap_result(
                status="success", data=cached, error=None, replan_hint="",
                trace=_minimal_trace(step.step_type, args, "Cache hit.", "Valid cached data."),
            )

        thought_json = self._react_thought(step.step_type, args, ctx)
        try:
            raw = tool_fn.invoke(args)
            observation = json.dumps(raw, ensure_ascii=False)[:600]
        except Exception as exc:
            raw = None
            observation = f"Tool exception: {exc}"

        reflection = self._react_reflect(step.step_type, args, raw, observation)
        status     = reflection.get("status", "failed" if _is_empty(raw) else "success")
        hint       = reflection.get("replan_hint", "")

        if _is_empty(raw):
            status = "failed"
            if not hint:
                hint = (
                    f"`{step.step_type}` returned no data for destination "
                    f"'{ctx.get('destination')}'. Check city name spelling."
                )

        return _wrap_result(
            status=status,
            data=raw,
            error=observation if status == "failed" else None,
            replan_hint=hint,
            trace={
                "thought":     thought_json.get("thought", ""),
                "action":      step.step_type,
                "args":        args,
                "observation": observation,
                "reflection":  reflection.get("reflection", ""),
            },
        )

    # ── fetch_activities handler ───────────────────────────────────────────

    def _run_fetch_activities(self, step, results, history, ctx, trip_days) -> dict:
        destination = ctx["destination"]
        prefs       = ctx["prefs"] or {}
        dietary     = str(prefs.get("dietary_restrictions") or "").lower().strip()
        pref_loc    = str(prefs.get("preferred_location") or "").strip()

        # Cache key based on destination only (API data doesn't vary by dietary filter —
        # the selector handles filtering downstream)
        cache_args = _canonical_args({"city": destination})
        cached = _find_cached_result(results, history, "fetch_activities", cache_args)
        if cached is not None:
            print(f"[DEBUG] fetch_activities: cache hit for '{destination}'")
            return _wrap_result(
                status="success", data=cached, error=None, replan_hint="",
                trace=_minimal_trace("fetch_activities", cache_args,
                                     "Cache hit.", "Using cached data."),
            )

        thought_json = self._react_thought("fetch_activities", cache_args, ctx)

        # ── 1. Fetch attractions from Google Maps ──────────────────────────
        # Use preferred_location as a query hint when available
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
            print(f"[DEBUG] fetch_attractions '{attraction_query}' → {len(raw_attractions)} results")
        except Exception as exc:
            print(f"[DEBUG] fetch_attractions FAILED: {exc}")

        if _is_empty(raw_attractions):
            print(f"[DEBUG] FAILED — no attractions for '{destination}'")
            return _wrap_result(
                status="failed",
                data=None,
                error=f"fetch_attractions returned nothing for '{destination}'.",
                replan_hint=(
                    f"Google Maps returned no attractions for '{destination}'. "
                    "Check the city name and GOOGLE_MAPS_API_KEY."
                ),
                trace={
                    "thought":     thought_json.get("thought", ""),
                    "action":      "fetch_attractions",
                    "args":        cache_args,
                    "observation": "Empty attractions list.",
                    "reflection":  "Cannot build schedule without activities.",
                },
            )

        # ── 2. Fetch restaurants from Google Maps ──────────────────────────
        # Build the cuisine query:
        #   • kosher / vegetarian / vegan  → pass as cuisine to get relevant results
        #   • preferred_location           → search near that area
        #   • no prefs                     → generic restaurants
        if dietary in ("kosher", "vegetarian", "vegan"):
            cuisine_hint = dietary
        else:
            cuisine_hint = None   # generic; selector will rank by proximity

        rest_query_parts = []
        if cuisine_hint:
            rest_query_parts.append(cuisine_hint)
        if pref_loc:
            rest_query_parts.append(f"near {pref_loc}")
        rest_query_parts.append(f"in {destination}")

        raw_restaurants: list[dict] = []
        try:
            rest_invoke_args: dict = {"city": destination}
            if cuisine_hint:
                rest_invoke_args["cuisine"] = cuisine_hint
            raw_restaurants = fetch_restaurants.invoke(rest_invoke_args) or []
            print(f"[DEBUG] fetch_restaurants {rest_invoke_args} → {len(raw_restaurants)} results")
        except Exception as exc:
            print(f"[DEBUG] fetch_restaurants FAILED: {exc}")

        # Soft dietary filter: try to narrow down, but never return empty list
        if raw_restaurants and dietary:
            filtered = _filter_restaurants_by_dietary(raw_restaurants, dietary)
            if filtered:
                raw_restaurants = filtered
                print(f"[DEBUG] dietary filter '{dietary}' → {len(raw_restaurants)} restaurants kept")
            else:
                print(f"[DEBUG] dietary filter '{dietary}' matched nothing — keeping all {len(raw_restaurants)}")

        print(f"[DEBUG] final pool: {len(raw_attractions)} attractions, {len(raw_restaurants)} restaurants")

        # ── 3. Resolve weather ────────────────────────────────────────────
        weather_cond = _resolve_weather(results, destination)
        print(f"[DEBUG] weather_cond: {weather_cond!r}")

        # ── 4. Blocked times from user prefs ──────────────────────────────
        blocked_times = list(prefs.get("blocked_times") or [])
        if blocked_times:
            print(f"[DEBUG] blocked_times: {blocked_times}")

        # ── 5. ActivitySelector — Day Planner LLM ────────────────────────
        # Passes ONLY the fields ActivitySelector actually uses from prefs
        planner_prefs = _extract_planner_prefs(prefs)
        print(f"[DEBUG] calling ActivitySelector: prefs={planner_prefs}, weather={weather_cond!r}")
        try:
            selection = select_activities_per_day(
                llm=self.llm,
                activities=raw_attractions,
                trip_days=trip_days,
                prefs=planner_prefs,
                destination=destination,
                restaurants=raw_restaurants,
                weather=weather_cond or None,
                blocked_times=blocked_times or None,
            )
            selector_obs = f"Day Planner produced {len(selection)}-day plan."
            print(f"[DEBUG] ActivitySelector OK: {selector_obs}")
            for dk, dv in selection.items():
                if isinstance(dv, dict):
                    print(f"  [DEBUG] {dk}: theme={dv.get('theme')!r} "
                          f"area={dv.get('area')!r} "
                          f"acts={dv.get('activities')} "
                          f"lunch={dv.get('lunch_restaurant')} "
                          f"coffee={dv.get('coffee_place')} "
                          f"dinner={dv.get('dinner_restaurant')} "
                          f"rests={dv.get('recommended_rest_blocks')}")
        except Exception as exc:
            print(f"[DEBUG] ActivitySelector FAILED: {exc} — using rating fallback")
            sorted_acts = sorted(raw_attractions, key=lambda a: -a.get("rating", 0))
            selection = {
                f"day_{d}": {
                    "theme":             f"Day {d} — Explore {destination}",
                    "area":              "",
                    "activities":        [a["name"] for a in sorted_acts[(d - 1) * 5: d * 5]],
                    "lunch_restaurant":  None,
                    "coffee_place":      None,
                    "dinner_restaurant": None,
                    "recommended_rest_blocks": [],
                }
                for d in range(1, trip_days + 1)
            }
            selector_obs = f"Day Planner failed ({exc}); used rating-sorted fallback."

        observation = (
            f"Fetched {len(raw_attractions)} attractions + {len(raw_restaurants)} restaurants "
            f"from Google Maps. Weather: {weather_cond or 'unknown'}. {selector_obs}"
        )
        data = {
            "activities":  raw_attractions,
            "restaurants": raw_restaurants,
            "selection":   selection,
        }
        return _wrap_result(
            status="success",
            data=data,
            error=None,
            replan_hint="",
            trace={
                "thought":     thought_json.get("thought", ""),
                "action":      "fetch_activities",
                "args":        cache_args,
                "observation": observation,
                "reflection":  "Attractions and restaurants fetched; day plans created.",
            },
        )

    # ── build_day_schedule handler ─────────────────────────────────────────

    def _run_build_day(
        self, step, results, destination, trip_days,
        current_plan_keys, mode, state,
    ) -> dict:
        day_num = step.day or 1
        action  = f"build_day_schedule (Day {day_num})"
        print(f"[DEBUG] ── build_day_schedule Day {day_num} (mode={mode}) ──")

        missing = _missing_prerequisites(results)
        if missing:
            print(f"[DEBUG] Day {day_num}: prerequisites missing: {missing}")
            return _wrap_result(
                status="failed",
                data=None,
                error=f"Day {day_num}: prerequisites missing: {missing}",
                replan_hint=(
                    f"Day {day_num} cannot be built: {missing} have no successful result. "
                    f"Re-include them in the plan before build_day_schedule."
                ),
                trace=_minimal_trace(action, {"day": day_num},
                                     f"Prerequisites missing: {missing}", ""),
            )

        acts_data = _unwrap_data(results, "fetch_activities")
        all_activities:  list[dict] = []
        all_restaurants: list[dict] = []
        selection: dict = {}
        if isinstance(acts_data, dict):
            all_activities  = acts_data.get("activities",  [])
            all_restaurants = acts_data.get("restaurants", [])
            selection       = acts_data.get("selection",   {})

        print(f"[DEBUG] Day {day_num}: pool = {len(all_activities)} acts, {len(all_restaurants)} restaurants")

        # Support both new dict day-plan and legacy list-of-names
        day_plan_raw = selection.get(f"day_{day_num}", {})
        if isinstance(day_plan_raw, list):
            day_plan = {"activities": day_plan_raw}
        else:
            day_plan = day_plan_raw or {}

        print(f"[DEBUG] Day {day_num} plan: theme={day_plan.get('theme')!r} "
              f"area={day_plan.get('area')!r} "
              f"acts={day_plan.get('activities')} "
              f"lunch={day_plan.get('lunch_restaurant')} "
              f"coffee={day_plan.get('coffee_place')} "
              f"dinner={day_plan.get('dinner_restaurant')}")

        day_names  = day_plan.get("activities", [])
        used       = _used_activities(results, current_plan_keys)
        day_names  = [n for n in day_names if n not in used]
        candidates = resolve_candidates(all_activities, day_plan, all_restaurants)

        print(f"[DEBUG] Day {day_num}: {len(candidates)} candidates resolved "
              f"({len(used)} names already used across other days)")

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

        # ── Build DayConfig ────────────────────────────────────────────────
        if mode == "with_travel_data":
            cfg = self._build_config_from_travel_plan(state, day_num, trip_days, candidates, day_plan)
        else:
            cfg = self._build_config_standalone(state, day_num, trip_days, candidates, day_plan)

        print(f"[DEBUG] Day {day_num} DayConfig: hotel={cfg.hotel_name!r} "
              f"weather={cfg.weather_condition!r} "
              f"blocked={cfg.blocked_times} "
              f"start={cfg.day_start_time} end={cfg.day_end_time}")

        # ── Run ScheduleEngine ─────────────────────────────────────────────
        try:
            builder = DayScheduleBuilder(cfg)
            slots   = builder.build(candidates, day_plan=day_plan)
            print(f"[DEBUG] Day {day_num}: ScheduleEngine built {len(slots)} slots")
            for s in slots:
                print(f"  [DEBUG]   {s.get('time')}–{s.get('end_time')} [{s.get('slot_type')}] {s.get('name')}")
        except Exception as exc:
            print(f"[DEBUG] Day {day_num}: ScheduleEngine EXCEPTION: {exc}")
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

        data = {
            "day":   day_num,
            "theme": theme,
            "area":  area,
            "slots": slots,
            "day_cost": day_cost,
            "hotel":    cfg.hotel_name,
            "hotel_price_per_night": 0.0 if mode == "standalone" else self._hotel_price(state),
        }
        return _wrap_result(
            status="success", data=data, error=None, replan_hint="",
            trace=_minimal_trace(action, {"day": day_num, "candidates": len(candidates)},
                                 f"Built {len(slots)} slots. Day cost: ${day_cost}.",
                                 f"Day {day_num} complete."),
        )

    def _build_config_from_travel_plan(
        self, state, day_num, trip_days, candidates, day_plan=None,
    ) -> DayConfig:
        day_plan = day_plan or {}
        # Raw hotel resolved by PlanCheckNode (has lat/lng/price)
        hotel = state.get("itinerary_selected_hotel") or {}
        flight_out = state.get("itinerary_selected_outbound_flight") or {}
        flight_ret = state.get("itinerary_selected_return_flight") or {}

        hotel_name = hotel.get("name", "Hotel")
        hotel_lat  = float(hotel.get("latitude") or hotel.get("lat") or 48.85)
        hotel_lng  = float(hotel.get("longitude") or hotel.get("lng") or 2.35)
        hotel_bk   = bool(hotel.get("breakfast_included") or hotel.get("breakfast_available", False))

        arrival_raw   = flight_out.get("arrival_time",   "14:00") if day_num == 1        else None
        departure_raw = flight_ret.get("departure_time", "20:00") if day_num == trip_days else None

        prefs = state.get("user_preferences") or {}
        weather_data = _unwrap_data(state.get("itinerary_plan", {}).get("step_results", {}), "fetch_weather") or {}
        weather_cond = str(weather_data.get("condition") or weather_data.get("summary") or "")
        blocked_times = list(prefs.get("blocked_times") or [])
        day_blocked   = [b for b in blocked_times if isinstance(b, dict) and b.get("day") == day_num]

        return DayConfig(
            day_number=day_num,
            total_days=trip_days,
            hotel_name=hotel_name,
            hotel_lat=hotel_lat,
            hotel_lng=hotel_lng,
            hotel_has_breakfast=hotel_bk,
            is_first_day=(day_num == 1),
            arrival_time=_normalize_time(str(arrival_raw)) if arrival_raw else None,
            is_last_day=(day_num == trip_days),
            departure_time=_normalize_time(str(departure_raw)) if departure_raw else None,
            day_start_time=prefs.get("day_start_time", "09:00"),
            day_end_time=prefs.get("day_end_time",   "21:00"),
            weather_condition=weather_cond,
            blocked_times=day_blocked,
            suggested_rest_blocks=day_plan.get("recommended_rest_blocks", []),
        )

    def _build_config_standalone(
        self, state, day_num, trip_days, candidates, day_plan=None,
    ) -> DayConfig:
        day_plan = day_plan or {}
        """Derive a city-centre home base from activity coordinates."""
        lats = [c.lat for c in candidates if c.lat]
        lngs = [c.lng for c in candidates if c.lng]
        center_lat = sum(lats) / len(lats) if lats else 0.0
        center_lng = sum(lngs) / len(lngs) if lngs else 0.0

        prefs = state.get("user_preferences") or {}
        weather_data = _unwrap_data(state.get("itinerary_plan", {}).get("step_results", {}), "fetch_weather") or {}
        weather_cond = str(weather_data.get("condition") or weather_data.get("summary") or "")
        blocked_times = list(prefs.get("blocked_times") or [])
        day_blocked   = [b for b in blocked_times if isinstance(b, dict) and b.get("day") == day_num]

        return DayConfig(
            day_number=day_num,
            total_days=trip_days,
            hotel_name="",
            hotel_lat=center_lat,
            hotel_lng=center_lng,
            hotel_has_breakfast=False,
            is_first_day=(day_num == 1),
            arrival_time=None,
            is_last_day=(day_num == trip_days),
            departure_time=None,
            day_start_time=prefs.get("day_start_time", "09:00"),
            day_end_time=prefs.get("day_end_time",   "21:00"),
            weather_condition=weather_cond,
            blocked_times=day_blocked,
            suggested_rest_blocks=day_plan.get("recommended_rest_blocks", []),
        )

    def _hotel_price(self, state) -> float:
        tp = state.get("travel_plan") or {}
        hotels = tp.get("hotels", [])
        if hotels:
            return float(hotels[0].get("price_per_night", 0))
        return 0.0

    # ── verify_budget handler ──────────────────────────────────────────────

    def _run_verify_budget(self, results, budget, trip_days, destination, origin, mode, state=None) -> dict:
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

        # Collect activities + meals cost from built day schedules
        activities_cost = 0.0
        meals_cost      = 0.0
        days_built      = 0
        for key, wrapped in results.items():
            if not key.startswith("build_day_schedule"):
                continue
            day_data = _inner_data(wrapped)
            if not isinstance(day_data, dict) or not day_data.get("slots"):
                continue
            days_built += 1
            for slot in day_data["slots"]:
                cost = float(slot.get("estimated_cost", 0))
                if slot.get("slot_type") == "meal":
                    meals_cost += cost
                else:
                    activities_cost += cost

        meal_per_day = meals_cost / days_built if days_built > 0 else 60.0

        # Get flight + hotel prices based on mode
        avg_prices: Optional[dict] = None  # populated in standalone mode for formatter

        if mode == "with_travel_data":
            # Read actual flight prices from the resolved raw flights
            _state = state or {}
            outbound_fl = _state.get("itinerary_selected_outbound_flight") or {}
            return_fl   = _state.get("itinerary_selected_return_flight") or {}
            flight_price = float(outbound_fl.get("price", 0) or 0)
            ret_price    = float(return_fl.get("price", 0) or return_fl.get("total_price", 0) or 0)

            # Read hotel price from any built day result
            hotel_per_night = 0.0
            for key, wrapped in results.items():
                if key.startswith("build_day_schedule"):
                    dd = _inner_data(wrapped)
                    if isinstance(dd, dict) and dd.get("hotel_price_per_night"):
                        hotel_per_night = float(dd["hotel_price_per_night"])
                        break
        else:
            # Standalone: fetch average prices for the formatter display, but do NOT
            # count flights in the budget — the user opted not to book flights.
            try:
                avg = get_average_location_cost.invoke({
                    "destination": destination,
                    "origin":      origin,
                    "trip_days":   trip_days,
                })
                hotel_per_night = float(avg.get("avg_hotel_per_night", 120))
                avg_prices = avg
            except Exception:
                hotel_per_night = 120.0
                avg_prices = {
                    "avg_flight_price":        400.0,
                    "avg_return_flight_price": 400.0,
                    "avg_hotel_per_night":     hotel_per_night,
                    "note": "estimated fallback",
                }
            # Flights are not booked in standalone — exclude from the budget gate.
            flight_price = 0.0
            ret_price    = 0.0

        try:
            data = calculate_trip_cost.invoke({
                "flight_price":                  flight_price,
                "return_flight_price":           ret_price,
                "hotel_price_per_night":         hotel_per_night,
                "trip_days":                     trip_days,
                "estimated_activities_budget":   activities_cost,
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
        replan_hint = ""
        if over_budget:
            overage = grand_total - budget
            replan_hint = (
                f"Budget exceeded by ${overage:.0f}. "
                "Options: cheaper activities, reduce trip_days by 1-2."
            )

        # Standalone: drop the 0-valued flight rows — they're misleading in the budget table.
        if mode == "standalone":
            data.pop("outbound_flight", None)
            data.pop("return_flight", None)

        # Store average prices in the payload so formatter can render estimated prices section.
        if avg_prices is not None:
            data["avg_prices"] = avg_prices

        return _wrap_result(
            status="success",
            data=data,
            error=None,
            replan_hint=replan_hint,
            trace=_minimal_trace("verify_budget",
                                 {"trip_days": trip_days, "budget": budget},
                                 observation,
                                 "Over budget." if over_budget else "Within budget."),
        )

    # ── ReAct helpers ──────────────────────────────────────────────────────

    def _react_thought(self, step_type, args, ctx) -> dict:
        prompt = (
            f"Step: {step_type}\nArgs: {json.dumps(args, ensure_ascii=False)}\n"
            f"Context: destination={ctx.get('destination')}, origin={ctx.get('origin')}, "
            f"budget=${ctx.get('budget')}, days={ctx.get('trip_days')}"
        )
        try:
            raw = self.llm.invoke([
                SystemMessage(content=REACT_THOUGHT_SYSTEM),
                HumanMessage(content=prompt),
            ]).content.strip()
            return json.loads(_strip_fences(raw))
        except Exception:
            return {"thought": f"Executing {step_type}.", "replan_hint": ""}

    def _react_reflect(self, step_type, args, result, observation) -> dict:
        prompt = (
            f"Step: {step_type}\nArgs: {json.dumps(args, ensure_ascii=False)}\n"
            f"Observation (truncated): {observation[:500]}"
        )
        try:
            raw = self.llm.invoke([
                SystemMessage(content=REACT_REFLECT_SYSTEM),
                HumanMessage(content=prompt),
            ]).content.strip()
            return json.loads(_strip_fences(raw))
        except Exception:
            empty = _is_empty(result)
            return {
                "status":      "failed" if empty else "success",
                "reflection":  "LLM reflection unavailable.",
                "replan_hint": f"`{step_type}` returned empty." if empty else "",
            }


# ---------------------------------------------------------------------------
# fetch_activities helpers
# ---------------------------------------------------------------------------

def _resolve_weather(results: dict, destination: str) -> str:
    """
    Extract a weather condition string for the Day Planner.
    Priority:
      1. fetch_weather step result (real forecast)
      2. get_average_weather for current season (seasonal average)
      3. Empty string (planner will ignore)
    """
    weather_data = _unwrap_data(results, "fetch_weather")
    if isinstance(weather_data, dict):
        cond = str(
            weather_data.get("condition") or
            weather_data.get("summary") or
            weather_data.get("description") or ""
        ).strip()
        if cond:
            return cond

    # Seasonal fallback
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
            print(f"[DEBUG] _resolve_weather: seasonal fallback ({season}) → {cond!r}")
            return cond
    except Exception as exc:
        print(f"[DEBUG] _resolve_weather seasonal fallback failed: {exc}")

    return ""


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
    """
    Soft filter: keep only restaurants whose name/categories/types match
    the dietary restriction.  Returns empty list if nothing matches
    (caller decides whether to fall back to full list).
    """
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
    """
    Return only the UserPreferences fields the ActivitySelector actually uses.
    Strips hotel/flight/price fields — those are irrelevant for day planning.
    """
    return {
        k: v for k, v in {
            "dietary_restrictions": prefs.get("dietary_restrictions"),
            "preferred_location":   prefs.get("preferred_location"),
            "blocked_times":        prefs.get("blocked_times"),
        }.items()
        if v is not None
    }


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------

def _is_empty(val) -> bool:
    if val is None:
        return True
    if isinstance(val, list):
        return len(val) == 0
    if isinstance(val, dict):
        error_val = val.get("error")
        if error_val:   # non-None, non-empty string
            return True
        return not any(v for k, v in val.items() if k != "error")
    return False


def _strip_fences(s: str) -> str:
    if s.startswith("```"):
        parts = s.split("```")
        s = parts[1] if len(parts) > 1 else s
        s = s.lstrip("json").strip()
    return s


def _canonical_args(args: dict) -> dict:
    return {k: v for k, v in sorted(args.items())}


def _find_cached_result(results, history, step_type, args) -> Optional[dict]:
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
        if wrapped.get("status") == "success":
            return wrapped.get("data")
        return None
    if isinstance(wrapped, dict) and not wrapped.get("error"):
        return wrapped
    if isinstance(wrapped, list):
        return wrapped  # type: ignore[return-value]
    return None


def _unwrap_data(results, prefix) -> Optional[dict]:
    for key in results:
        if key.startswith(prefix):
            return _inner_data(results[key])
    return None


def _missing_prerequisites(results: dict) -> list[str]:
    hard_prereqs = {"fetch_activities"}
    return [
        p for p in hard_prereqs
        if not any(
            k.startswith(p) and _inner_data(results[k]) is not None
            for k in results
        )
    ]


def _used_activities(results, current_plan_keys) -> set:
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


def _history_entry(step, result) -> dict:
    return {
        "step_type":   step.step_type,
        "step_id":     step.step_id,
        "status":      result.get("status", "unknown"),
        "error":       result.get("error"),
        "replan_hint": result.get("replan_hint", ""),
        "args":        result.get("trace", {}).get("args", {}),
    }


def _state_update(current_index, plan_state, results, history,
                  log_lines, feasible) -> dict:
    return {
        "current_step_index": current_index + 1,
        "itinerary_feasible": feasible,
        "itinerary_plan": {
            **plan_state,
            "step_results":      results,
            "execution_history": history,
        },
        "messages": [AIMessage(content="\n".join(log_lines), name="executor_log")],
    }