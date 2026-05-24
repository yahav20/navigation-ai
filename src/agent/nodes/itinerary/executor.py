# src/agent/nodes/itinerary/executor.py
"""
ItineraryExecutorNode — v2
==========================
Key architectural change vs v1:
  - LLM is NO LONGER responsible for building the day schedule.
  - LLM only selects & ranks activities (ActivitySelector).
  - DayScheduleBuilder (schedule_engine.py) deterministically builds the schedule.

Flow per execution step:
  fetch_* steps   → unchanged (tool calls)
  fetch_activities → also triggers ActivitySelector (once, cached)
  build_day_N     → DayConfig + ActivityCandidate list → ScheduleEngine
  verify_budget   → unchanged (calculate_trip_cost tool)
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from langchain_core.language_models import BaseChatModel

from agent.nodes.itinerary.schemas import ExecutionPlan
from agent.nodes.itinerary.itinerary_tools import (
    search_outbound_flights, search_return_flights,
    search_hotels, search_activities,
    get_weather, calculate_trip_cost,
)
from agent.nodes.itinerary.schedule_engine import (
    DayConfig, DayScheduleBuilder,
)
from agent.nodes.itinerary.activity_selector import (
    select_activities_per_day, resolve_candidates,
)
from agent.state import AgentState

logger = logging.getLogger(__name__)

DEFAULT_MEALS_PER_DAY = 60.0


class ItineraryExecutorNode:
    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm

    def __call__(self, state: AgentState) -> dict:
        plan_state = state.get("itinerary_plan", {})
        plan = ExecutionPlan(**plan_state["execution_plan"])
        results = dict(plan_state.get("step_results", {}))

        destination = state.get("destination_city", "")
        origin = state.get("current_city", "")
        trip_days = state.get("trip_days", plan.total_days)
        budget = state.get("total_budget", 0)
        prefs = state.get("user_preferences", {})

        current_index = state.get("current_step_index", 0)

        if current_index >= len(plan.steps):
            return {"skipped": True}

        step = plan.steps[current_index]
        key = f"{step.step_type}_{step.step_id}"
        cache_key = step.step_type

        print(f"\n--- ⚙️ EXECUTING STEP {current_index + 1}/{len(plan.steps)}: {step.step_type} ---")

        # ── Cache check (skip duplicate tool calls on replanning) ──
        cached_key = next(
            (k for k in results
             if k.startswith(cache_key)
             and "error" not in str(results[k])
             and "skipped" not in str(results[k])),
            None
        )

        # build_day_schedule and verify_budget are NEVER cached (always recompute)
        skip_cache_types = {"build_day_schedule", "verify_budget", "fetch_activities"}

        if cached_key and step.step_type not in skip_cache_types:
            print(f"⏩ Skipping — using cached data for '{step.step_type}'.")
            results[key] = results[cached_key]
        else:
            try:
                results[key] = self._run(step, results, destination, origin,
                                         trip_days, budget, prefs, state)
                print(f"✅ Step {step.step_type} completed.")
            except Exception as e:
                logger.exception("Step %s failed", step.step_type)
                results[key] = {"error": str(e), "step_type": step.step_type}

        return {
            "current_step_index": current_index + 1,
            "itinerary_plan": {**plan_state, "step_results": results},
        }

    # ── Step dispatcher ────────────────────────────────────────────────────

    def _run(self, step, results, destination, origin, trip_days, budget, prefs, state):
        dietary = str(prefs.get("dietary_restrictions", "")).lower()
        kosher = "kosher" in dietary
        veg = "vegetarian" in dietary
        vegan = "vegan" in dietary

        if step.step_type == "fetch_flights":
            return search_outbound_flights.invoke({"origin": origin, "destination": destination})

        if step.step_type == "fetch_return_flights":
            return search_return_flights.invoke({"origin": destination, "destination": origin})

        if step.step_type == "fetch_hotels":
            return search_hotels.invoke({"city": destination, "kosher_only": kosher})

        if step.step_type == "fetch_activities":
            raw = search_activities.invoke({
                "city": destination,
                "kosher_only": kosher,
                "vegetarian_friendly": veg,
                "vegan_friendly": vegan,
            })
            # ── Activity selection is done once here and cached in results ──
            selection = select_activities_per_day(
                llm=self.llm,
                activities=raw if isinstance(raw, list) else [],
                trip_days=trip_days,
                prefs=prefs,
                destination=destination,
            )
            # Store both raw list and the LLM selection
            return {"activities": raw, "selection": selection}

        if step.step_type == "fetch_weather":
            return get_weather.invoke({"city": destination})

        if step.step_type == "build_day_schedule":
            return self._build_day(step, results, destination, trip_days, prefs)

        if step.step_type == "verify_budget":
            return self._verify_budget(results, budget, trip_days)

        return {"skipped": True}

    # ── Day builder (pure Python, no LLM) ─────────────────────────────────

    def _build_day(self, step, results, destination, trip_days, prefs):
        day_num = step.day or 1

        outbound = _first(results, "fetch_flights")
        ret = _first(results, "fetch_return_flights")
        hotel_raw = _first(results, "fetch_hotels")

        # fetch_activities now returns a dict with "activities" and "selection"
        acts_result = _first_dict(results, "fetch_activities")
        all_activities: list[dict] = (
            acts_result.get("activities", []) if acts_result else []
        )
        selection: dict[str, list[str]] = (
            acts_result.get("selection", {}) if acts_result else {}
        )

        # ── Hotel info ──
        hotel_name = (hotel_raw or {}).get("name", "Hotel") if isinstance(hotel_raw, dict) else "Hotel"
        hotel_lat = float((hotel_raw or {}).get("latitude") or (hotel_raw or {}).get("lat") or 48.85)
        hotel_lng = float((hotel_raw or {}).get("longitude") or (hotel_raw or {}).get("lng") or 2.35)
        hotel_bk = bool((hotel_raw or {}).get("breakfast_available", False)) if isinstance(hotel_raw, dict) else False
        hotel_price = float((hotel_raw or {}).get("price_per_night", 0)) if isinstance(hotel_raw, dict) else 0

        # ── Flight anchors ──
        arrival_time   = _normalize_time(_safe_str(outbound, "arrival_time",   "14:00"))
        departure_time = _normalize_time(_safe_str(ret,      "departure_time", "20:00"))

        # ── Build DayConfig ──
        cfg = DayConfig(
            day_number=day_num,
            total_days=trip_days,
            hotel_name=hotel_name,
            hotel_lat=hotel_lat,
            hotel_lng=hotel_lng,
            hotel_has_breakfast=hotel_bk,
            is_first_day=(day_num == 1),
            arrival_time=arrival_time if day_num == 1 else None,
            is_last_day=(day_num == trip_days),
            departure_time=departure_time if day_num == trip_days else None,
        )

        # ── Resolve activity candidates for this day ──
        day_key = f"day_{day_num}"
        day_names = selection.get(day_key, [])

        # Exclude activities already used in previous days
        used = _used_activities(results)
        day_names = [n for n in day_names if n not in used]

        candidates = resolve_candidates(all_activities, day_names)

        print(f"📅 Day {day_num}: {len(candidates)} candidate activities → ScheduleEngine")

        # ── Run deterministic scheduler ──
        builder = DayScheduleBuilder(cfg)
        slots = builder.build(candidates)

        day_cost = round(sum(float(s.get("estimated_cost", 0)) for s in slots), 2)

        day_themes = {
            1: f"Arrival & first impressions of {destination}",
            trip_days: f"Final day & farewell to {destination}",
        }
        theme = day_themes.get(day_num, f"Day {day_num} — Explore {destination}")

        return {
            "day": day_num,
            "theme": theme,
            "slots": slots,
            "day_cost": day_cost,
            "hotel": hotel_name,
            "hotel_price_per_night": hotel_price,
        }

    # ── Budget verifier ────────────────────────────────────────────────────

    def _verify_budget(self, results, budget, trip_days):
        outbound = _first(results, "fetch_flights")
        ret = _first(results, "fetch_return_flights")
        hotel_raw = _first(results, "fetch_hotels")

        activity_cost = sum(
            v.get("day_cost", 0)
            for k, v in results.items()
            if k.startswith("build_day_schedule") and isinstance(v, dict)
        )

        return calculate_trip_cost.invoke({
            "flight_price": _safe_float(outbound, "price"),
            "return_flight_price": _safe_float(ret, "price"),
            "hotel_price_per_night": _safe_float(hotel_raw, "price_per_night"),
            "trip_days": trip_days,
            "estimated_activities_budget": activity_cost,
            "estimated_meals_budget_per_day": DEFAULT_MEALS_PER_DAY,
        })


# ---------------------------------------------------------------------------
# Utility helpers
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
    if isinstance(v, dict) and not v.get("error"):
        return v
    return None


def _first_dict(results: dict, prefix: str) -> Optional[dict]:
    """Like _first but always returns the raw dict (not first element of list)."""
    k = _find_key(results, prefix)
    if not k:
        return None
    v = results[k]
    if isinstance(v, dict) and not v.get("error"):
        return v
    return None


def _safe_str(d: Optional[dict], key: str, default: str) -> str:
    if not d:
        return default
    return str(d.get(key) or default)


def _safe_float(d: Optional[dict], key: str, default: float = 0.0) -> float:
    if not d or not isinstance(d, dict):
        return default
    try:
        return float(d.get(key) or default)
    except (TypeError, ValueError):
        return default


def _used_activities(results: dict) -> set:
    used: set[str] = set()
    for v in results.values():
        if isinstance(v, dict) and "slots" in v:
            for s in v["slots"]:
                if s.get("slot_type") == "activity":
                    name = s.get("name", "")
                    if name:
                        used.add(name)
    return used
def _normalize_time(raw: str) -> str:
    """
    Accepts '18:00', '2026-06-01 18:00', '2026-06-01T18:00:00', etc.
    Always returns 'HH:MM'.
    """
    if not raw:
        return raw
    # ISO datetime with T
    if "T" in raw:
        raw = raw.split("T")[1]
    # 'YYYY-MM-DD HH:MM' or 'YYYY-MM-DD HH:MM:SS'
    if " " in raw:
        raw = raw.split(" ")[1]
    # 'HH:MM:SS' → 'HH:MM'
    return raw[:5]
