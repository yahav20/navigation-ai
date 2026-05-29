"""
ItineraryExecutorNode — v4
==========================

ARCHITECTURE
────────────
The Executor runs ONE step per graph invocation. After each step it writes
`itinerary_feasible` into state so the Observer can gate the next step.

Flow per invocation:
  1. Load plan, results, history from state
  2. Anti-loop guard — if step_type failed too many times → fallback_used
  3. Dispatch to the correct handler:
       fetch_*          → _run_fetch_step  (ReAct: Thought → Tool → Reflect)
       fetch_activities → _run_fetch_step  THEN _select_activities (separate concern)
       build_day_*      → _run_build_day   (pure Python, no LLM thought needed)
       verify_budget    → _run_verify_budget (pure Python)
  4. Write structured result to step_results
  5. Append to execution_history
  6. If result.status == "failed" → set itinerary_feasible=False so Observer
     knows to halt and replan BEFORE advancing to the next step

CRITICAL FIX: fetch_activities dispatch
  fetch_activities was registered in TOOL_REGISTRY *and* had a dead elif branch.
  Now: fetch_activities is NOT in TOOL_REGISTRY. It has its own dedicated
  handler (_run_fetch_activities) that fetches, then runs ActivitySelector.

CRITICAL FIX: failure signalling
  On any failed step the Executor now sets itinerary_feasible=False in the
  returned state. The Observer reads this on the *next* invocation's mid-exec
  check and triggers a replan before the next step runs.

STRUCTURED STEP RESULTS
  Every entry in step_results:
    {
      "status":      "success" | "failed" | "fallback_used",
      "data":        <raw output or None>,
      "error":       <str | None>,
      "replan_hint": <actionable string for Planner>,
      "trace": {
        "thought":     str,
        "action":      str,
        "args":        dict,
        "observation": str,
        "reflection":  str,
      }
    }

EXECUTION HISTORY
  plan_state["execution_history"] — list of:
    { step_type, step_id, status, error, replan_hint, args }
  Used by anti-loop guard and Observer's replan context builder.

TOOL DE-DUPLICATION
  Same (step_type, canonical_args) that previously succeeded → cache hit,
  no tool call. Failed steps are always retried (Planner put them there
  for a reason). build_day_schedule and verify_budget are never cached.

ANTI-LOOP GUARD
  If a step_type appears as "failed" ≥ MAX_STEP_RETRIES times in
  execution_history → return status="fallback_used" immediately.
"""
from __future__ import annotations

import json
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agent.nodes.itinerary.schemas import ExecutionPlan
from agent.nodes.itinerary.itinerary_tools import (
    search_outbound_flights, search_return_flights,
    search_hotels, search_activities,
    get_weather, calculate_trip_cost,
)
from agent.nodes.itinerary.schedule_engine import DayConfig, DayScheduleBuilder
from agent.nodes.itinerary.activity_selector import select_activities_per_day, resolve_candidates
from agent.llm import silent
from agent.state import AgentState

# ── Safety limits ──────────────────────────────────────────────────────────
MAX_STEP_RETRIES = 2   # failures of same step_type before fallback_used

# ── Tool registry — fetch_* steps only (NOT fetch_activities or compute steps)
# Each entry: step_type → (tool_fn, arg_builder_fn)
# ──────────────────────────────────────────────────────────────────────────

def _args_fetch_flights(destination: str, origin: str, **_) -> dict:
    return {"origin": origin, "destination": destination}

def _args_fetch_return_flights(destination: str, origin: str, **_) -> dict:
    return {"origin": destination, "destination": origin}

def _args_fetch_hotels(destination: str, prefs: dict, **_) -> dict:
    dietary = str(prefs.get("dietary_restrictions", "")).lower()
    return {"city": destination, "kosher_only": "kosher" in dietary}

def _args_fetch_weather(destination: str, **_) -> dict:
    return {"city": destination}

TOOL_REGISTRY: dict[str, tuple] = {
    "fetch_flights":        (search_outbound_flights,  _args_fetch_flights),
    "fetch_return_flights": (search_return_flights,    _args_fetch_return_flights),
    "fetch_hotels":         (search_hotels,            _args_fetch_hotels),
    "fetch_weather":        (get_weather,              _args_fetch_weather),
    # fetch_activities intentionally EXCLUDED — handled by _run_fetch_activities
}

# ── Step types that MUST succeed before build_day_schedule can run ─────────
# Used by the Planner dependency validator (see planner.py) AND by the
# Executor to generate an accurate replan_hint when a build step finds
# its prerequisites missing.
BUILD_PREREQUISITES = {"fetch_flights", "fetch_return_flights", "fetch_hotels", "fetch_activities"}

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
- A dict containing an "error" key → "failed"

Respond ONLY as JSON (no markdown fences):
{
  "status": "success" | "failed",
  "reflection": "<one-sentence evaluation>",
  "replan_hint": "<specific corrective action for the Planner; empty string on success>"
}
"""


# ── Node ───────────────────────────────────────────────────────────────────

class ItineraryExecutorNode:
    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = silent(llm)

    # ── Main entry point ───────────────────────────────────────────────────

    def __call__(self, state: AgentState) -> dict:
        plan_state    = state.get("itinerary_plan", {})
        plan          = ExecutionPlan(**plan_state["execution_plan"])
        results       = dict(plan_state.get("step_results", {}))
        history: list[dict] = list(plan_state.get("execution_history", []))

        destination = state.get("destination_city", "")
        origin      = state.get("current_city", "")
        trip_days   = state.get("trip_days", plan.total_days)
        budget      = state.get("total_budget", 0)
        prefs       = state.get("user_preferences", {})

        current_index = state.get("current_step_index", 0)

        if current_index >= len(plan.steps):
            # All steps done — let the Observer finalise
            return _state_update(current_index, plan_state, results, history,
                                 ["✅ All steps complete — passing to Observer."],
                                 feasible=True)

        step     = plan.steps[current_index]
        step_key = f"{step.step_type}_{step.step_id}"

        log_lines = [
            f"⚙️ **Step {current_index + 1}/{len(plan.steps)}:** `{step.step_type}`"
            + (f" (Day {step.day})" if step.day else "")
        ]

        ctx = dict(destination=destination, origin=origin, trip_days=trip_days,
                   budget=budget, prefs=prefs)

        # ── Anti-loop guard ────────────────────────────────────────────────
        prior_fails = _prior_failures(history, step.step_type)
        if prior_fails >= MAX_STEP_RETRIES:
            result = _wrap_result(
                status="fallback_used",
                data=None,
                error=f"`{step.step_type}` has failed {prior_fails} times previously.",
                replan_hint=(
                    f"`{step.step_type}` repeatedly failed. "
                    "Escalate to Fallback or remove from plan."
                ),
                trace=_minimal_trace(step.step_type, {},
                                     f"{prior_fails} prior failures — anti-loop guard.",
                                     "Refusing to retry."),
            )
            log_lines.append(f"🚫 Anti-loop: {prior_fails} prior failures — escalating to fallback.")
            return self._commit(step, step_key, result, current_index,
                                plan_state, results, history, log_lines)

        # ── Dispatch ───────────────────────────────────────────────────────
        if step.step_type in TOOL_REGISTRY:
            result = self._run_fetch_step(step, results, history, ctx)

        elif step.step_type == "fetch_activities":
            result = self._run_fetch_activities(step, results, history, ctx, trip_days)

        elif step.step_type == "build_day_schedule":
            current_plan_keys = {f"{s.step_type}_{s.step_id}" for s in plan.steps}
            result = self._run_build_day(step, results, destination, trip_days,
                                         current_plan_keys)

        elif step.step_type == "verify_budget":
            result = self._run_verify_budget(results, budget, trip_days)

        else:
            result = _wrap_result(
                status="failed",
                data=None,
                error=f"Unknown step_type: `{step.step_type}`",
                replan_hint=f"Remove or correct unknown step `{step.step_type}` from the plan.",
                trace=_minimal_trace(step.step_type, {}, "No handler registered.", ""),
            )

        return self._commit(step, step_key, result, current_index,
                            plan_state, results, history, log_lines)

    # ── Commit result and build state update ───────────────────────────────

    def _commit(
        self,
        step,
        step_key: str,
        result: dict,
        current_index: int,
        plan_state: dict,
        results: dict,
        history: list[dict],
        log_lines: list[str],
    ) -> dict:
        results[step_key] = result
        history.append(_history_entry(step, result))

        status = result.get("status", "unknown")
        icon   = {"success": "✅", "failed": "❌", "fallback_used": "🚫"}.get(status, "•")
        log_lines.append(f"{icon} **{status}**" + (f" — {result['error']}" if result.get("error") else ""))
        if result.get("replan_hint"):
            log_lines.append(f"💡 *Hint:* {result['replan_hint']}")

        feasible = status == "success"

        return _state_update(current_index, plan_state, results, history,
                             log_lines, feasible=feasible)

    # ── Fetch step handler (ReAct loop) ────────────────────────────────────

    def _run_fetch_step(self, step, results: dict, history: list[dict], ctx: dict) -> dict:
        tool_fn, arg_builder = TOOL_REGISTRY[step.step_type]
        args = _canonical_args(arg_builder(**ctx))

        # ── Cache hit ──
        cached = _find_cached_result(results, history, step.step_type, args)
        if cached is not None:
            return _wrap_result(
                status="success", data=cached, error=None, replan_hint="",
                trace=_minimal_trace(step.step_type, args,
                                     "Cache hit — reusing prior result.",
                                     "Valid cached data; no tool call needed."),
            )

        # ── ReAct: Thought ──
        thought_json = self._react_thought(step.step_type, args, ctx)

        # ── Action: call the tool ──
        try:
            raw = tool_fn.invoke(args)
            observation = json.dumps(raw, ensure_ascii=False)[:600]
        except Exception as exc:
            raw = None
            observation = f"Tool exception: {exc}"

        # ── Reflection: evaluate result ──
        reflection = self._react_reflect(step.step_type, args, raw, observation)
        status     = reflection.get("status", "failed" if _is_empty(raw) else "success")
        hint       = reflection.get("replan_hint", "")
        ref_text   = reflection.get("reflection", "")

        # Hard override — empty/None always fails regardless of LLM reflection
        if _is_empty(raw):
            status = "failed"
            if not hint:
                hint = (
                    f"`{step.step_type}` returned no data for "
                    f"origin='{ctx.get('origin')}' destination='{ctx.get('destination')}'. "
                    "Check spelling, verify DB has this route/city."
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
                "reflection":  ref_text,
            },
        )

    # ── fetch_activities handler (fetch + ActivitySelector) ────────────────

    def _run_fetch_activities(
        self, step, results: dict, history: list[dict], ctx: dict, trip_days: int
    ) -> dict:
        """
        Two-phase:
          1. Fetch raw activities from DB (same ReAct pattern as other fetches)
          2. Run ActivitySelector (LLM ranks + clusters by day)
        Both phases must succeed for status="success".
        ActivitySelector failure falls back to rating-sorted round-robin.
        """
        destination = ctx["destination"]
        prefs       = ctx["prefs"]
        dietary     = str(prefs.get("dietary_restrictions", "")).lower()

        args = _canonical_args({
            "city":                 destination,
            "kosher_only":          "kosher" in dietary,
            "vegetarian_friendly":  "vegetarian" in dietary,
            "vegan_friendly":       "vegan" in dietary,
        })

        # Cache check
        cached = _find_cached_result(results, history, "fetch_activities", args)
        if cached is not None:
            return _wrap_result(
                status="success", data=cached, error=None, replan_hint="",
                trace=_minimal_trace("fetch_activities", args,
                                     "Cache hit.", "Using cached activities + selection."),
            )

        # Phase 1 — Fetch
        thought_json = self._react_thought("fetch_activities", args, ctx)
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
                    "Verify city name spelling or check that the DB has activities for this city."
                ),
                trace={
                    "thought":     thought_json.get("thought", ""),
                    "action":      "fetch_activities",
                    "args":        args,
                    "observation": observation,
                    "reflection":  "Empty activity list — cannot build schedule.",
                },
            )

        # Phase 2 — ActivitySelector
        selector_thought = (
            f"Running ActivitySelector for {trip_days} days. "
            f"Applying geographic clustering and energy-curve ordering."
        )
        try:
            selection = select_activities_per_day(
                llm=self.llm,
                activities=raw,
                trip_days=trip_days,
                prefs=prefs,
                destination=destination,
            )
            selector_obs = (
                f"ActivitySelector assigned activities to {len(selection)} days: "
                + ", ".join(f"day_{d}: {len(v)} activities" for d, v in
                            enumerate(selection.values(), 1))
            )
        except Exception as exc:
            # Fallback: round-robin by rating — still success, degraded quality
            sorted_acts = sorted(raw, key=lambda a: -a.get("rating", 0))
            names = [a["name"] for a in sorted_acts]
            selection = {
                f"day_{d}": names[(d - 1) * 5: d * 5]
                for d in range(1, trip_days + 1)
            }
            selector_obs = f"ActivitySelector LLM failed ({exc}); used rating-sorted fallback."

        data = {"activities": raw, "selection": selection}
        return _wrap_result(
            status="success",
            data=data,
            error=None,
            replan_hint="",
            trace={
                "thought":     thought_json.get("thought", "") + " | " + selector_thought,
                "action":      "fetch_activities",
                "args":        args,
                "observation": observation + " | " + selector_obs,
                "reflection":  "Activities fetched and assigned to days.",
            },
        )

    # ── build_day_schedule handler (pure Python — no LLM thought) ──────────

    def _run_build_day(
        self, step, results: dict, destination: str, trip_days: int,
        current_plan_keys: set[str],
    ) -> dict:
        day_num = step.day or 1
        action  = f"build_day_schedule (Day {day_num})"

        # ── Prerequisite check ─────────────────────────────────────────────
        missing = _missing_prerequisites(results)
        if missing:
            return _wrap_result(
                status="failed",
                data=None,
                error=f"Day {day_num}: prerequisites missing: {missing}",
                replan_hint=(
                    f"Day {day_num} cannot be built: {missing} have no successful result yet. "
                    f"Re-include {missing} in the plan before build_day_schedule steps."
                ),
                trace=_minimal_trace(action, {"day": day_num},
                                     f"Prerequisites missing: {missing}",
                                     "Cannot build schedule without flight/hotel/activity data."),
            )

        # ── Resolve inputs ─────────────────────────────────────────────────
        outbound  = _cheapest_data(results, "fetch_flights")
        ret       = _cheapest_data(results, "fetch_return_flights")
        hotel_raw = _cheapest_data(results, "fetch_hotels")
        acts_data = _unwrap_data(results, "fetch_activities")

        all_activities: list[dict] = []
        selection: dict[str, list[str]] = {}
        if isinstance(acts_data, dict):
            all_activities = acts_data.get("activities", [])
            selection      = acts_data.get("selection", {})

        # Activities for this day (exclude already-scheduled in THIS plan run only —
        # stale build_day_schedule_* results from prior plan attempts use renumbered
        # step_ids and would otherwise pollute `used`, emptying day candidates).
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
                    "Re-run fetch_activities so ActivitySelector can reassign activities, "
                    "or increase activities pool for this destination."
                ),
                trace=_minimal_trace(action, {"day": day_num},
                                     "resolve_candidates returned [].",
                                     "No schedulable activities for this day."),
            )

        # ── Build DayConfig ────────────────────────────────────────────────
        hotel_name  = _safe_get(hotel_raw, "name", "Hotel")
        hotel_lat   = float(_safe_get(hotel_raw, "latitude") or _safe_get(hotel_raw, "lat") or 48.85)
        hotel_lng   = float(_safe_get(hotel_raw, "longitude") or _safe_get(hotel_raw, "lng") or 2.35)
        hotel_bk    = bool(_safe_get(hotel_raw, "breakfast_available", False))
        hotel_price = float(_safe_get(hotel_raw, "price_per_night", 0))

        arrival_raw    = _safe_get(outbound, "arrival_time",   "14:00")
        departure_raw  = _safe_get(ret,      "departure_time", "20:00")
        arrival_time   = _normalize_time(str(arrival_raw))
        departure_time = _normalize_time(str(departure_raw))

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

        # ── Run ScheduleEngine ─────────────────────────────────────────────
        try:
            builder = DayScheduleBuilder(cfg)
            slots   = builder.build(candidates)
        except Exception as exc:
            return _wrap_result(
                status="failed",
                data=None,
                error=f"ScheduleEngine exception on Day {day_num}: {exc}",
                replan_hint=(
                    f"ScheduleEngine crashed on Day {day_num}: {exc}. "
                    "Check activity opening/closing times and coordinates in the DB."
                ),
                trace=_minimal_trace(action, {"day": day_num},
                                     f"ScheduleEngine raised: {exc}", ""),
            )

        if not slots:
            return _wrap_result(
                status="failed",
                data=None,
                error=f"Day {day_num}: ScheduleEngine produced no slots.",
                replan_hint=(
                    f"Day {day_num}: ScheduleEngine returned empty schedule. "
                    "All candidates may be outside their operating hours on this day. "
                    "Try different activities or broaden the candidates pool."
                ),
                trace=_minimal_trace(action, {"day": day_num, "candidates": len(candidates)},
                                     "ScheduleEngine.build() returned [].",
                                     "No valid time window found for any candidate."),
            )

        # ── Assemble result ────────────────────────────────────────────────
        day_cost = round(sum(float(s.get("estimated_cost", 0)) for s in slots), 2)
        day_themes = {
            1:         f"Arrival & first impressions of {destination}",
            trip_days: f"Final day & farewell to {destination}",
        }
        theme = day_themes.get(day_num, f"Day {day_num} — Explore {destination}")

        data = {
            "day":                 day_num,
            "theme":               theme,
            "slots":               slots,
            "day_cost":            day_cost,
            "hotel":               hotel_name,
            "hotel_price_per_night": hotel_price,
        }

        return _wrap_result(
            status="success",
            data=data,
            error=None,
            replan_hint="",
            trace=_minimal_trace(
                action, {"day": day_num, "candidates": len(candidates)},
                f"Built {len(slots)} slots. Day cost: ${day_cost}.",
                f"Day {day_num} schedule complete.",
            ),
        )

    # ── verify_budget handler (pure Python) ────────────────────────────────

    def _run_verify_budget(self, results: dict, budget: float, trip_days: int) -> dict:
        # ── Prerequisite check ─────────────────────────────────────────────
        missing = _missing_prerequisites(results)
        if missing:
            return _wrap_result(
                status="failed",
                data=None,
                error=f"verify_budget prerequisites missing: {missing}",
                replan_hint=(
                    f"Cannot calculate budget: {missing} have no successful result. "
                    f"Re-include those fetch steps in the plan."
                ),
                trace=_minimal_trace("verify_budget", {"budget": budget},
                                     f"Missing: {missing}", ""),
            )

        outbound = _cheapest_data(results, "fetch_flights")
        ret      = _cheapest_data(results, "fetch_return_flights")
        hotel    = _cheapest_data(results, "fetch_hotels")

        # Aggregate actual costs from built day schedules
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

        try:
            data = calculate_trip_cost.invoke({
                "flight_price":                  float(_safe_get(outbound, "price", 0)),
                "return_flight_price":           float(_safe_get(ret,      "price", 0)),
                "hotel_price_per_night":         float(_safe_get(hotel,    "price_per_night", 0)),
                "trip_days":                     trip_days,
                "estimated_activities_budget":   activities_cost,
                "estimated_meals_budget_per_day": meal_per_day,
            })
        except Exception as exc:
            return _wrap_result(
                status="failed",
                data=None,
                error=f"calculate_trip_cost raised: {exc}",
                replan_hint=f"Budget tool failed: {exc}. Check flight/hotel prices are numeric.",
                trace=_minimal_trace("verify_budget", {}, str(exc), ""),
            )

        grand_total  = float(data.get("grand_total", 0))
        over_budget  = bool(budget) and grand_total > budget * 1.05
        observation  = (
            f"Grand total: ${grand_total:.0f} | Budget: ${budget or 'flexible'} | "
            + ("⚠️ OVER BUDGET" if over_budget else "✅ Within budget")
        )

        replan_hint = ""
        if over_budget:
            overage = grand_total - budget
            replan_hint = (
                f"Budget exceeded by ${overage:.0f}. "
                "Options: (1) cheaper hotel tier, "
                "(2) reduce trip_days by 1–2, "
                "(3) remove highest-cost activities from the selection."
            )

        return _wrap_result(
            status="success",       # budget check is informational — Observer decides action
            data=data,
            error=None,
            replan_hint=replan_hint,
            trace=_minimal_trace("verify_budget", {"trip_days": trip_days, "budget": budget},
                                 observation, "Over budget." if over_budget else "Within budget."),
        )

    # ── ReAct helpers (used only for fetch_* steps) ────────────────────────

    def _react_thought(self, step_type: str, args: dict, ctx: dict) -> dict:
        prompt = (
            f"Step: {step_type}\n"
            f"Args: {json.dumps(args, ensure_ascii=False)}\n"
            f"Context: destination={ctx.get('destination')}, origin={ctx.get('origin')}, "
            f"budget=${ctx.get('budget')}, days={ctx.get('trip_days')}"
        )
        try:
            raw = self.llm.invoke([
                SystemMessage(content=REACT_THOUGHT_SYSTEM),
                HumanMessage(content=prompt),
            ]).content.strip()
            raw = _strip_fences(raw)
            return json.loads(raw)
        except Exception:
            return {"thought": f"Executing {step_type}.", "replan_hint": ""}

    def _react_reflect(self, step_type: str, args: dict, result, observation: str) -> dict:
        prompt = (
            f"Step: {step_type}\n"
            f"Args: {json.dumps(args, ensure_ascii=False)}\n"
            f"Observation (truncated): {observation[:500]}"
        )
        try:
            raw = self.llm.invoke([
                SystemMessage(content=REACT_REFLECT_SYSTEM),
                HumanMessage(content=prompt),
            ]).content.strip()
            raw = _strip_fences(raw)
            return json.loads(raw)
        except Exception:
            empty = _is_empty(result)
            return {
                "status":      "failed" if empty else "success",
                "reflection":  "LLM reflection unavailable.",
                "replan_hint": f"`{step_type}` returned empty." if empty else "",
            }


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------

def _is_empty(val) -> bool:
    """True when val conveys no usable data."""
    if val is None:
        return True
    if isinstance(val, list):
        return len(val) == 0
    if isinstance(val, dict):
        return bool(val.get("error")) or not any(
            v for k, v in val.items() if k != "error"
        )
    return False


def _strip_fences(s: str) -> str:
    if s.startswith("```"):
        parts = s.split("```")
        s = parts[1] if len(parts) > 1 else s
        s = s.lstrip("json").strip()
    return s


def _canonical_args(args: dict) -> dict:
    return {k: v for k, v in sorted(args.items())}


def _prior_failures(history: list[dict], step_type: str) -> int:
    return sum(1 for h in history
               if h.get("step_type") == step_type and h.get("status") == "failed")


def _find_cached_result(
    results: dict, history: list[dict], step_type: str, args: dict
) -> Optional[dict]:
    """Return cached successful result for identical (step_type, args). None if not found."""
    never_cache = {"build_day_schedule", "verify_budget"}
    if step_type in never_cache:
        return None
    for h in history:
        if (h.get("step_type") == step_type
                and h.get("status") == "success"
                and h.get("args") == args):
            k = next(
                (k for k in results
                 if k.startswith(step_type)
                 and _inner_data(results[k]) is not None),
                None,
            )
            if k:
                return _inner_data(results[k])
    return None


def _inner_data(wrapped) -> Optional[dict]:
    """Extract the data payload from a v3 wrapped result or return the raw value."""
    if isinstance(wrapped, dict) and "status" in wrapped:
        if wrapped.get("status") == "success":
            return wrapped.get("data")
        return None   # failed result — not usable as cache
    # v2 raw result
    if isinstance(wrapped, dict) and not wrapped.get("error"):
        return wrapped
    if isinstance(wrapped, list):
        return wrapped  # type: ignore[return-value]
    return None


def _unwrap_data(results: dict, prefix: str) -> Optional[dict]:
    """Find the first successful result for a given prefix and return its data."""
    for key in results:
        if key.startswith(prefix):
            return _inner_data(results[key])
    return None


def _missing_prerequisites(results: dict) -> list[str]:
    """
    Return prerequisite step_types that have no successful result yet.
    Used by build_day_schedule and verify_budget to detect dependency failures.
    Only checks the truly blocking ones (flights + hotels + activities).
    """
    # fetch_weather is informational — not a hard prerequisite
    hard_prereqs = {"fetch_flights", "fetch_hotels", "fetch_activities"}
    missing = []
    for prereq in hard_prereqs:
        found = any(
            key.startswith(prereq) and _inner_data(results[key]) is not None
            for key in results
        )
        if not found:
            missing.append(prereq)
    return missing


def _cheapest_data(results: dict, prefix: str) -> Optional[dict]:
    """
    Return the cheapest item from a fetch result's data payload.
    Handles: list of flights, list of hotels, or single dict.
    """
    data = _unwrap_data(results, prefix)
    if data is None:
        return None

    # Hotels tool wraps in {"hotels": [...]}
    if isinstance(data, dict) and "hotels" in data:
        items = data["hotels"]
    elif isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        return data
    else:
        return None

    if not items:
        return None

    def _price(item: dict) -> float:
        if not isinstance(item, dict):
            return 9999.0
        if "price" in item:
            return float(item["price"])
        if "price_per_night" in item:
            return float(item["price_per_night"])
        return 9999.0

    priced = [i for i in items if isinstance(i, dict) and
              ("price" in i or "price_per_night" in i)]
    return min(priced, key=_price) if priced else (items[0] if items else None)


def _used_activities(results: dict, current_plan_keys: set[str]) -> set:
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


def _safe_get(d: Optional[dict], key: str, default=None):
    if not d or not isinstance(d, dict):
        return default
    return d.get(key) or default


def _normalize_time(raw: str) -> str:
    if not raw:
        return "12:00"
    if "T" in raw:
        raw = raw.split("T")[1]
    if " " in raw:
        raw = raw.split(" ")[1]
    return raw[:5]


def _minimal_trace(action: str, args: dict, observation: str, reflection: str) -> dict:
    return {
        "thought":     "",
        "action":      action,
        "args":        args,
        "observation": observation,
        "reflection":  reflection,
    }


def _wrap_result(
    status: str, data, error: Optional[str], replan_hint: str, trace: dict
) -> dict:
    return {
        "status":      status,
        "data":        data,
        "error":       error,
        "replan_hint": replan_hint,
        "trace":       trace,
    }


def _history_entry(step, result: dict) -> dict:
    return {
        "step_type":   step.step_type,
        "step_id":     step.step_id,
        "status":      result.get("status", "unknown"),
        "error":       result.get("error"),
        "replan_hint": result.get("replan_hint", ""),
        "args":        result.get("trace", {}).get("args", {}),
    }


def _state_update(
    current_index: int,
    plan_state: dict,
    results: dict,
    history: list[dict],
    log_lines: list[str],
    feasible: bool,
) -> dict:
    return {
        "current_step_index": current_index + 1,
        "itinerary_feasible": feasible,
        "itinerary_plan": {
            **plan_state,
            "step_results":      results,
            "execution_history": history,
        },
        "progress_log": ["\n".join(log_lines)],
    }
