# src/agent/nodes/itinerary/observer.py
"""
ItineraryObserverNode — v3
==========================
The Observer is the SINGLE SOURCE OF TRUTH for all state transitions.
It decides: continue | replan | fallback | complete

Key improvements over v2:

STRUCTURED REPLAN CONTEXT
  _trigger_replan() now writes a JSON object into observer_reason:
    {
      "error_code":       str,   # machine-readable category
      "error_message":    str,   # human-readable description
      "failed_step":      str,   # step_type that caused the failure
      "replan_hint":      str,   # specific corrective suggestion for Planner
      "completed_steps":  list,  # step_types with clean cached results
      "system_state":     dict,  # budget, trip_days, destination, origin snapshot
    }
  The Planner (v3) parses this JSON to generate a minimal corrective plan.

RETRY vs REPLAN DISTINCTION
  retry_count  — how many times the Observer triggered a replan cycle
  replan_count — how many full plans the Planner has generated
  Both tracked independently; hard-stop triggers on either.

CRITICAL-TOOL EMPTY RESULT DETECTION (mid-execution)
  Empty results from fetch_flights / fetch_hotels are caught immediately,
  not after building the whole schedule. This prevents wasted build_day steps.

ERROR CLASSIFICATION
  Every failure is classified into an error_code:
    STEP_ERROR          — tool raised an exception
    NO_FLIGHTS          — flight search returned empty list
    NO_HOTELS           — hotel search returned empty list
    EMPTY_DAY           — build_day_schedule produced no slots
    OVERLAP             — time overlap between slots
    BUDGET_EXCEEDED     — grand total > budget
    MISSING_BREAKFAST   — no morning food slot
    MISSING_LUNCH       — no midday food slot
    MISSING_DINNER      — no evening food slot
    TRANSPORT_GAP       — far-apart activities with no transport slot
    LLM_REJECT          — LLM quality reviewer explicitly rejected the plan
    MAX_RETRIES         — hard stop, forward to Fallback

SOFT WARNINGS
  MISSING_BREAKFAST / MISSING_LUNCH / MISSING_DINNER are treated as soft
  warnings on the first pass. They are logged but do not trigger replan
  unless ALL three are missing (implies a data problem).

OBSERVER_ACTION
  Set on every return path so edge router always has a valid signal:
    "continue"  — mid-execution, next step
    "replan"    — failure, go back to Planner
    "fallback"  — hard stop, go to Fallback
    "complete"  — all steps done, markdown generated
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from agent.nodes.itinerary.schemas import ExecutionPlan
from agent.nodes.itinerary.schedule_engine import haversine_km, GeoPoint, WALK_MAX_KM
from agent.state import AgentState

MAX_RETRIES = 3   # replan cycles before hard-stop to Fallback

# ── Critical fetch steps: empty result = immediate replan ─────────────────
CRITICAL_FETCH_STEPS = {"fetch_flights", "fetch_return_flights", "fetch_hotels"}

# ── Hard validation error codes: always trigger replan ────────────────────
HARD_ERROR_CODES = {"OVERLAP", "ZERO_DURATION", "EMPTY_DAY", "BUDGET_EXCEEDED", "NO_FLIGHTS", "NO_HOTELS"}

# ── Meal error codes: soft unless ALL three missing ────────────────────────
MEAL_CODES = {"MISSING_BREAKFAST", "MISSING_LUNCH", "MISSING_DINNER"}

# ── LLM prompt: final quality review + markdown generation ────────────────

OBSERVER_SYSTEM = """
You are the final travel itinerary quality reviewer AND copywriter.

You receive a structured trip plan summary. Your tasks:

1. Check for SEVERE structural problems only:
   - A day has fewer than 2 activities
   - No food venue is present across the entire trip
   - The schedule is logically impossible (e.g. dinner at 07:00)

2. If you find a SEVERE problem, reply ONLY with a JSON object:
   {"reject": true, "reason": "<one-sentence explanation>"}

3. If the plan is acceptable, output ONLY the final markdown itinerary
   using the format below. Do NOT add JSON around it.

MARKDOWN FORMAT:
# ✈️ Your [N]-Day [Destination] Itinerary

## 🛫 Flight Details
**Outbound:** [Airline] [Flight#] | Dep: [time] → Arr: [time]
**Return:** [Airline] [Flight#] | Dep: [time] → Arr: [time]
*(For connecting flights, list each leg separately)*

## 🏨 Accommodation
**Hotel:** [Name] — [stars]★ | $[price]/night

## 📅 Day 1 — [Theme]
| Time | Activity | Duration | Est. Cost |
|------|----------|----------|-----------|
| HH:MM | [icon] [Name] | X min | $Y |
...
**Day total: $XX**

[Repeat for each day]

---
## 💰 Trip Budget Summary
| Category | Cost |
|----------|------|
| Outbound flight | $X |
| Return flight | $X |
| Hotel ([N] nights) | $X |
| Activities | $X |
| Meals | $X |
| **Grand Total** | **$X** |

✅ Within budget  /  ⚠️ Slightly over budget

---
*💡 Tips and local recommendations here.*
"""


# ---------------------------------------------------------------------------
# ValidationError — lightweight data class
# ---------------------------------------------------------------------------

class ValidationError:
    def __init__(self, code: str, message: str, day: Optional[int] = None,
                 replan_hint: str = ""):
        self.code = code
        self.message = message
        self.day = day
        self.replan_hint = replan_hint  # specific suggestion for the Planner

    def __str__(self):
        prefix = f"Day {self.day}: " if self.day else ""
        return f"[{self.code}] {prefix}{self.message}"

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "day": self.day,
            "replan_hint": self.replan_hint,
        }


# ---------------------------------------------------------------------------
# Result unwrapper — handles both v2 (raw dict) and v3 (wrapped) formats
# ---------------------------------------------------------------------------

def _unwrap(val: dict) -> dict:
    """
    v3 executor wraps every result as {"status": ..., "data": {...}, ...}.
    v2 stored the raw tool output directly.
    This helper always returns the inner data dict, or {} on failure.
    """
    if not isinstance(val, dict):
        return {}
    if "data" in val and isinstance(val["data"], dict):
        return val["data"]
    return val


# ---------------------------------------------------------------------------
# Layer 1 — Pure Python Validators
# ---------------------------------------------------------------------------

def validate_schedule(results: dict, budget: float, trip_days: int) -> list[ValidationError]:
    """
    Run all structural checks. Returns list of ValidationError objects.
    Empty list = structurally valid.
    """
    errors: list[ValidationError] = []

    for key, val in results.items():
        if not key.startswith("build_day_schedule") or not isinstance(val, dict):
            continue
        day_data = _unwrap(val)
        day = day_data.get("day", "?")
        slots = day_data.get("slots", [])

        if not slots:
            errors.append(ValidationError(
                "EMPTY_DAY",
                "No slots generated for this day",
                day,
                replan_hint=(
                    f"Day {day} produced no slots. Check that fetch_activities returned "
                    "results and the ActivitySelector assigned activities to this day."
                ),
            ))
            continue

        errors.extend(_check_overlaps(slots, day))
        errors.extend(_check_meals(slots, day))
        errors.extend(_check_transport_gaps(slots, day))

    budget_error = _check_budget(results, budget, trip_days)
    if budget_error:
        errors.append(budget_error)

    return errors


def _parse_hm(s: str) -> datetime:
    return datetime.strptime(s[:5], "%H:%M")


def _check_overlaps(slots: list[dict], day) -> list[ValidationError]:
    errors = []
    prev_end: Optional[datetime] = None
    for slot in slots:
        start_str = slot.get("time") or slot.get("start_time")
        end_str = slot.get("end_time")
        if not start_str or not end_str:
            continue
        try:
            start = _parse_hm(start_str)
            end = _parse_hm(end_str)
        except ValueError:
            continue

        if prev_end and start < prev_end:
            errors.append(ValidationError(
                "OVERLAP",
                f"'{slot.get('name')}' starts at {start_str} but previous slot ends at "
                f"{prev_end.strftime('%H:%M')}",
                day,
                replan_hint=(
                    f"Day {day} has overlapping slots. The ScheduleEngine may have received "
                    "activities with incorrect opening/closing times, or too many candidates."
                ),
            ))
        if end <= start:
            errors.append(ValidationError(
                "ZERO_DURATION",
                f"'{slot.get('name')}' has zero or negative duration",
                day,
                replan_hint=f"Day {day} slot '{slot.get('name')}' has invalid time range.",
            ))
        prev_end = end
    return errors


def _check_meals(slots: list[dict], day) -> list[ValidationError]:
    """Check for breakfast, midday food, and dinner slots."""
    errors = []
    by_hour: list[tuple[int, bool]] = []

    for s in slots:
        t = s.get("time") or s.get("start_time", "")
        if not t:
            continue
        try:
            h = int(t.split(":")[0])
        except ValueError:
            continue
        is_food = s.get("slot_type") == "meal" or bool(s.get("food_available"))
        by_hour.append((h, is_food))

    has_breakfast = any(h <= 10 and f for h, f in by_hour)
    has_lunch     = any(11 <= h <= 15 and f for h, f in by_hour)
    has_dinner    = any(h >= 18 and f for h, f in by_hour)

    if not has_breakfast:
        errors.append(ValidationError(
            "MISSING_BREAKFAST",
            "No morning food slot (before 10:00)",
            day,
            replan_hint=(
                f"Day {day} is missing breakfast. Ensure ActivitySelector includes a "
                "café or breakfast venue in the morning block."
            ),
        ))
    if not has_lunch:
        errors.append(ValidationError(
            "MISSING_LUNCH",
            "No midday food slot (11:00–15:00)",
            day,
            replan_hint=(
                f"Day {day} is missing lunch. Ensure a restaurant or food tour is "
                "assigned to the 11:00–15:00 window."
            ),
        ))
    if not has_dinner:
        errors.append(ValidationError(
            "MISSING_DINNER",
            "No evening food slot (after 18:00)",
            day,
            replan_hint=(
                f"Day {day} is missing dinner. Ensure a dinner venue is in the evening block."
            ),
        ))
    return errors


def _check_transport_gaps(slots: list[dict], day) -> list[ValidationError]:
    """
    Flag cases where two activity slots are > WALK_MAX_KM apart
    with no transport slot between them.
    """
    errors = []
    prev_act: Optional[dict] = None
    prev_idx = -1
    has_transport_between = False

    for i, slot in enumerate(slots):
        stype = slot.get("slot_type")

        if stype == "transport":
            has_transport_between = True
            continue

        if stype != "activity":
            continue

        if prev_act is not None:
            lat1 = prev_act.get("lat") or prev_act.get("latitude")
            lng1 = prev_act.get("lng") or prev_act.get("longitude")
            lat2 = slot.get("lat") or slot.get("latitude")
            lng2 = slot.get("lng") or slot.get("longitude")

            if all(v is not None for v in (lat1, lng1, lat2, lng2)):
                try:
                    dist = haversine_km(
                        GeoPoint(float(lat1), float(lng1)),
                        GeoPoint(float(lat2), float(lng2)),
                    )
                    if dist > WALK_MAX_KM and not has_transport_between:
                        errors.append(ValidationError(
                            "TRANSPORT_GAP",
                            f"'{prev_act.get('name')}' → '{slot.get('name')}' is "
                            f"{dist:.1f} km apart with no transport slot",
                            day,
                            replan_hint=(
                                f"Day {day}: add a transport slot between "
                                f"'{prev_act.get('name')}' and '{slot.get('name')}'. "
                                "These are too far apart to walk."
                            ),
                        ))
                except (TypeError, ValueError):
                    pass

        prev_act = slot
        prev_idx = i
        has_transport_between = False

    return errors


def _check_budget(results: dict, budget: float, trip_days: int) -> Optional[ValidationError]:
    """Check verify_budget result against the declared budget."""
    if not budget:
        return None  # budget is flexible — skip

    budget_key = next((k for k in results if k.startswith("verify_budget")), None)
    if not budget_key:
        return None

    val = _unwrap(results[budget_key])
    if not isinstance(val, dict):
        return None

    grand_total = float(val.get("grand_total", 0))
    if grand_total > budget * 1.05:  # 5% tolerance
        overage = grand_total - budget
        pct = (overage / budget) * 100
        return ValidationError(
            "BUDGET_EXCEEDED",
            f"Grand total ${grand_total:.0f} exceeds budget ${budget:.0f} "
            f"(over by ${overage:.0f} / {pct:.0f}%)",
            replan_hint=(
                f"Total cost ${grand_total:.0f} exceeds budget ${budget:.0f}. "
                f"Consider: (1) cheaper hotel tier, (2) reducing trip_days by 1-2, "
                f"(3) removing high-cost activities. Overage: ${overage:.0f}."
            ),
        )
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _completed_step_types(step_results: dict) -> list[str]:
    """Return step_types that have a successful (non-error) cached result.
    Handles both v2 (raw dict) and v3 (wrapped {"status":...}) formats.
    """
    completed = []
    for key, val in step_results.items():
        if not isinstance(val, dict):
            continue
        # v3 structured result: check status field
        if "status" in val:
            if val["status"] != "success":
                continue
        else:
            # v2 raw result: skip if it has an error key
            if "error" in val and val["error"]:
                continue
        # key format: "{step_type}_{step_id}" — strip trailing _N
        parts = key.rsplit("_", 1)
        step_type = parts[0] if len(parts) == 2 and parts[1].isdigit() else key
        if step_type and step_type not in completed:
            completed.append(step_type)
    return completed


def _build_replan_context(
    error_code: str,
    error_message: str,
    failed_step: str,
    replan_hint: str,
    step_results: dict,
    state: AgentState,
) -> str:
    """Build the structured JSON string that the Planner (v3) will parse."""
    ctx = {
        "error_code": error_code,
        "error_message": error_message,
        "failed_step": failed_step,
        "replan_hint": replan_hint,
        "completed_steps": _completed_step_types(step_results),
        "system_state": {
            "budget": state.get("total_budget", 0),
            "trip_days": state.get("trip_days", 3),
            "destination": state.get("destination_city", ""),
            "origin": state.get("current_city", ""),
        },
    }
    return json.dumps(ctx, ensure_ascii=False)


def _build_summary(results: dict, budget: float, trip_days: int) -> str:
    lines = [f"Trip: {trip_days} days | Budget: ${budget}"]
    for key, val in results.items():
        if not isinstance(val, dict):
            continue
        # Skip failed results
        if val.get("status") == "failed" or (val.get("error") and not val.get("data")):
            continue
        inner = _unwrap(val)
        # Avoid dumping the full flight list — just show first entry
        display_val = inner
        if key.startswith(("fetch_flights", "fetch_return_flights")):
            if isinstance(inner, list) and inner:
                display_val = inner[0]
            elif isinstance(inner, dict) and inner.get("activities"):
                display_val = {"note": "activities omitted for brevity"}
        lines.append(f"--- {key.upper()} ---")
        lines.append(json.dumps(display_val, ensure_ascii=False))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class ItineraryObserverNode:
    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm

    def __call__(self, state: AgentState) -> dict:
        plan_state  = state.get("itinerary_plan", {})
        results     = plan_state.get("step_results", {})
        retry_count = plan_state.get("retry_count", 0)
        replan_count = plan_state.get("replan_count", 0)
        current_index = state.get("current_step_index", 0)
        plan_steps  = plan_state.get("execution_plan", {}).get("steps", [])

        # ── 1. Mid-execution: validate the step that just ran ──────────────
        if current_index > 0 and current_index <= len(plan_steps):
            last_step = plan_steps[current_index - 1]
            step_type = last_step.get("step_type", "")
            step_id   = last_step.get("step_id", 0)
            last_key  = f"{step_type}_{step_id}"
            last_result = results.get(last_key, {})

            # 1a. Explicit tool error
            if isinstance(last_result, dict) and last_result.get("error"):
                return self._trigger_replan(
                    plan_state=plan_state,
                    retry_count=retry_count,
                    replan_count=replan_count,
                    error_code="STEP_ERROR",
                    error_message=last_result["error"],
                    failed_step=step_type,
                    replan_hint=f"Step `{step_type}` failed with error: {last_result['error']}. Retry it or use a simpler variant.",
                    results=results,
                    state=state,
                )

            # 1b. Critical fetch returned empty
            if step_type in CRITICAL_FETCH_STEPS:
                is_empty = (
                    (isinstance(last_result, list) and len(last_result) == 0)
                    or (isinstance(last_result, dict) and not last_result.get("error")
                        and not any(last_result.values()))
                )
                if is_empty:
                    error_code = {
                        "fetch_flights":        "NO_FLIGHTS",
                        "fetch_return_flights": "NO_FLIGHTS",
                        "fetch_hotels":         "NO_HOTELS",
                    }.get(step_type, "EMPTY_RESULT")

                    hints = {
                        "NO_FLIGHTS": (
                            f"No flights found for this route. Consider: (1) the destination "
                            f"may not be in the DB, (2) check spelling of origin/destination."
                        ),
                        "NO_HOTELS": (
                            f"No hotels found at destination. Consider relaxing kosher/star filters."
                        ),
                    }
                    return self._trigger_replan(
                        plan_state=plan_state,
                        retry_count=retry_count,
                        replan_count=replan_count,
                        error_code=error_code,
                        error_message=f"`{step_type}` returned no results",
                        failed_step=step_type,
                        replan_hint=hints.get(error_code, f"`{step_type}` returned empty."),
                        results=results,
                        state=state,
                    )

            # ── No-flights early exit: skip replanning entirely ──
            # If fetch_flights returned an empty list, there are no flights
            # to this destination at all. Replanning won't help — route
            # directly to the alternative_destination node.
            if last_step["step_type"] in ("fetch_flights", "fetch_return_flights"):
                is_empty = (
                    (isinstance(last_result, list) and len(last_result) == 0)
                    or (isinstance(last_result, dict) and not last_result)
                    or last_result is None
                )
                if is_empty:
                    flight_type = "outbound" if last_step["step_type"] == "fetch_flights" else "return"
                    return {
                        "itinerary_feasible": False,
                        "itinerary_fallback_reason": f"no_{flight_type}_flights",
                        "observer_action": "no_flights",
                        "messages": [AIMessage(
                            content=f"✈️ **No {flight_type} flights found** — routing to alternative destinations.",
                            name="observer_log",
                        )],
                    }

        # ── 2. Not done yet: continue ──
        if current_index < len(plan_steps):
            return {
                "itinerary_feasible": True,
                "observer_action": "continue",
            }

        # ── 3. All steps done — Layer 1 structural validation ─────────────
        budget    = state.get("total_budget", 0)
        trip_days = state.get("trip_days", 3)

        all_errors = validate_schedule(results, budget, trip_days)

        hard_errors = [e for e in all_errors if e.code in HARD_ERROR_CODES]
        meal_errors = [e for e in all_errors if e.code in MEAL_CODES]
        soft_warnings = [e for e in all_errors
                         if e.code not in HARD_ERROR_CODES and e.code not in MEAL_CODES]

        # Meal errors: only replan if ALL three are missing (implies data issue)
        if len(meal_errors) == 3 and not hard_errors:
            hard_errors = meal_errors  # treat as hard

        if hard_errors:
            # Pick the most critical error to lead the replan context
            primary = hard_errors[0]
            extra_hints = " | ".join(e.replan_hint for e in hard_errors[1:] if e.replan_hint)
            combined_hint = primary.replan_hint
            if extra_hints:
                combined_hint += f" Additional issues: {extra_hints}"

            return self._trigger_replan(
                plan_state=plan_state,
                retry_count=retry_count,
                replan_count=replan_count,
                error_code=primary.code,
                error_message="; ".join(str(e) for e in hard_errors),
                failed_step=primary.failed_step_type_hint(results),
                replan_hint=combined_hint,
                results=results,
                state=state,
            )

        # Log soft warnings (meal gaps that didn't trigger replan)
        warn_msg = ""
        if meal_errors or soft_warnings:
            lines = ["⚠️ **Soft warnings (non-blocking):**"]
            for w in (meal_errors + soft_warnings):
                lines.append(f"  • `{w.code}` {w}")
            warn_msg = "\n".join(lines)

        # ── 4. Layer 2 — LLM quality review + markdown generation ──────────
        try:
            summary = _build_summary(results, budget, trip_days)
            output = self.llm.invoke([
                SystemMessage(content=OBSERVER_SYSTEM),
                HumanMessage(content=summary),
            ])
            content = output.content.strip()
        except Exception as e:
            content = ""

        # Check for structured JSON rejection
        if content.startswith("{"):
            try:
                rejection = json.loads(content)
                if rejection.get("reject"):
                    reason = rejection.get("reason", "LLM quality check failed")
                    return self._trigger_replan(
                        plan_state=plan_state,
                        retry_count=retry_count,
                        replan_count=replan_count,
                        error_code="LLM_REJECT",
                        error_message=reason,
                        failed_step="llm_review",
                        replan_hint=(
                            f"LLM quality reviewer rejected the plan: {reason}. "
                            "Ensure each day has at least 2 activities and at least "
                            "1 food venue."
                        ),
                        results=results,
                        state=state,
                    )
            except json.JSONDecodeError:
                pass  # not a rejection — treat as markdown

        # Legacy plain-text REJECT: support for backwards compatibility
        if content.startswith("REJECT:"):
            reason = content.replace("REJECT:", "").strip()
            return self._trigger_replan(
                plan_state=plan_state,
                retry_count=retry_count,
                replan_count=replan_count,
                error_code="LLM_REJECT",
                error_message=reason,
                failed_step="llm_review",
                replan_hint=f"LLM rejected: {reason}. Add more activities/food venues.",
                results=results,
                state=state,
            )

        # ── 5. Success ─────────────────────────────────────────────────────
        if content and not content.startswith("{"):
            if content.startswith("```"):
                markdown = content.split("```")[1].lstrip("markdown").strip().rstrip("```").strip()
            else:
                markdown = content
        else:
            markdown = _generate_fallback_markdown(results, trip_days, budget)

        success_msg = f"✅ **Itinerary complete.**"
        if warn_msg:
            success_msg += f"\n\n{warn_msg}"

        return {
            "itinerary_plan": {**plan_state, "final_markdown": markdown},
            "itinerary_feasible": True,
            "observer_action": "complete",
            "messages": [
                AIMessage(content=success_msg, name="observer_log"),
            ],
        }

    # ── Replan trigger ─────────────────────────────────────────────────────

    def _trigger_replan(
        self,
        plan_state: dict,
        retry_count: int,
        replan_count: int,
        error_code: str,
        error_message: str,
        failed_step: str,
        replan_hint: str,
        results: dict,
        state: AgentState,
    ) -> dict:
        new_retry = retry_count + 1

        observer_reason_json = _build_replan_context(
            error_code=error_code,
            error_message=error_message,
            failed_step=failed_step,
            replan_hint=replan_hint,
            step_results=results,
            state=state,
        )

        # Hard stop
        if new_retry >= MAX_RETRIES:
            hard_reason = json.dumps({
                "error_code": "MAX_RETRIES",
                "error_message": f"max_retries_exceeded after {new_retry} attempts. Last error: {error_message}",
                "failed_step": failed_step,
                "replan_hint": replan_hint,
                "completed_steps": _completed_step_types(results),
                "system_state": {
                    "budget": state.get("total_budget", 0),
                    "trip_days": state.get("trip_days", 3),
                    "destination": state.get("destination_city", ""),
                    "origin": state.get("current_city", ""),
                },
            }, ensure_ascii=False)

            return {
                "itinerary_feasible": False,
                "observer_action": "fallback",
                "itinerary_fallback_reason": hard_reason,
                "itinerary_plan": {
                    **plan_state,
                    "retry_count": new_retry,
                    "replan_count": replan_count,
                    "observer_reason": hard_reason,
                },
                "messages": [AIMessage(
                    content=(
                        f"❌ **OBSERVER → FALLBACK** "
                        f"(max retries={MAX_RETRIES} reached)\n"
                        f"*Error:* `{error_code}` — {error_message}"
                    ),
                    name="observer_log",
                )],
            }

        # Soft replan
        return {
            "itinerary_feasible": False,
            "observer_action": "replan",
            "itinerary_fallback_reason": error_message,
            "itinerary_plan": {
                **plan_state,
                "retry_count": new_retry,
                "replan_count": replan_count,
                "observer_reason": observer_reason_json,
            },
            "messages": [AIMessage(
                content=(
                    f"🔄 **OBSERVER → REPLAN** "
                    f"(attempt {new_retry}/{MAX_RETRIES})\n"
                    f"*Error code:* `{error_code}`\n"
                    f"*Message:* {error_message}\n"
                    f"*Hint:* {replan_hint}"
                ),
                name="observer_log",
            )],
        }


# ---------------------------------------------------------------------------
# ValidationError helper — resolve which step_type produced the error
# ---------------------------------------------------------------------------

def _step_type_from_code(code: str) -> str:
    mapping = {
        "EMPTY_DAY":        "build_day_schedule",
        "OVERLAP":          "build_day_schedule",
        "ZERO_DURATION":    "build_day_schedule",
        "TRANSPORT_GAP":    "build_day_schedule",
        "MISSING_BREAKFAST": "build_day_schedule",
        "MISSING_LUNCH":    "build_day_schedule",
        "MISSING_DINNER":   "build_day_schedule",
        "BUDGET_EXCEEDED":  "verify_budget",
        "NO_FLIGHTS":       "fetch_flights",
        "NO_HOTELS":        "fetch_hotels",
        "LLM_REJECT":       "llm_review",
    }
    return mapping.get(code, "unknown")


# Monkey-patch a helper onto ValidationError so we can call it above
def _failed_step_hint(self, results: dict) -> str:
    return _step_type_from_code(self.code)

ValidationError.failed_step_type_hint = _failed_step_hint


# ---------------------------------------------------------------------------
# Fallback markdown generator (no LLM)
# ---------------------------------------------------------------------------

def _generate_fallback_markdown(results: dict, trip_days: int, budget: float) -> str:
    lines = ["# ✈️ Your Trip Itinerary\n"]

    out_key = next((k for k in results if k.startswith("fetch_flights")), None)
    ret_key = next((k for k in results if k.startswith("fetch_return_flights")), None)

    def _first_flight(key):
        if not key:
            return {}
        inner = _unwrap(results[key])
        if isinstance(inner, list) and inner:
            return inner[0]
        if isinstance(inner, dict):
            return inner
        return {}

    outbound = _first_flight(out_key)
    return_fl = _first_flight(ret_key)

    if outbound or return_fl:
        lines.append("## 🛫 Flight Details")
        for title, flight in [("Outbound", outbound), ("Return", return_fl)]:
            if not flight:
                continue
            if "route" in flight:
                lines.append(f"**{title} (Connecting):**")
                for leg in flight.get("route", []):
                    lines.append(
                        f"  - {leg.get('airline')} {leg.get('flight')} | "
                        f"{leg.get('from')} → {leg.get('to')} | "
                        f"Dep: {leg.get('departure_time')} Arr: {leg.get('arrival_time')}"
                    )
            else:
                lines.append(
                    f"**{title} (Direct):** {flight.get('airline')} "
                    f"{flight.get('flight_number')} | "
                    f"Dep: {flight.get('departure_time')} → Arr: {flight.get('arrival_time')}"
                )
        lines.append("")

    hotel_key = next((k for k in results if k.startswith("fetch_hotels")), None)
    if hotel_key:
        hotel_inner = _unwrap(results[hotel_key])
        if isinstance(hotel_inner, list) and hotel_inner:
            hotel_inner = hotel_inner[0]
        if isinstance(hotel_inner, dict) and hotel_inner.get("name"):
            lines.append("## 🏨 Accommodation")
            lines.append(
                f"**{hotel_inner.get('name', 'Hotel')}** — "
                f"{hotel_inner.get('stars', '?')}★ | "
                f"${hotel_inner.get('price_per_night', '?')}/night\n"
            )

    for d in range(1, trip_days + 1):
        key = next(
            (k for k in results if k.startswith("build_day_schedule")),
            None,
        )
        # Find the key whose unwrapped data has day == d
        key = next(
            (k for k in results
             if k.startswith("build_day_schedule")
             and isinstance(_unwrap(results[k]), dict)
             and _unwrap(results[k]).get("day") == d),
            None,
        )
        if not key:
            continue
        day_data = _unwrap(results[key])
        lines.append(f"\n## 📅 Day {d} — {day_data.get('theme', '')}")
        lines.append("\n| Time | Activity | Duration | Est. Cost |")
        lines.append("|------|----------|----------|-----------|")
        for slot in day_data.get("slots", []):
            icon = {
                "activity": "🎯", "meal": "🍽️", "transport": "🚕",
                "rest": "😴", "checkin": "🏨",
            }.get(slot.get("slot_type", ""), "•")
            lines.append(
                f"| {slot.get('time', '')} | {icon} {slot.get('name', '')} | "
                f"{slot.get('duration_minutes', '')} min | ${slot.get('estimated_cost', 0):.0f} |"
            )
        lines.append(f"\n**Day total: ${day_data.get('day_cost', 0):.0f}**")

    budget_key = next((k for k in results if k.startswith("verify_budget")), None)
    if budget_key:
        b = _unwrap(results[budget_key])
        if isinstance(b, dict) and b.get("grand_total") is not None:
            lines.append("\n---\n## 💰 Budget Summary\n")
            lines.append("| Category | Cost |")
            lines.append("|----------|------|")
            for cat, val in b.items():
                if cat != "grand_total":
                    lines.append(f"| {cat.replace('_', ' ').title()} | ${float(val):.0f} |")
            grand = b.get("grand_total", 0)
            lines.append(f"\n**Grand Total: ${float(grand):.0f}**")
            if budget:
                remaining = budget - float(grand)
                emoji = "✅" if remaining >= 0 else "⚠️"
                lines.append(f"\n{emoji} Budget remaining: ${remaining:.0f}")

    return "\n".join(lines)