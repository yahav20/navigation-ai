"""
ItineraryPlannerNode
====================
Generates a minimal ordered execution plan for the schedule builder.
All plan generation is deterministic — no LLM calls except in update mode.

Three modes (set by PlanCheckNode / edge.py / prior planner call):
  with_travel_data — state["travel_plan"] already has flights + hotel.
  standalone       — no booking data. Steps include fetch_avg_prices.
  update           — surgical re-plan for a single-turn itinerary change.

Update mode (first entry):
  Makes one LLM call to extract a DayUpdateInstruction from the user's
  message, resolves the target day deterministically, then emits a single
  update_day_schedule or apply_global_preference step.
  Subsequent replans in update mode are fully deterministic (re-reads the
  already-extracted itinerary_update_request from state).

Replan flow:
  On the first call, replan_context is empty → fresh plan.
  On subsequent calls (from Replanner), replan_context contains a JSON blob
  describing what failed and which steps already succeeded. The planner emits
  only the remaining steps, derived from completed_steps in state/context.

Safety:
  MAX_REPLANS hard-stops the Planner and hands off to the Formatter with an
  error message when the Replanner has exhausted retries.
"""
from __future__ import annotations

import json
import math
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.core.llm import silent
from agent.itinerary.schemas import DayUpdateInstruction, ExecutionPlan, PlanStep
from agent.core.state import AgentState

# ── Safety limits ──────────────────────────────────────────────────────────
MAX_REPLANS = 3

# ── Valid step types ───────────────────────────────────────────────────────
VALID_STEP_TYPES = {
    "fetch_activities",
    "fetch_weather",
    "fetch_avg_prices",
    "fetch_min_prices",
    "switch_travel_options",
    "build_day_schedule",
    "verify_budget",
    "update_day_schedule",
    "apply_global_preference",
}

# Steps that must complete before any build_day_schedule can run
PREREQUISITE_STEPS = {"fetch_activities"}

# ── Update extraction prompt ───────────────────────────────────────────────
_EXTRACTION_PROMPT = """Extract a structured update instruction from the user's message about their itinerary.

update_type — one of:
  exclude_activity    user wants to remove an activity ("I don't want the Eiffel Tower")
  replace_activity    user wants to swap one for another ("replace Eiffel Tower with PSG Museum")
  add_activity        user wants to add a new activity ("add PSG Museum to one of the days")
  exclude_restaurant  user wants to remove a restaurant or meal slot
  replace_restaurant  user wants to swap a restaurant
  global_preference   structural change across all days ("no breakfast", "remove all coffees")

scope — one of:
  single_day   user named a specific day ("on day 3", "from day 2")
  any_day      user doesn't care which day ("somewhere", "to one of the days")
  all_days     applies to every day ("remove breakfast from all days", "no morning coffee")

day      integer day number when scope is single_day, null otherwise
target_name     what to remove/replace (verbatim or close to it), null if not applicable
replacement_name  what to add in its place or just add, null if not specified
preference_hint  extra context such as "I like football", "something cheaper", "outdoor"
excluded_slot_types  slot type strings to remove from ALL days.
  Use ["breakfast"] for "no breakfast", ["meal"] for all meals, [] otherwise.
  Populate only for global_preference.
"""


# ---------------------------------------------------------------------------
# Helpers — deterministic planning
# ---------------------------------------------------------------------------

def _completed_step_types(step_results: dict) -> list[str]:
    """Return step_types that have a successful result cached."""
    completed: list[str] = []
    for key, val in step_results.items():
        if not isinstance(val, dict):
            continue
        if val.get("status") != "success":
            continue
        parts = key.rsplit("_", 1)
        step_type = parts[0] if len(parts) == 2 and parts[1].isdigit() else key
        if step_type and step_type not in completed:
            completed.append(step_type)
    return completed


def _completed_days(step_results: dict) -> set[int]:
    """Return day numbers that have been successfully built."""
    days: set[int] = set()
    for key, val in step_results.items():
        if not key.startswith("build_day_schedule"):
            continue
        if not isinstance(val, dict) or val.get("status") != "success":
            continue
        inner = val.get("data", {})
        if isinstance(inner, dict) and isinstance(inner.get("day"), int):
            days.add(inner["day"])
    return days


def _parse_replan_context(raw: str) -> Optional[dict]:
    """Parse JSON replan context written by the Replanner."""
    if not raw:
        return None
    try:
        ctx = json.loads(raw)
        if isinstance(ctx, dict):
            return ctx
    except (json.JSONDecodeError, TypeError):
        pass
    return {
        "error_code":      "unknown",
        "error_message":   raw,
        "failed_step":     None,
        "replan_hint":     raw,
        "completed_steps": [],
    }


def _validate_and_fix(
    plan: ExecutionPlan,
    completed: list[str],
    trip_days: int,
    destination: str,
    completed_days: set[int],
    mode: str = "standalone",
    need_min_prices: bool = False,
    need_switch_travel: bool = False,
) -> ExecutionPlan:
    """Enforce hard ordering constraints and fill missing day steps."""
    steps = plan.steps

    # Strip any verify_budget steps — budget is handled by the Critic
    steps = [s for s in steps if s.step_type != "verify_budget"]

    # Strip price-fetch steps if not standalone
    if mode != "standalone":
        steps = [s for s in steps if s.step_type not in ("fetch_avg_prices", "fetch_min_prices")]

    # Strip switch_travel_options unless explicitly requested
    if not need_switch_travel:
        steps = [s for s in steps if s.step_type != "switch_travel_options"]

    # Drop already-completed non-build steps and deduplicate
    seen_types: set[str] = set()
    deduped: list[PlanStep] = []
    for s in steps:
        if s.step_type in completed and s.step_type != "build_day_schedule":
            continue
        if s.step_type != "build_day_schedule":
            if s.step_type in seen_types:
                continue
            seen_types.add(s.step_type)
        deduped.append(s)
    steps = deduped

    build_steps = [s for s in steps if s.step_type == "build_day_schedule"]
    other_steps = [s for s in steps if s.step_type != "build_day_schedule"]

    # Ensure fetch_activities is present before build steps
    other_types = {s.step_type for s in other_steps}
    if build_steps and "fetch_activities" not in other_types and "fetch_activities" not in completed:
        other_steps.append(PlanStep(
            step_id=0,
            step_type="fetch_activities",
            description=f"Fetch and select activities in {destination}",
        ))

    # In standalone mode: inject fetch_avg_prices / fetch_min_prices if absent
    if mode == "standalone":
        if "fetch_avg_prices" not in other_types and "fetch_avg_prices" not in completed:
            other_steps.append(PlanStep(
                step_id=0,
                step_type="fetch_avg_prices",
                description=f"Fetch average flight + hotel prices for {destination}",
            ))
        if need_min_prices and "fetch_min_prices" not in other_types and "fetch_min_prices" not in completed:
            other_steps.append(PlanStep(
                step_id=0,
                step_type="fetch_min_prices",
                description=f"Fetch minimum available flight + hotel prices for {destination}",
            ))

    if mode == "with_travel_data":
        if need_switch_travel and "switch_travel_options" not in other_types and "switch_travel_options" not in completed:
            other_steps.append(PlanStep(
                step_id=0,
                step_type="switch_travel_options",
                description="Switch to cheapest available flight and hotel options",
            ))

    # Drop build steps for already-completed or out-of-range days
    build_steps = [
        s for s in build_steps
        if s.day is not None and s.day <= trip_days and s.day not in completed_days
    ]

    # Fill missing days
    existing_days = {s.day for s in build_steps if s.day}
    required_days = set(range(1, trip_days + 1)) - completed_days
    for d in sorted(required_days - existing_days):
        build_steps.append(PlanStep(
            step_id=0,
            step_type="build_day_schedule",
            description=f"Day {d}: build deterministic schedule",
            day=d,
        ))
    build_steps.sort(key=lambda s: s.day or 999)

    final_steps = other_steps + build_steps
    for i, s in enumerate(final_steps, start=1):
        s.step_id = i

    _CANONICAL_DESC: dict[str, str] = {
        "fetch_activities":      f"Fetch and select activities in {destination}",
        "fetch_weather":         f"Seasonal weather conditions for {destination}",
        "fetch_avg_prices":      f"Fetch average flight + hotel prices for {destination}",
        "fetch_min_prices":      f"Fetch minimum available flight + hotel prices for {destination}",
        "switch_travel_options": "Switch to cheapest available flight and hotel options",
    }
    for s in final_steps:
        if s.step_type in _CANONICAL_DESC:
            s.description = _CANONICAL_DESC[s.step_type]

    plan.steps = final_steps
    return plan


def _default_plan(
    destination: str,
    origin: str,
    trip_days: int,
    replan_count: int,
    completed: list[str],
    mode: str = "standalone",
    need_min_prices: bool = False,
    need_switch_travel: bool = False,
) -> ExecutionPlan:
    """Deterministic plan — only includes steps not already completed."""
    steps: list[PlanStep] = []
    sid = 1

    def add(step_type: str, desc: str, day: Optional[int] = None) -> None:
        nonlocal sid
        if step_type != "build_day_schedule" and step_type in completed:
            return
        steps.append(PlanStep(step_id=sid, step_type=step_type, description=desc, day=day))
        sid += 1

    add("fetch_activities", f"Fetch and select activities in {destination}")
    add("fetch_weather",    f"Seasonal weather in {destination}")
    if mode == "standalone":
        add("fetch_avg_prices", f"Fetch average flight + hotel prices for {destination}")
        if need_min_prices:
            add("fetch_min_prices", f"Fetch minimum available flight + hotel prices for {destination}")
    if mode == "with_travel_data" and need_switch_travel:
        add("switch_travel_options", "Switch to cheapest available flight and hotel options")
    for d in range(1, trip_days + 1):
        add("build_day_schedule", f"Day {d}: build schedule", day=d)

    return ExecutionPlan(
        destination=destination,
        origin=origin,
        total_days=trip_days,
        steps=steps,
        retry_count=replan_count,
    )


# ---------------------------------------------------------------------------
# Helpers — update mode (day resolution)
# ---------------------------------------------------------------------------

def _day_of(wrapped: dict) -> Optional[int]:
    if not isinstance(wrapped, dict):
        return None
    inner = wrapped.get("data") if "data" in wrapped else wrapped
    return inner.get("day") if isinstance(inner, dict) else None


def _find_day_of_target(target_name: str, results: dict) -> Optional[int]:
    """Find which built day currently contains a slot named like target_name."""
    target_lower = target_name.lower()
    for key, val in results.items():
        if not key.startswith("build_day_schedule") or not isinstance(val, dict):
            continue
        inner = val.get("data", val) if isinstance(val, dict) else {}
        if not isinstance(inner, dict):
            continue
        day_num = inner.get("day")
        if not day_num:
            continue
        for slot in inner.get("slots", []):
            if target_lower in slot.get("name", "").lower():
                return day_num
    return None


def _day_with_fewest_activities(results: dict) -> Optional[int]:
    """Return the day number that has the fewest activity slots (used as fallback placement)."""
    counts: dict[int, int] = {}
    for key, val in results.items():
        if not key.startswith("build_day_schedule") or not isinstance(val, dict):
            continue
        inner = val.get("data", val) if isinstance(val, dict) else {}
        day_num = inner.get("day") if isinstance(inner, dict) else None
        if not day_num:
            continue
        counts[day_num] = sum(
            1 for s in inner.get("slots", []) if s.get("slot_type") == "activity"
        )
    return min(counts, key=lambda d: counts[d]) if counts else None


def _find_best_day(activity_name: str, results: dict) -> Optional[int]:
    """Return the day whose current activity centroid is nearest to the given activity."""
    acts_raw = next(
        (v for k, v in results.items() if k.startswith("fetch_activities")), None
    )
    if not isinstance(acts_raw, dict):
        return None
    acts_data     = acts_raw.get("data", acts_raw)
    all_activities: list[dict] = acts_data.get("activities", []) if isinstance(acts_data, dict) else []

    target = next(
        (a for a in all_activities if activity_name.lower() in a.get("name", "").lower()),
        None,
    )
    if not target:
        return None

    tlat = float(target.get("latitude") or target.get("lat") or 0)
    tlng = float(target.get("longitude") or target.get("lng") or 0)
    if not (tlat or tlng):
        return None

    act_idx = {a["name"]: a for a in all_activities}
    best_day, best_dist = None, float("inf")

    for key, val in results.items():
        if not key.startswith("build_day_schedule") or not isinstance(val, dict):
            continue
        inner   = val.get("data", val) if isinstance(val, dict) else {}
        day_num = inner.get("day") if isinstance(inner, dict) else None
        if not day_num:
            continue

        act_names = [s["name"] for s in inner.get("slots", []) if s.get("slot_type") == "activity"]
        coords = [
            (float(a.get("latitude") or a.get("lat") or 0),
             float(a.get("longitude") or a.get("lng") or 0))
            for name in act_names
            if (a := act_idx.get(name))
        ]
        valid = [(la, lo) for la, lo in coords if la or lo]
        if not valid:
            if best_day is None:
                best_day = day_num
            continue

        clat = sum(c[0] for c in valid) / len(valid)
        clng = sum(c[1] for c in valid) / len(valid)
        dlat = math.radians(tlat - clat)
        dlng = math.radians(tlng - clng)
        a_v  = (math.sin(dlat / 2) ** 2
                + math.cos(math.radians(clat)) * math.cos(math.radians(tlat))
                * math.sin(dlng / 2) ** 2)
        dist = 6371 * 2 * math.asin(math.sqrt(max(0.0, a_v)))
        if dist < best_dist:
            best_dist, best_day = dist, day_num

    return best_day


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class ItineraryPlannerNode:
    def __init__(self, llm: BaseChatModel | None = None) -> None:
        # LLM is only used on the first entry into update mode (one structured call).
        # All other planning paths are fully deterministic.
        self._update_llm = (
            llm.with_structured_output(DayUpdateInstruction, method="function_calling")
            if llm else None
        )

    def __call__(self, state: AgentState) -> dict:
        destination = state.get("destination_city", "")
        origin      = state.get("current_city", "")
        trip_days   = state.get("trip_days", 3)
        budget      = state.get("total_budget", 0)
        mode        = state.get("itinerary_mode", "standalone")
        intent      = state.get("intent", "")

        prev_plan          = state.get("itinerary_plan") or {}
        replan_count       = prev_plan.get("replan_count", 0)
        step_results       = prev_plan.get("step_results", {})
        replan_raw         = prev_plan.get("replan_context", "")
        need_min_prices    = bool(state.get("use_min_prices_for_budget"))
        need_switch_travel = bool(state.get("switch_travel_triggered")) and mode == "with_travel_data"

        is_replan = bool(replan_raw)

        # ── Hard stop ──────────────────────────────────────────────────────
        if replan_count >= MAX_REPLANS:
            reason = replan_raw or "max_replans_exceeded"
            return {
                "itinerary_feasible":       False,
                "itinerary_fallback_reason": reason,
                "messages": [AIMessage(
                    content=(
                        f"❌ **MAX REPLANS REACHED** (`replan={replan_count}`). "
                        "Passing to Formatter."
                    ),
                    name="planner_log",
                )],
            }

        # ── Update mode: first entry ────────────────────────────────────────
        # Detected by update_itinerary intent with no active replan context.
        # Subsequent replans within update mode have non-empty replan_raw.
        extra_state: dict = {}
        is_first_update = intent == "update_itinerary" and not replan_raw

        if is_first_update:
            mode         = "update"
            replan_count = 0  # fresh replan budget for this update

            instruction: Optional[DayUpdateInstruction] = None
            if self._update_llm:
                messages   = state.get("messages", [])
                last_human = next(
                    (m for m in reversed(messages) if getattr(m, "type", "") == "human"), None
                )
                if last_human:
                    try:
                        instruction = self._update_llm.invoke([
                            SystemMessage(content=_EXTRACTION_PROMPT),
                            HumanMessage(content=str(last_human.content)),
                        ])
                    except Exception:
                        pass

            if instruction:
                # Resolve any_day additions → geographically best-fit day, fallback to least-busy
                if instruction.scope == "any_day" and instruction.replacement_name:
                    best = _find_best_day(instruction.replacement_name, step_results)
                    if not best:
                        best = _day_with_fewest_activities(step_results) or 1
                    instruction = instruction.model_copy(
                        update={"day": best, "scope": "single_day"}
                    )

                # Resolve missing day for exclusions → search slot names
                if not instruction.day and instruction.target_name and instruction.scope != "all_days":
                    found = _find_day_of_target(instruction.target_name, step_results)
                    if found:
                        instruction = instruction.model_copy(
                            update={"day": found, "scope": "single_day"}
                        )

                # Drop the affected day so the executor rebuilds it
                if instruction.scope == "single_day" and instruction.day:
                    step_results = {
                        k: v for k, v in step_results.items()
                        if not (
                            k.startswith("build_day_schedule")
                            and _day_of(v) == instruction.day
                        )
                        and not k.startswith("verify_budget")
                    }

                extra_state = {
                    "itinerary_update_request": instruction.model_dump(),
                    "itinerary_mode": "update",
                }

        # ── Context assembly ───────────────────────────────────────────────
        replan_context = _parse_replan_context(replan_raw) if is_replan else None
        completed = _completed_step_types(step_results)

        if replan_context and isinstance(replan_context.get("completed_steps"), list):
            for s in replan_context["completed_steps"]:
                if s not in completed:
                    completed.append(s)

        # ── Log header ─────────────────────────────────────────────────────
        if is_first_update:
            plan_md = f"🔄 **UPDATE (extracting instruction):** `{destination}` · mode `update`\n"
        elif is_replan:
            reason_msg  = (replan_context or {}).get("error_message", replan_raw)
            failed_step = (replan_context or {}).get("failed_step", "unknown")
            plan_md = (
                f"🔄 **REPLANNING**\n"
                f"*Reason:* {reason_msg}\n"
                f"*Failed step:* `{failed_step}`\n"
                f"*Completed so far:* {completed}\n"
            )
        else:
            plan_md = (
                f"**PLANNING:** `{destination}` · `{trip_days} days` · "
                f"budget `${budget}` · mode `{mode}`\n"
            )

        # ── Plan generation ────────────────────────────────────────────────
        comp_days = _completed_days(step_results)

        if mode == "update":
            # On first entry, prefer the just-extracted instruction over stale state
            update_req = (
                extra_state.get("itinerary_update_request")
                if is_first_update and extra_state.get("itinerary_update_request")
                else (state.get("itinerary_update_request") or {})
            )
            update_scope = update_req.get("scope", "single_day")
            update_day   = update_req.get("day")

            if update_scope == "all_days":
                steps = [PlanStep(
                    step_id=1,
                    step_type="apply_global_preference",
                    description="Apply preference change to all days",
                )]
            else:
                steps = [PlanStep(
                    step_id=1,
                    step_type="update_day_schedule",
                    description=f"Update Day {update_day}",
                    day=update_day,
                )] if update_day else []

            plan = ExecutionPlan(
                destination=destination,
                origin=origin,
                total_days=trip_days,
                steps=steps,
                retry_count=replan_count,
            )
        else:
            plan = _default_plan(
                destination, origin, trip_days, replan_count, completed,
                mode, need_min_prices, need_switch_travel,
            )
            plan = _validate_and_fix(
                plan, completed, trip_days, destination, comp_days,
                mode, need_min_prices, need_switch_travel,
            )

        plan_md += "\n**Execution Plan:**\n"
        for step in plan.steps:
            day_tag = f" (Day {step.day})" if step.day else ""
            plan_md += f"  • `[{step.step_id:02d}]` **{step.step_type}**{day_tag}"
            if step.description:
                plan_md += f" — _{step.description}_"
            plan_md += "\n"

        if completed:
            plan_md += (
                f"\n⏩ **Skipping already-completed:** "
                + ", ".join(f"`{s}`" for s in completed) + "\n"
            )

        result = {
            "current_step_index": 0,
            "itinerary_plan": {
                "execution_plan":    plan.model_dump(),
                "step_results":      step_results,
                "replan_count":      replan_count + 1,
                "replan_context":    "",
                "recovery_attempts": prev_plan.get("recovery_attempts", 0),
            },
            "itinerary_feasible":       True,
            "itinerary_fallback_reason": None,
            "progress_log": [plan_md.strip()],
        }
        result.update(extra_state)
        return result
