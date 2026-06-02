"""
ItineraryPlannerNode — v4
=========================
Generates a minimal ordered execution plan for the schedule builder.

Two modes (set by PlanCheckNode / edge.py):
  with_travel_data — state["travel_plan"] already has flights + hotel.
                     Steps include build_day_schedule anchored to real
                     arrival/departure times.
  standalone       — no booking data. Steps are the same but build_day_schedule
                     has no flight anchors.

Budget verification is NOT a plan step — it is computed by the Replanner after
all day schedules are built and enforced exclusively by the Critic node.

Replan flow:
  On the first call, replan_context is empty → fresh plan.
  On subsequent calls (from Replanner), replan_context contains a JSON blob
  describing what failed and which steps already succeeded. The Planner emits
  only the remaining steps.

Safety:
  MAX_REPLANS hard-stops the Planner and hands off to the Formatter with an
  error message when the Replanner has exhausted retries.
"""
from __future__ import annotations

import json
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.nodes.itinerary.schemas import ExecutionPlan, PlanStep
from agent.state import AgentState

# ── Safety limits ──────────────────────────────────────────────────────────
MAX_REPLANS = 3

# ── Valid step types ───────────────────────────────────────────────────────
VALID_STEP_TYPES = {
    "fetch_activities",
    "fetch_weather",
    "build_day_schedule",
}

# Steps that must complete before any build_day_schedule can run
PREREQUISITE_STEPS = {"fetch_activities"}

# ── System prompt ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are a travel schedule planner. Output a minimal ordered list of execution steps.

STEP TYPES (use exact names):
  fetch_activities, fetch_weather,
  build_day_schedule (needs day=N)

RULES:
- fetch_activities MUST come before all build_day_schedule steps.
- Do NOT include verify_budget — budget verification is handled automatically after scheduling.
- NEVER re-emit steps listed in "completed_steps".
- On replan: emit only remaining steps, skip completed ones.
- Each build_day_schedule needs its own entry with the correct day number.
- If replanning due to missing resources, use suggested_adjustments:
    trip_days (int) to reduce days, or total_budget (float) to raise budget.

Output only the structured JSON. Include a description for every step.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_user_message(
    destination: str,
    origin: str,
    trip_days: int,
    budget: float,
    prefs: dict,
    mode: str,
    replan_context: Optional[dict],
    completed_step_types: list[str],
    replan_count: int,
) -> str:
    parts = [
        f"Destination: {destination}",
        f"Origin: {origin}",
        f"Trip duration: {trip_days} days",
        f"Budget: ${budget or 'flexible'}",
        f"Mode: {mode}",
        f"User preferences: {json.dumps(prefs, ensure_ascii=False)}",
    ]
    if completed_step_types:
        parts.append(
            f"\nCompleted steps (DO NOT re-emit): {json.dumps(completed_step_types)}"
        )
    if replan_context:
        parts.append(
            f"\nREPLAN CONTEXT (attempt {replan_count}/{MAX_REPLANS}):\n"
            + json.dumps(replan_context, ensure_ascii=False, indent=2)
        )
    else:
        parts.append("\nThis is the initial plan — include all necessary steps.")
    return "\n".join(parts)


def _completed_step_types(step_results: dict) -> list[str]:
    """Return step_types that have a successful (non-error) v3 result cached."""
    completed: list[str] = []
    for key, val in step_results.items():
        if not isinstance(val, dict):
            continue
        if val.get("status") != "success":
            continue
        # key format: "{step_type}_{step_id}" — strip trailing _N
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
    """Parse JSON replan context written by the Replanner. Falls back gracefully."""
    if not raw:
        return None
    try:
        ctx = json.loads(raw)
        if isinstance(ctx, dict):
            return ctx
    except (json.JSONDecodeError, TypeError):
        pass
    return {
        "error_code": "unknown",
        "error_message": raw,
        "failed_step": None,
        "replan_hint": raw,
        "completed_steps": [],
    }


def _validate_and_fix(
    plan: ExecutionPlan,
    completed: list[str],
    trip_days: int,
    destination: str,
    completed_days: set[int],
) -> ExecutionPlan:
    """Enforce hard ordering constraints and fill missing day steps."""
    steps = plan.steps

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

    plan.steps = final_steps
    return plan


def _default_plan(
    destination: str,
    origin: str,
    trip_days: int,
    replan_count: int,
    completed: list[str],
) -> ExecutionPlan:
    """Minimal viable plan — only includes steps not already completed."""
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
# Node
# ---------------------------------------------------------------------------

class ItineraryPlannerNode:
    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm.with_structured_output(ExecutionPlan, method="function_calling")

    def __call__(self, state: AgentState) -> dict:
        destination = state.get("destination_city", "")
        origin      = state.get("current_city", "")
        trip_days   = state.get("trip_days", 3)
        budget      = state.get("total_budget", 0)
        prefs       = state.get("user_preferences", {})
        mode        = state.get("itinerary_mode", "standalone")

        prev_plan    = state.get("itinerary_plan") or {}
        replan_count = prev_plan.get("replan_count", 0)
        step_results = prev_plan.get("step_results", {})
        replan_raw   = prev_plan.get("replan_context", "")

        is_replan = bool(replan_raw)

        # ── Hard stop ──────────────────────────────────────────────────────
        if replan_count >= MAX_REPLANS:
            reason = replan_raw or "max_replans_exceeded"
            return {
                "itinerary_feasible": False,
                "itinerary_fallback_reason": reason,
                "messages": [AIMessage(
                    content=(
                        f"❌ **MAX REPLANS REACHED** (`replan={replan_count}`). "
                        "Passing to Formatter."
                    ),
                    name="planner_log",
                )],
            }

        # ── Context assembly ───────────────────────────────────────────────
        replan_context = _parse_replan_context(replan_raw) if is_replan else None
        completed = _completed_step_types(step_results)

        # Merge completed steps from replanner context
        if replan_context and isinstance(replan_context.get("completed_steps"), list):
            for s in replan_context["completed_steps"]:
                if s not in completed:
                    completed.append(s)

        # ── Log header ────────────────────────────────────────────────────
        if is_replan:
            reason_msg  = (replan_context or {}).get("error_message", replan_raw)
            failed_step = (replan_context or {}).get("failed_step", "unknown")
            plan_md = (
                f"🔄 **REPLANNING** (attempt {replan_count + 1}/{MAX_REPLANS})\n"
                f"*Reason:* {reason_msg}\n"
                f"*Failed step:* `{failed_step}`\n"
                f"*Completed so far:* {completed}\n"
            )
        else:
            plan_md = (
                f"🧠 **PLANNING:** `{destination}` · `{trip_days} days` · "
                f"budget `${budget}` · mode `{mode}`\n"
            )

        # ── LLM call ──────────────────────────────────────────────────────
        user_msg = _build_user_message(
            destination=destination,
            origin=origin,
            trip_days=trip_days,
            budget=budget,
            prefs=prefs,
            mode=mode,
            replan_context=replan_context,
            completed_step_types=completed,
            replan_count=replan_count,
        )
        try:
            plan: ExecutionPlan = self.llm.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_msg),
            ])
        except Exception as e:
            plan_md += f"\n⚠️ LLM failed (`{e}`) — using default plan\n"
            plan = _default_plan(destination, origin, trip_days, replan_count, completed)

        # ── Validate & fix ─────────────────────────────────────────────────
        # Always use trip_days from state — the critic owns any day/budget adjustments.
        comp_days = _completed_days(step_results)
        plan = _validate_and_fix(plan, completed, trip_days, destination, comp_days)

        plan_md += "\n📝 **Execution Plan:**\n"
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

        # trip_days and total_budget adjustments are managed by ItineraryCriticNode;
        # the planner never modifies these state fields directly.
        state_updates: dict = {}

        result = {
            "current_step_index": 0,
            "itinerary_plan": {
                "execution_plan": plan.model_dump(),
                "step_results":   step_results,
                "replan_count":   replan_count + 1,
                "replan_context": "",          # cleared after consumption
                "recovery_attempts": prev_plan.get("recovery_attempts", 0),
            },
            "itinerary_feasible": True,
            "itinerary_fallback_reason": None,
            "messages": [AIMessage(content=plan_md.strip(), name="planner_log")],
        }
        result.update(state_updates)
        return result
