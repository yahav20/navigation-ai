# src/agent/nodes/itinerary/planner.py
"""
ItineraryPlannerNode — v3
=========================
Key improvements over v2:

ADAPTIVE PLANNING
  - LLM is NOT given a fixed step order — it reasons from context.
  - On replan, completed steps are excluded so the executor never repeats them.
  - Structured replan context (JSON) replaces the plain string hint.

STRUCTURED REPLAN CONTEXT
  Each replan call includes a JSON snapshot:
    {
      "error_code": "...",       # category of failure
      "error_message": "...",    # human-readable reason
      "failed_step": "...",      # step_type that failed
      "replan_hint": "...",      # specific corrective suggestion
      "completed_steps": [...],  # step_types already done — SKIP THESE
      "system_state": {          # budget, days, etc.
        "budget": ...,
        "trip_days": ...,
        "destination": "...",
        "origin": "...",
      }
    }

RETRY vs REPLAN DISTINCTION
  - retry_count  : how many times the same step was retried by the executor
  - replan_count : how many times the planner generated a new plan

SAFETY
  - MAX_REPLANS hard-stops the planner and triggers fallback.
  - Completed steps are NEVER re-emitted (history-based dedup).
  - Default plan is only used when LLM output is unparseable.
"""
from __future__ import annotations

import json
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from agent.nodes.itinerary.schemas import ExecutionPlan, PlanStep
from agent.state import AgentState

# ── Safety limits ─────────────────────────────────────────────────────────
MAX_REPLANS = 3   # how many times Planner may generate a new plan
MAX_RETRIES = 3   # guard passed in from Observer; Planner checks before running
# ── Step types the Planner knows about ────────────────────────────────────
VALID_STEP_TYPES = {
    "fetch_flights",
    "fetch_return_flights",
    "fetch_hotels",
    "fetch_activities",
    "fetch_weather",
    "build_day_schedule",
    "verify_budget",
}

# Steps that MUST appear before build_day_schedule
PREREQUISITE_STEPS = {
    "fetch_flights",
    "fetch_return_flights",
    "fetch_hotels",
    "fetch_activities",
}

# ── System Prompt ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are an adaptive travel execution planner inside a multi-agent system.

Your ONLY responsibility: generate a minimal, ordered list of execution steps
that will produce a complete, feasible itinerary.

You DO NOT execute tools.
You DO NOT generate the final itinerary.
You ONLY decide:
- what steps should run
- what order they should run in
- what should be retried
- what can be skipped during replanning

SYSTEM ARCHITECTURE:
- Planner (you):
    Creates and updates adaptive execution plans.
- Executor:
    Executes ONE step at a time using tools autonomously.
    Returns structured execution results:
      status: success | failed | retrying | fallback_used
      data
      error
      replan_hint
      trace
- Observer:
    Validates execution results and controls system state transitions.
    Decides whether to:
      continue
      retry
      replan
      fallback
      finalize

AVAILABLE STEP TYPES:
  fetch_flights         — outbound flights (origin → destination)
  fetch_return_flights  — return flights (destination → origin)
  fetch_hotels          — available hotels at destination
  fetch_weather         — seasonal weather at destination
  fetch_activities      — activities/attractions at destination
  build_day_schedule    — deterministic day builder (requires day=N field)
  verify_budget         — final cost check (must be last)

ARCHITECTURE NOTES:
  - fetch_activities also runs AI activity-selection internally; it MUST come
    before all build_day_schedule steps.
  - build_day_schedule is deterministic — you do NOT specify times or costs.
  - verify_budget always comes last.
  - The Executor may fail, retry, or use fallback strategies.
  - You must use execution history and failures when replanning.

ADAPTIVE RULES:
  1. Only include steps that are actually needed given the context.
  2. NEVER re-emit a step listed in "completed_steps" — those results are cached.
  3. On replan, only emit the REMAINING steps (start from the first incomplete one).
  4. If a fetch step failed, include it again so the executor retries.
  5. If budget is a concern, still include verify_budget — do not skip it.
  6. If a previous build_day_schedule failed for a specific day, re-emit only
     that day's build_day_schedule step (not all days).
  7. Prefer minimal viable plans over maximal ones.
  8. You MAY dynamically add, remove, reorder, or retry steps based on:
       - execution history
       - observer feedback
       - failures
       - partial results
  9. NEVER duplicate already completed successful work.
 10. Avoid infinite retry loops by preferring minimal recovery plans.

ORDERING CONSTRAINTS (non-negotiable):
  - fetch_activities MUST appear before any build_day_schedule.
  - verify_budget MUST be the final step.
  - fetch_* steps should appear before build_day_schedule steps.
  - build_day_schedule requires all prerequisite fetch steps.

OUTPUT:
Return ONLY a valid ExecutionPlan (structured JSON).
No markdown.
No explanation.
"""
# ── Per-step human prompt builder ─────────────────────────────────────────

def _build_user_message(
    destination: str,
    origin: str,
    trip_days: int,
    budget: float,
    prefs: dict,
    replan_context: Optional[dict],
    completed_step_types: list[str],
    replan_count: int,
) -> str:
    parts = [
        f"Destination: {destination}",
        f"Origin: {origin}",
        f"Trip duration: {trip_days} days",
        f"Budget: ${budget or 'flexible'}",
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


# ── Completed-step tracker ─────────────────────────────────────────────────

def _completed_step_types(step_results: dict) -> list[str]:
    """
    Return step_types that have a non-error result cached in step_results.
    Used to tell the Planner which steps it must NOT re-emit.
    """
    completed = []
    for key, val in step_results.items():
        if isinstance(val, dict) and "error" in val:
            continue  # failed — allow replanning
        # key format is "{step_type}_{step_id}"
        step_type = "_".join(key.split("_")[:-1]) if "_" in key else key
        if step_type and step_type not in completed:
            completed.append(step_type)
    return completed


def _parse_replan_context(observer_reason: str) -> Optional[dict]:
    """
    Try to parse a JSON replan context from the observer_reason string.
    Falls back gracefully to a plain-text wrapper if it isn't valid JSON.
    """
    if not observer_reason:
        return None
    try:
        ctx = json.loads(observer_reason)
        if isinstance(ctx, dict):
            return ctx
    except (json.JSONDecodeError, TypeError):
        pass
    # Plain string — wrap it
    return {
        "error_code": "unknown",
        "error_message": observer_reason,
        "failed_step": None,
        "replan_hint": observer_reason,
        "completed_steps": [],
    }


# ── Validation & sanitisation ──────────────────────────────────────────────

def _validate_and_fix(
    plan: ExecutionPlan,
    completed: list[str],
    trip_days: int,
    destination: str,
    origin: str,
) -> ExecutionPlan:
    """
    Post-process the LLM plan to enforce hard constraints:
      1. Remove steps whose step_type is already completed.
      2. Ensure fetch_activities precedes build_day_schedule.
      3. Ensure verify_budget is last.
      4. Fill in missing day= fields for build_day_schedule.
      5. Remove duplicate step_types (keep first occurrence, except build_day_schedule).
      6. Re-number step_ids sequentially.
    """
    steps = plan.steps

    # 1. Drop already-completed non-build steps
    seen_types: set[str] = set()
    deduped = []
    for s in steps:
        if s.step_type in completed and s.step_type != "build_day_schedule":
            continue  # already done
        if s.step_type != "build_day_schedule":
            if s.step_type in seen_types:
                continue  # duplicate non-day step
            seen_types.add(s.step_type)
        deduped.append(s)

    steps = deduped

    # 2. Separate build_day and non-build
    build_steps = [s for s in steps if s.step_type == "build_day_schedule"]
    other_steps = [s for s in steps if s.step_type != "build_day_schedule"
                   and s.step_type != "verify_budget"]
    budget_steps = [s for s in steps if s.step_type == "verify_budget"]

    # 3. Ensure fetch_activities is present before build steps (if not completed)
    other_types = {s.step_type for s in other_steps}
    if build_steps and "fetch_activities" not in other_types and "fetch_activities" not in completed:
        other_steps.append(PlanStep(
            step_id=0,
            step_type="fetch_activities",
            description=f"Activities in {destination} + AI selection",
        ))
    build_steps = [
        s for s in build_steps
        if s.day is not None
    ]
    # 4. Fill missing days in build_day_schedule
    existing_days = {s.day for s in build_steps if s.day}
    all_days = set(range(1, trip_days + 1))
    missing_days = sorted(all_days - existing_days)

    for d in missing_days:
        build_steps.append(PlanStep(
            step_id=0,
            step_type="build_day_schedule",
            description=f"Day {d}: deterministic schedule build",
            day=d,
        ))

    # Sort build steps by day
    build_steps.sort(key=lambda s: s.day or 999)

    # 5. Ensure verify_budget exists
    if not budget_steps:
        budget_steps = [PlanStep(
            step_id=0,
            step_type="verify_budget",
            description="Verify total cost vs budget",
        )]

    # 6. Reassemble with sequential IDs
    final_steps = other_steps + build_steps + budget_steps
    for i, s in enumerate(final_steps, start=1):
        s.step_id = i

    plan.steps = final_steps
    return plan


# ── Node ───────────────────────────────────────────────────────────────────

class ItineraryPlannerNode:
    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm.with_structured_output(ExecutionPlan)

    def __call__(self, state: AgentState) -> dict:
        destination = state.get("destination_city", "")
        origin = state.get("current_city", "")
        trip_days = state.get("trip_days", 3)
        budget = state.get("total_budget", 0)
        prefs = state.get("user_preferences", {})

        prev_plan = state.get("itinerary_plan") or {}
        retry_count = prev_plan.get("retry_count", 0)
        replan_count = prev_plan.get("replan_count", 0)
        observer_reason_raw = prev_plan.get("observer_reason", "")
        step_results = prev_plan.get("step_results", {})

        # ── Hard stop ──────────────────────────────────────────────────────
        if replan_count >= MAX_REPLANS or retry_count >= MAX_RETRIES:
            reason = observer_reason_raw or "max_retries_exceeded"
            return {
                "itinerary_feasible": False,
                "itinerary_fallback_reason": reason,
                "messages": [AIMessage(
                    content=f"❌ **MAX RETRIES/REPLANS REACHED** (`replan={replan_count}`, "
                            f"`retry={retry_count}`). Passing to Fallback.",
                    name="planner_log",
                )],
            }

        # ── Context assembly ───────────────────────────────────────────────
        is_replan = replan_count > 0 or bool(observer_reason_raw)
        replan_context = _parse_replan_context(observer_reason_raw) if is_replan else None
        completed = _completed_step_types(step_results)

        # Merge completed steps from replan context if Observer provided them
        if replan_context and isinstance(replan_context.get("completed_steps"), list):
            for s in replan_context["completed_steps"]:
                if s not in completed:
                    completed.append(s)

        # ── Log header ────────────────────────────────────────────────────
        if is_replan:
            plan_md = (
                f"🔄 **REPLANNING** (plan attempt {replan_count + 1}/{MAX_REPLANS})\n"
                f"*Reason:* {replan_context.get('error_message', observer_reason_raw)}\n"
                f"*Failed step:* `{replan_context.get('failed_step', 'unknown')}`\n"
                f"*Completed so far:* {completed}\n"
            )
        else:
            plan_md = f"🧠 **PLANNING:** `{destination}` · `{trip_days} days` · budget `${budget}`\n"

        # ── LLM call ──────────────────────────────────────────────────────
        user_msg = _build_user_message(
            destination=destination,
            origin=origin,
            trip_days=trip_days,
            budget=budget,
            prefs=prefs,
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
            plan_md += f"\n⚠️ **LLM failed** (`{e}`) — using default plan\n"
            plan = _default_plan(destination, origin, trip_days, replan_count, completed)

        # ── Validate & fix plan ───────────────────────────────────────────
        plan = _validate_and_fix(plan, completed, trip_days, destination, origin)

        # ── Build log ────────────────────────────────────────────────────
        plan_md += "\n📝 **Execution Plan:**\n"
        for step in plan.steps:
            day_tag = f" (Day {step.day})" if step.day else ""
            plan_md += f"  • `[{step.step_id:02d}]` **{step.step_type}**{day_tag}"
            if step.description:
                plan_md += f" — _{step.description}_"
            plan_md += "\n"

        if completed:
            plan_md += f"\n⏩ **Skipping already-completed steps:** {', '.join(f'`{s}`' for s in completed)}\n"

        return {
            "current_step_index": 0,
            "itinerary_plan": {
                "execution_plan": plan.model_dump(),
                "step_results": step_results,  # preserve cached results
                "retry_count": retry_count,
                "replan_count": replan_count + 1,
                "observer_reason": "",
            },
            "itinerary_feasible": True,
            "itinerary_fallback_reason": None,
            "messages": [AIMessage(content=plan_md.strip(), name="planner_log")],
        }


# ── Default / fallback plan ────────────────────────────────────────────────

def _default_plan(
    destination: str,
    origin: str,
    trip_days: int,
    replan_count: int,
    completed: list[str],
) -> ExecutionPlan:
    """
    Minimal viable plan — only includes steps not already completed.
    Used when LLM output is unparseable.
    """
    steps: list[PlanStep] = []
    sid = 1

    def add(step_type: str, desc: str, day: Optional[int] = None) -> None:
        nonlocal sid
        # Skip if already done (and it's not a build_day_schedule step)
        if step_type != "build_day_schedule" and step_type in completed:
            return
        steps.append(PlanStep(step_id=sid, step_type=step_type, description=desc, day=day))
        sid += 1

    add("fetch_flights",        f"Outbound flights {origin} → {destination}")
    add("fetch_return_flights", f"Return flights {destination} → {origin}")
    add("fetch_hotels",         f"Hotels in {destination}")
    add("fetch_weather",        f"Weather in {destination}")
    add("fetch_activities",     f"Activities in {destination} + AI selection")

    for d in range(1, trip_days + 1):
        add("build_day_schedule", f"Day {d}: deterministic schedule build", day=d)

    add("verify_budget", "Verify total cost vs budget")

    return ExecutionPlan(
        destination=destination,
        origin=origin,
        total_days=trip_days,
        steps=steps,
        retry_count=replan_count,
    )