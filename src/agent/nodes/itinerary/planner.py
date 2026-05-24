# src/agent/nodes/itinerary/planner.py
"""
ItineraryPlannerNode — v2
=========================
Unchanged responsibility: generate an ordered ExecutionPlan.

What changed vs v1:
  - Explicit note in system prompt that build_day_schedule is
    now handled by the ScheduleEngine (LLM does NOT write times).
  - Cleaner separation of concerns documented.
  - fetch_activities step always comes BEFORE build_day_schedule steps
    (required because the executor runs ActivitySelector during fetch_activities).
"""
from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agent.nodes.itinerary.schemas import ExecutionPlan, PlanStep
from agent.state import AgentState

logger = logging.getLogger(__name__)
MAX_RETRIES = 3

SYSTEM_PROMPT = """
You are a deterministic travel execution planner.

Your ONLY responsibility: generate an ordered list of execution steps.

ARCHITECTURE NOTE:
  - fetch_activities triggers AI-based activity SELECTION (geographic clustering, energy curve).
  - build_day_schedule runs a deterministic ScheduleEngine (not LLM-based).
  - You do NOT generate times, costs, or schedules. Only the step list.

RULES:
1. fetch_* steps appear EXACTLY ONCE each.
2. fetch_activities MUST come before all build_day_schedule steps.
3. build_day_schedule appears once per day (day=1, day=2, ...).
4. verify_budget MUST always be last.
5. NEVER generate duplicate steps.
6. Order: fetch_flights → fetch_return_flights → fetch_hotels → fetch_weather
         → fetch_activities → build_day_1 → ... → build_day_N → verify_budget
"""


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
        observer_reason = prev_plan.get("observer_reason", "")

        if retry_count == 0:
            print(f"\n--- 🧠 PLANNING: {destination} · {trip_days} days ---")
        else:
            print(f"\n--- 🔄 REPLANNING (attempt {retry_count}/{MAX_RETRIES}): {observer_reason} ---")

        if retry_count >= MAX_RETRIES:
            print("--- ❌ MAX RETRIES REACHED. Passing to Fallback. ---")
            return {
                "itinerary_feasible": False,
                "itinerary_fallback_reason": observer_reason or "max_retries_exceeded",
            }

        user_msg = (
            f"Plan a {trip_days}-day trip.\n"
            f"Origin: {origin}\nDestination: {destination}\n"
            f"Budget: ${budget or 'flexible'}\nPreferences: {prefs}\n"
            + (f"\nPREVIOUS FAILURE — fix this: {observer_reason}" if observer_reason else "")
        )

        try:
            plan: ExecutionPlan = self.llm.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_msg),
            ])
            plan.retry_count = retry_count
        except Exception as e:
            logger.error("Planner LLM failed: %s — using default plan", e)
            plan = _default_plan(destination, origin, trip_days, retry_count)

        print("📝 Execution Plan:")
        for step in plan.steps:
            day_tag = f" (Day {step.day})" if step.day else ""
            print(f"  [{step.step_id}] {step.step_type}{day_tag}")

        return {
            "current_step_index": 0,
            "itinerary_plan": {
                "execution_plan": plan.model_dump(),
                "step_results": prev_plan.get("step_results", {}),
                "retry_count": retry_count,
                "observer_reason": "",
            },
            "itinerary_feasible": True,
            "itinerary_fallback_reason": None,
        }


def _default_plan(destination: str, origin: str, trip_days: int, retry_count: int) -> ExecutionPlan:
    steps = [
        PlanStep(step_id=1, step_type="fetch_flights",
                 description=f"Outbound flights {origin} → {destination}"),
        PlanStep(step_id=2, step_type="fetch_return_flights",
                 description=f"Return flights {destination} → {origin}"),
        PlanStep(step_id=3, step_type="fetch_hotels",
                 description=f"Hotels in {destination}"),
        PlanStep(step_id=4, step_type="fetch_weather",
                 description=f"Weather in {destination}"),
        PlanStep(step_id=5, step_type="fetch_activities",
                 description=f"Activities in {destination} + AI selection"),
    ]
    for d in range(1, trip_days + 1):
        steps.append(PlanStep(
            step_id=5 + d,
            step_type="build_day_schedule",
            description=f"Day {d}: deterministic schedule build",
            day=d,
        ))
    steps.append(PlanStep(
        step_id=6 + trip_days,
        step_type="verify_budget",
        description="Verify total cost vs budget",
    ))
    return ExecutionPlan(
        destination=destination, origin=origin,
        total_days=trip_days, steps=steps,
        retry_count=retry_count,
    )
