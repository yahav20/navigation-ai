"""
ItineraryPlannerNode
====================
Receives: destination, origin, trip_days, budget, user_preferences.
Produces: ExecutionPlan — an ordered list of PlanSteps.

The LLM is forced to output a valid ExecutionPlan via structured output.
No tool calls here — the Planner only *plans*, never *executes*.
"""
# src/agent/nodes/itinerary/planner.py
from __future__ import annotations
import logging
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from agent.nodes.itinerary.schemas import ExecutionPlan, PlanStep
from agent.state import AgentState

logger = logging.getLogger(__name__)

MAX_RETRIES = 3

SYSTEM_PROMPT = """
You are a deterministic travel execution planner.

Your ONLY responsibility:
generate an ordered execution plan.

RULES:
1. fetch_* tools appear EXACTLY ONCE.
2. build_day_schedule appears once per day.
3. verify_budget must always be last.
4. Plans must support:
   - geo-aware transportation
   - strict timing validation
   - meal integration logic
   - arrival/departure anchors
   - non-overlapping schedules

NEVER generate duplicate steps.
NEVER skip verify_budget.
"""
class ItineraryPlannerNode:
    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm.with_structured_output(ExecutionPlan)

    def __call__(self, state: AgentState) -> dict:
        destination = state.get("destination_city", "")
        origin      = state.get("current_city", "")
        trip_days   = state.get("trip_days", 3)
        budget      = state.get("total_budget", 0)
        prefs       = state.get("user_preferences", {})

        prev_plan       = state.get("itinerary_plan") or {}
        retry_count     = prev_plan.get("retry_count", 0)
        observer_reason = prev_plan.get("observer_reason", "")

        # הדפסת הטרמינל לשלב התכנון (מזהה אם זה תכנון ראשון או Replanning)
        if retry_count == 0:
            print(f"\n--- 🧠 PLANNING PHASE: Creating initial plan for {destination} ({trip_days} days) ---")
        else:
            print(f"\n--- 🔄 REPLANNING PHASE (Attempt {retry_count}/{MAX_RETRIES}): Adjusting plan due to '{observer_reason}' ---")

        if retry_count >= MAX_RETRIES:
            print("--- ❌ MAX RETRIES REACHED. Passing to Fallback. ---")
            return {
                "itinerary_feasible": False,
                "itinerary_fallback_reason": observer_reason or "max_retries_exceeded",
            }

        user_msg = (
            f"Plan a {trip_days}-day trip.\n"
            f"Origin: {origin}\nDestination: {destination}\n"
            f"Budget: ${budget or 'flexible'}\nUser preferences: {prefs}\n"
            + (f"PREVIOUS FAILURE TO AVOID/FIX: {observer_reason}" if observer_reason else "")
        )

        try:
            plan: ExecutionPlan = self.llm.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_msg),
            ])
            plan.retry_count = retry_count
        except Exception as e:
            logger.error("Planner failed: %s", e)
            plan = _default_plan(destination, origin, trip_days, retry_count)

        # הדפסת התוכנית לטרמינל
        print("📝 Generated Plan:")
        for step in plan.steps:
            print(f"  - Step {step.step_id}: {step.step_type} ({step.description})")

        # מחזירים איפוס של ה-current_step_index כדי שה-Executor יתחיל מההתחלה
        return {
            "current_step_index": 0,
            "itinerary_plan": {
                "execution_plan": plan.model_dump(),
                "step_results": prev_plan.get("step_results", {}), # שומרים תוצאות קודמות כדי למנוע קריאות כפולות
                "retry_count": retry_count,
                "observer_reason": "",
            },
            "itinerary_feasible": True,
            "itinerary_fallback_reason": None,
        }

def _default_plan(destination: str, origin: str, trip_days: int, retry_count: int) -> ExecutionPlan:
    steps = [
        PlanStep(step_id=1, step_type="fetch_flights",
                 description=f"Fetch outbound flights {origin}->{destination}"),
        PlanStep(step_id=2, step_type="fetch_return_flights",
                 description=f"Fetch return flights {destination}->{origin}"),
        PlanStep(step_id=3, step_type="fetch_hotels",
                 description=f"Fetch hotels in {destination}"),
        PlanStep(step_id=4, step_type="fetch_weather",
                 description=f"Fetch weather in {destination}"),
        PlanStep(step_id=5, step_type="fetch_activities",
                 description=f"Fetch top-rated activities in {destination}"),
    ]
    for d in range(1, trip_days + 1):
        steps.append(PlanStep(
            step_id=5 + d, step_type="build_day_schedule",
            description=f"Day {d}: build full day schedule", day=d,
        ))
    steps.append(PlanStep(
        step_id=6 + trip_days, step_type="verify_budget",
        description="Verify total cost against budget",
    ))
    return ExecutionPlan(destination=destination, origin=origin,
                         total_days=trip_days, steps=steps, retry_count=retry_count)
