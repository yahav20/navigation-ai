"""
ItineraryPlannerNode — Plan & Execute, Step 1: PLAN

Receives the data bundle from FlightSearchNode (already in state).
Filters it by user preferences (deterministic).
Runs a feasibility check (deterministic — no LLM).
If feasible: asks LLM to produce an ExecutionPlan (list of steps).
The LLM does NOT build the itinerary — it only decides what to do.
"""
from __future__ import annotations

import json
import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agent.nodes.itinerary.helpers import (
    check_feasibility,
    default_plan,
    filter_bundle,
    strip_fences,
)
from agent.nodes.itinerary.schemas import ExecutionPlan
from agent.state import AgentState
import time
time.sleep(7) 
logger = logging.getLogger(__name__)


PLANNER_SYSTEM = """You are a travel planning orchestrator.
Given available travel data, produce a structured execution plan as JSON.

The plan must contain these steps IN ORDER:
1. select_flight  — choose the best outbound flight from the list
2. select_hotel   — choose the best hotel matching user preferences
3. build_day      — YOU MUST CREATE EXACTLY ONE STEP PER TRIP DAY (e.g. for a 3-day trip, create Day 1, Day 2, and Day 3).
4. verify_budget  — final check that total cost fits the budget

Rules:
- Reference ONLY data that exists in the provided bundle.
- Each build_day step MUST state the day number in its description.
- You must generate EXACTLY the number of 'build_day' steps as the requested 'total_days'.
- Return ONLY valid JSON — no markdown, no explanation.
- MANDATORY: You must generate a plan that covers EXACTLY {trip_days} days. 
- If you have fewer activities than days, distribute them, or suggest leisure/exploration time. 
- Do NOT output a 1-day trip if the user requested {trip_days} days.
- STRICT COMPLIANCE: The user requested {trip_days} days. You MUST populate the plan with exactly {trip_days} 'build_day' steps.
- DO NOT shorten the trip to save money unless the budget is absolutely impossible to meet for the full duration.
- If you find 3 days are too expensive, prioritize selecting cheaper hotels or removing activities before removing days.
If you are struggling to fit the budget for 3 days, look for hotels with fewer stars.
OUTPUT SCHEMA EXAMPLE (For a 3-day trip):
{
  "destination": "string",
  "origin": "string",
  "total_days": 3,
  "steps": [
    {"step_id": 1, "step_type": "select_flight",  "description": "Select outbound flight", "depends_on": []},
    {"step_id": 2, "step_type": "select_hotel",   "description": "Select best hotel", "depends_on": [1]},
    {"step_id": 3, "step_type": "build_day",      "description": "Day 1: Arrival", "depends_on": [1, 2]},
    {"step_id": 4, "step_type": "build_day",      "description": "Day 2: Explore", "depends_on": [1, 2]},
    {"step_id": 5, "step_type": "build_day",      "description": "Day 3: Final day", "depends_on": [1, 2]},
    {"step_id": 6, "step_type": "verify_budget",  "description": "Verify total cost", "depends_on": [1,2,3,4,5]}
  ]
}
"""


class ItineraryPlannerNode:
    """
    Outputs written to state:
        itinerary_plan.execution_plan  — ExecutionPlan dict
        itinerary_plan.filtered_bundle — pre-filtered data for executor
        itinerary_plan.observer_retries — initialised to 0
        itinerary_feasible             — bool
        itinerary_fallback_reason      — str | None
    """

    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm

    def __call__(self, state: AgentState) -> dict:
        destination   = state.get("destination_city", "")
        origin        = state.get("current_city", "")
        trip_days     = state.get("trip_days", 3)
        budget        = state.get("total_budget", 0)
        prefs         = state.get("user_preferences", {})
        flight_options = state.get("flight_options", [])
        raw_bundle    = state.get("itinerary_data_bundle", {})
        plan_state = state.get("itinerary_plan", {})
        current_retries = plan_state.get("observer_retries", 0)
        logger.info("ItineraryPlannerNode: %s→%s %dd $%s prefs=%s",
                    origin, destination, trip_days, budget, list(prefs.keys()))

        # ── 1. Filter data (no LLM) ────────────────────────────────────
        bundle = filter_bundle(raw_bundle, flight_options, prefs, budget, trip_days)

        # ── 2. Feasibility check (no LLM) ─────────────────────────────
        feasibility = check_feasibility(bundle, budget, trip_days)
        if not feasibility["feasible"]:
            return {
                "itinerary_plan": {"filtered_bundle": bundle, "observer_retries": 0},
                "itinerary_feasible": False,
                "itinerary_fallback_reason": feasibility["reason"],
            }

        # ── 3. Ask LLM for the execution plan only ─────────────────────
        slim_flights = [
            {"airline": f.get("airline"), "price": f.get("price"), "flight_number": f.get("flight_number")}
            for f in bundle.get("flights", [])
        ]
        
        slim_returns = [
            {"airline": f.get("airline"), "price": f.get("price"), "flight_number": f.get("flight_number")}
            for f in bundle.get("return_flights", [])
        ]
        
        slim_hotels = [
            {"name": h.get("name"), "stars": h.get("stars"), "price_per_night": h.get("price_per_night")}
            for h in bundle.get("hotels", [])
        ]

        context = f"""
Destination: {destination}
Origin: {origin}
Trip days: {trip_days}
Budget: ${budget or 'flexible'}
User preferences: {json.dumps(prefs, ensure_ascii=False)}

Outbound flights available ({len(slim_flights)}):
{json.dumps(slim_flights, ensure_ascii=False, indent=2)}

Return flights available ({len(slim_returns)}):
{json.dumps(slim_returns, ensure_ascii=False, indent=2)}

Hotels available ({len(slim_hotels)}):
{json.dumps(slim_hotels, ensure_ascii=False, indent=2)}

Activity names available (pick from these only):
{json.dumps([a.get('name') for a in bundle.get('activities', [])], ensure_ascii=False)}
"""
        raw = self.llm.invoke([
            SystemMessage(content=PLANNER_SYSTEM),
            HumanMessage(content=context),
        ]).content.strip()

        raw = strip_fences(raw)
        try:
            plan = ExecutionPlan(**json.loads(raw))
        except Exception as e:
            logger.warning("Planner JSON parse failed (%s) — using default plan", e)
            plan = default_plan(destination, origin, trip_days)
        print(plan.steps)
        return {
            "itinerary_plan": {
                "execution_plan": plan.model_dump(),
                "filtered_bundle": bundle,
                "observer_retries": current_retries,
            },
            "itinerary_feasible": True,
            "itinerary_fallback_reason": None,
        }