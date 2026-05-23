"""
ItineraryObserverNode — reviews results, re-plans or renders final markdown.
Three outcomes: COMPLETE → summary | REPLAN → planner | hard-fail → fallback
"""
from __future__ import annotations
import logging
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.nodes.itinerary.schemas import (
    ExecutionPlan, FinalResponse, ObserverOutput, RevisedPlan,
)
from agent.state import AgentState

logger = logging.getLogger(__name__)
MAX_RETRIES = 3

OBSERVER_SYSTEM = """You are a travel plan quality reviewer.
Inspect the execution results summary and decide:

A) COMPLETE: all days built, budget OK, no errors.
   Output: {"result": {"status":"complete","markdown":"<full itinerary markdown>"}}
   The markdown MUST include:
   - Trip header with destination, total estimated cost
   - Outbound flight: airline, flight number, departure_time, arrival_time, price
   - Return flight: airline, flight number, departure_time, arrival_time, price
   - Hotel: name, stars, price per night, breakfast info
   - Day-by-day: for each day list every slot with time, name, duration, cost
   - Cost breakdown table (flight + return + hotel + activities + meals = total)
   Use ONLY values from the results summary. Never invent times or prices.

B) REPLAN: a fixable issue exists.
   Common reasons and suggested adjustments:
   - budget_exceeded → try {"trip_days": current_days - 1} or {"total_budget": budget + 500}
   - missing_day → add missing build_day_schedule step
   - tool_error → retry the failed fetch step
   Output: {"result": {"status":"replan","reason":"...","remaining_steps":[...],"adjustments":{...}}}
"""


class ItineraryObserverNode:
    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm.with_structured_output(ObserverOutput)

    def __call__(self, state: AgentState) -> dict:
        plan_state   = state.get("itinerary_plan", {})
        results      = plan_state.get("step_results", {})
        retry_count  = plan_state.get("retry_count", 0)
        budget       = state.get("total_budget", 0)
        trip_days    = state.get("trip_days", 3)

        if retry_count >= MAX_RETRIES:
            return {"itinerary_feasible": False,
                    "itinerary_fallback_reason": "max_retries_exceeded"}

        hard_fail = _check_hard_failures(results)
        if hard_fail:
            logger.info("Observer hard-fail: %s", hard_fail)
            return {"itinerary_feasible": False, "itinerary_fallback_reason": hard_fail}

        summary = _build_summary(results, budget, trip_days)

        try:
            output: ObserverOutput = self.llm.invoke([
                SystemMessage(content=OBSERVER_SYSTEM),
                HumanMessage(content=summary),
            ])
            result = output.result
        except Exception as e:
            logger.error("Observer LLM error: %s", e)
            result = RevisedPlan(status="replan", reason=f"observer_error:{e}",
                                  remaining_steps=[], adjustments={})

        if isinstance(result, FinalResponse):
            logger.info("Observer: COMPLETE")
            return {
                "itinerary_plan": {**plan_state, "final_markdown": result.markdown},
                "itinerary_feasible": True,
                "messages": [AIMessage(content=result.markdown)],
            }

        if isinstance(result, RevisedPlan):
            new_retry = retry_count + 1
            logger.info("Observer: REPLAN reason=%s retry=%d", result.reason, new_retry)
            if new_retry >= MAX_RETRIES:
                return {"itinerary_feasible": False,
                        "itinerary_fallback_reason": result.reason}

            updates: dict = {
                "itinerary_plan": {
                    **plan_state,
                    "retry_count": new_retry,
                    "observer_reason": result.reason,
                    "step_results": results,
                },
                "itinerary_feasible": False,
                "itinerary_fallback_reason": result.reason,
            }
            if "trip_days"    in result.adjustments:
                updates["trip_days"]    = result.adjustments["trip_days"]
            if "total_budget" in result.adjustments:
                updates["total_budget"] = result.adjustments["total_budget"]
            return updates

        return {"itinerary_feasible": False,
                "itinerary_fallback_reason": "observer_unexpected_output"}


def _check_hard_failures(results: dict) -> Optional[str]:
    for k, v in results.items():
        if k.startswith("fetch_flights"):
            if (isinstance(v, list) and not v) or (isinstance(v, dict) and v.get("error")):
                return "no_flights"
        if k.startswith("fetch_hotels"):
            if (isinstance(v, list) and not v) or (isinstance(v, dict) and v.get("error")):
                return "no_hotels"
    return None


def _build_summary(results: dict, budget: float, trip_days: int) -> str:
    lines = [f"Budget: ${budget or 'flexible'} | Trip days: {trip_days}\n"]
    for key, val in results.items():
        if isinstance(val, dict) and val.get("error"):
            lines.append(f"STEP {key}: ERROR — {val['error']}")
        elif key.startswith("fetch_flights") and isinstance(val, list):
            f = val[0] if val else None
            if f:
                lines.append(f"OUTBOUND FLIGHT: {f.get('airline')} {f.get('flight_number')} "
                             f"${f.get('price')} departs={f.get('departure_time','?')} "
                             f"arrives={f.get('arrival_time','?')}")
            else:
                lines.append("OUTBOUND FLIGHT: none found")
        elif key.startswith("fetch_return_flights") and isinstance(val, list):
            f = val[0] if val else None
            if f:
                lines.append(f"RETURN FLIGHT: {f.get('airline')} {f.get('flight_number')} "
                             f"${f.get('price')} departs={f.get('departure_time','?')} "
                             f"arrives={f.get('arrival_time','?')}")
        elif key.startswith("fetch_hotels") and isinstance(val, list):
            h = val[0] if val else None
            if h:
                lines.append(f"HOTEL: {h.get('name')} {h.get('stars')}* "
                             f"${h.get('price_per_night')}/night breakfast={h.get('breakfast_available')}")
            else:
                lines.append("HOTEL: none found")
        elif key.startswith("fetch_activities") and isinstance(val, list):
            lines.append(f"ACTIVITIES: {len(val)} available")
        elif key.startswith("build_day_schedule") and isinstance(val, dict):
            lines.append(f"DAY {val.get('day')}: {val.get('theme')} | "
                        f"{len(val.get('slots',[]))} slots | day_cost=${val.get('day_cost',0)}")
        elif key.startswith("verify_budget") and isinstance(val, dict):
            total = val.get("grand_total", 0)
            ok = not budget or total <= budget * 1.10
            lines.append(f"BUDGET CHECK: grand_total=${total} | ok={ok}")
    return "\n".join(lines)
