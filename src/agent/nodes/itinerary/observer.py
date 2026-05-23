# src/agent/nodes/itinerary/observer.py
from __future__ import annotations
import logging
import json
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from agent.nodes.itinerary.schemas import ExecutionPlan, ObserverOutput, FinalResponse, RevisedPlan
from agent.state import AgentState

logger = logging.getLogger(__name__)
MAX_RETRIES = 3

# === התוספות שהיו חסרות וגרמו לקריסה ===

OBSERVER_SYSTEM = """You are the Lead Travel Itinerary Reviewer.
Examine the provided execution results of the travel plan.
1. Check if the 'grand_total' cost from verify_budget exceeds the user's budget.
2. Ensure all requested days have a 'build_day_schedule' result.

If there is a SEVERE issue (like budget exceeded by more than 10%, or missing critical data), output a RevisedPlan with adjustments.
If the plan is good, output a FinalResponse with a beautiful, readable Markdown itinerary for the user.
"""

def _build_summary(results: dict, budget: float, trip_days: int) -> str:
    """פונקציה המייצרת טקסט עשיר עבור ה-LLM מתוך התוצאות שה-Executor אסף"""
    summary_lines = [f"Trip Details: {trip_days} Days | Budget: ${budget}"]
    for key, val in results.items():
        if isinstance(val, dict) and not val.get("error"):
            # מצמצם את הפלט כדי שה-LLM יוכל לקרוא אותו בקלות
            summary_lines.append(f"--- {key.upper()} ---")
            summary_lines.append(json.dumps(val, ensure_ascii=False))
    return "\n".join(summary_lines)

# =========================================

class ItineraryObserverNode:
    def __init__(self, llm: BaseChatModel) -> None:
         self.llm = llm.with_structured_output(ObserverOutput)

    def __call__(self, state: AgentState) -> dict:
        plan_state    = state.get("itinerary_plan", {})
        results       = plan_state.get("step_results", {})
        retry_count   = plan_state.get("retry_count", 0)
        current_index = state.get("current_step_index", 0)
        plan_steps    = plan_state.get("execution_plan", {}).get("steps", [])

        # 1. בדיקה האם הצעד האחרון נכשל (Replanner Trigger)
        if current_index > 0:
             last_step = plan_steps[current_index - 1]
             last_key = f"{last_step['step_type']}_{last_step['step_id']}"
             last_result = results.get(last_key, {})
             
             if isinstance(last_result, dict) and last_result.get("error"):
                  error_msg = last_result["error"]
                  print(f"\n--- ⚠️ OBSERVER: Step failure detected: {error_msg} ---")
                  print("--- 🔄 TRIGGERING REPLANNER ---")
                  
                  new_retry = retry_count + 1
                  if new_retry >= MAX_RETRIES:
                       return {"itinerary_feasible": False, "itinerary_fallback_reason": "max_retries_exceeded"}
                  
                  return {
                      "itinerary_feasible": False, 
                      "itinerary_fallback_reason": error_msg,
                      "itinerary_plan": {**plan_state, "retry_count": new_retry, "observer_reason": error_msg}
                  }

        # 2. אם לא סיימנו את התוכנית, ממשיכים לצעד הבא
        if current_index < len(plan_steps):
             return {"itinerary_feasible": True, "observer_action": "continue"}

        # 3. סיימנו את כל הצעדים בהצלחה! מעבירים ל-LLM לבדיקה סופית ועיצוב
        print("\n--- 🧐 OBSERVER: All steps executed. Performing final holistic review... ---")
        budget    = state.get("total_budget", 0)
        trip_days = state.get("trip_days", 3)
        
        try:
            summary = _build_summary(results, budget, trip_days) 
            output = self.llm.invoke([
                SystemMessage(content=OBSERVER_SYSTEM),
                HumanMessage(content=summary),
            ])
            # חילוץ התוצאה מהמודל המובנה של LangChain
            result = getattr(output, "result", output)
        except Exception as e:
             # התיקון הקריטי: הדפסה בולטת של השגיאה כדי שלא "תיבלע" ותבזבז לך טוקנים!
             print(f"\n❌ OBSERVER CODE CRASHED DURING REVIEW: {e}")
             return {"itinerary_feasible": False, "itinerary_fallback_reason": f"Observer Code Crash: {e}"}

        # אם ה-LLM קובע שיש חריגת תקציב ודורש תכנון מחדש
        if isinstance(result, RevisedPlan):
             print(f"\n--- ⚠️ OBSERVER: Holistic issue detected: {result.reason} ---")
             print("--- 🔄 TRIGGERING REPLANNER ---")
             new_retry = retry_count + 1
             if new_retry >= MAX_RETRIES:
                  return {"itinerary_feasible": False, "itinerary_fallback_reason": result.reason}
                  
             updates = {
                 "itinerary_feasible": False,
                 "itinerary_fallback_reason": result.reason,
                 "itinerary_plan": {**plan_state, "retry_count": new_retry, "observer_reason": result.reason}
             }
             if hasattr(result, "adjustments"):
                 if "trip_days" in result.adjustments:
                      updates["trip_days"] = result.adjustments["trip_days"]
                 if "total_budget" in result.adjustments:
                      updates["total_budget"] = result.adjustments["total_budget"]
             return updates

        # אם הגענו לכאן - ה-LLM אישר את התוכנית לחלוטין
        print("\n--- 🎉 OBSERVER: Plan approved! Generating final markdown. ---")
        
        markdown_output = getattr(result, "markdown", str(result))
        
        return {
            "itinerary_plan": {**plan_state, "final_markdown": markdown_output},
            "itinerary_feasible": True,
            "observer_action": "complete",
            "messages": [AIMessage(content=markdown_output)],
        }