"""
ItineraryExecutorNode — v3
==========================
Key improvements to meet production standards:

AUTONOMOUS REACT LOOP (Internal)
  - The Executor acts as a self-contained ReAct agent within a single LangGraph node.
  - It binds available tools and runs a Thought -> Action -> Observation loop up to 5 times.
  - Prevents graph-level infinite loops by containing tool-execution complexity locally.

STRICT STRUCTURED OUTPUT
  - After tool execution, the agent's message trace is passed to an extraction model.
  - Guarantees the output schema: `status`, `data`, `error`, `replan_hint`, `trace`.
  - Seamlessly integrates with the Observer's `_unwrap()` expectation.

STATE MANAGEMENT
  - Increments `current_step_index` by 1 upon completion so the Observer can validate
    the newly executed step.
  - Appends the wrapped result to `step_results` using the unique `step_type_step_id` key.

FAILURE ISOLATION
  - Catches Python exceptions natively during tool calls and formats them as `failed` statuses.
  - Automatically generates `replan_hints` if a tool crashes or returns empty data.
"""
from __future__ import annotations
from agent.nodes.itinerary.activity_selector import select_activities_per_day
import json
from typing import Any, Literal, Union
from agent.nodes.itinerary.schemas import StepExecutionResult
from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from agent.state import AgentState

# Note: Adjust the import path to wherever your specific itinerary tools are defined.
from .itinerary_tools import ITINERARY_TOOLS 

MAX_TOOL_TURNS = 5



# ── System Prompts ────────────────────────────────────────────────────────

EXECUTOR_SYSTEM = """
You are an autonomous ReAct Executor Agent within a robust travel planning system.
Your goal is to execute a SINGLE specific step from a larger travel execution plan.

You have access to a suite of travel tools. 
Follow the ReAct loop:
1. THOUGHT: Analyze the user context, system state, and the specific Step task.
2. ACTION: Call the necessary tool(s) to fulfill the task.
3. OBSERVATION: Review the tool outputs.

CRITICAL RULES:
- Only execute the tools required for THIS specific step. Do not skip ahead.
- Avoid repeating the exact same tool call with the exact same arguments if it failed previously.
- When you have successfully gathered the data or determined it is unavailable, summarize your findings.
"""

EXTRACTION_SYSTEM = """
You are a data extraction parser. 
Review the preceding conversation trace between the ReAct Agent and the Tools.
Extract the final outcome into the StepExecutionResult schema.

RULES:
- `data` MUST be a REAL JSON object or REAL JSON array.
- NEVER return JSON as a quoted string.
- NEVER serialize arrays or objects into text.
- Example VALID:
  "data": [{"flight":"AF123"}]

- Example INVALID:
  "data": "[{\"flight\":\"AF123\"}]"
- If no flights/hotels/activities were found, or a tool crashed, status is "failed". Provide a clear `error` and a `replan_hint`.
- The `trace` should be a 1-2 sentence summary of what the agent did.
"""

# ── Node ───────────────────────────────────────────────────────────────────

class ItineraryExecutorNode:
    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm
        self.tools_map: dict[str, BaseTool] = {t.name: t for t in ITINERARY_TOOLS}
        self.llm_with_tools = self.llm.bind_tools(ITINERARY_TOOLS)
        self.extractor = self.llm.with_structured_output(StepExecutionResult)

    def __call__(self, state: AgentState) -> dict:
        plan_state = state.get("itinerary_plan", {})
        execution_plan = plan_state.get("execution_plan", {})
        steps = execution_plan.get("steps", [])
        current_index = state.get("current_step_index", 0)
        step_results = plan_state.get("step_results", {})

        # ── 1. Bounds Check ────────────────────────────────────────────────
        if current_index >= len(steps):
            # No steps left; Observer will handle completion
            return {}

        current_step = steps[current_index]
        step_type = current_step.get("step_type", "unknown")
        step_id = current_step.get("step_id", 0)
        day_tag = f" (Day {current_step.get('day')})" if current_step.get("day") else ""
        step_key = f"{step_type}_{step_id}"

        destination = execution_plan.get("destination", state.get("destination_city", ""))
        origin = execution_plan.get("origin", state.get("current_city", ""))
        budget = state.get("total_budget", 0)
        trip_days = state.get("trip_days", 3)

        # ── 2. Build Execution Context ─────────────────────────────────────
        task_context = (
            f"🎯 CURRENT TASK: {step_type}{day_tag}\n"
            f"Description: {current_step.get('description', 'No description provided.')}\n\n"
            f"System State:\n"
            f"- Origin: {origin}\n"
            f"- Destination: {destination}\n"
            f"- Trip Days: {trip_days}\n"
            f"- Total Budget: ${budget}\n\n"
            f"Prior Step Results Summary (Context):\n"
            f"{self._summarize_prior_results(step_results)}"
        )

        messages = [
            SystemMessage(content=EXECUTOR_SYSTEM),
            HumanMessage(content=task_context)
        ]

        # ── 3. Internal ReAct Loop ─────────────────────────────────────────
        turn_count = 0
        hard_crash_error = None

        while turn_count < MAX_TOOL_TURNS:
            try:
                response = self.llm_with_tools.invoke(messages)
                messages.append(response)
            except Exception as e:
                hard_crash_error = f"LLM execution failed: {str(e)}"
                break

            # If the LLM didn't call any tools, it considers the task complete
            if not getattr(response, "tool_calls", None):
                break

            # Execute tools securely
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                try:
                    tool = self.tools_map[tool_name]
                    tool_output = tool.invoke(tool_args)
                    messages.append(ToolMessage(
                        tool_call_id=tool_call["id"],
                        name=tool_name,
                        content=json.dumps(tool_output, ensure_ascii=False)
                    ))
                except Exception as e:
                    messages.append(ToolMessage(
                        tool_call_id=tool_call["id"],
                        name=tool_name,
                        content=f"Error executing {tool_name}: {str(e)}"
                    ))

            turn_count += 1

        # ── 4. Structured Extraction ───────────────────────────────────────
        if hard_crash_error:
            final_result = StepExecutionResult(
                status="failed",
                error=hard_crash_error,
                replan_hint="Executor crashed mid-operation. Please simplify constraints or retry.",
                trace="System encountered an unhandled exception during the ReAct loop."
            )
        else:
            messages.append(SystemMessage(content=EXTRACTION_SYSTEM))
            try:
                final_result: StepExecutionResult = self.extractor.invoke(messages)
                
                
                        
            except Exception as e:
                final_result = StepExecutionResult(
                    status="failed",
                    error=f"Extraction formatting failed: {str(e)}",
                    replan_hint="Tool succeeded but formatting failed. Retry the step.",
                    trace="Failed at output parsing phase."
                )

        # =================================================================
        # 🧠 THE SMART ACTIVITY SELECTOR INTEGRATION
        # =================================================================
        if final_result.status == "success" and step_type == "fetch_activities":
            raw_activities = final_result.data
            
            if isinstance(raw_activities, list) and len(raw_activities) > 0:
                # קריאה לקובץ שלך! מחלקים את האטרקציות לימים בעזרת ה-LLM
                grouped_days = select_activities_per_day(
                    llm=self.llm,
                    activities=raw_activities,
                    trip_days=trip_days,
                    prefs=state.get("user_preferences", {}),
                    destination=destination
                )
                
                final_result.data = {
                    "grouped_by_day": grouped_days,
                    "raw_pool": raw_activities
                }

        # ── 5. State Update ────────────────────────────────────────────────
        step_results[step_key] = final_result.model_dump()

        log_msg = (
            f"⚙️ **EXECUTOR:** Completed `{step_type}`\n"
            f"*Status:* `{final_result.status}` | *Trace:* {final_result.trace}"
        )
        if final_result.status == "failed":
            log_msg += f"\n*Error:* {final_result.error}"

        return {
            "current_step_index": current_index + 1,  # Advance index so Observer tests THIS step
            "itinerary_plan": {
                **plan_state,
                "step_results": step_results
            },
            "messages": [AIMessage(content=log_msg, name="executor_log")]
        }

    def _summarize_prior_results(self, step_results: dict) -> str:
        """Provide lightweight context of previous steps without blowing up the context window."""
        if not step_results:
            return "No previous steps executed yet."
        
        summary_lines = []
        for key, res in step_results.items():
            if isinstance(res, dict) and res.get("status") == "success":
                # Only pass keys to show it was done, avoiding massive JSON strings
                summary_lines.append(f"- {key}: Completed successfully.")
            elif isinstance(res, dict) and res.get("status") == "failed":
                summary_lines.append(f"- {key}: FAILED. Error: {res.get('error')}")
        
        return "\n".join(summary_lines)
 