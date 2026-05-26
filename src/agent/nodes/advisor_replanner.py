"""Replanner node — reviews execution results and decides to continue or finish."""
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from agent.state import AgentState
from agent.nodes.advisor_planner import AdvisorPlan, PlannedToolCall, _step_to_args
from agent.nodes.advisor_executor import _is_empty, format_tool_result

MAX_REPLAN_STEPS = 5


# ---------------------------------------------------------------------------
# Decision schema
#
# Simpler than Union[Plan, Response]: the LLM returns the remaining steps to
# execute, pruned of anything that became redundant. An empty list means done.
# This avoids the model confusing "I need more data" with a finish signal.
# ---------------------------------------------------------------------------

class _RemainingPlan(BaseModel):
    """Remaining steps after pruning. Return an empty list to signal completion."""
    steps: list[PlannedToolCall] = Field(
        description=(
            "Steps still worth executing. Prune any that became redundant given "
            "what was already collected. Return an EMPTY list when the data collected "
            "so far is sufficient to answer the user's question."
        )
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_data_collected(tool_results: list[dict]) -> str:
    """Build the combined DATA COLLECTED block from all accumulated tool results."""
    if not tool_results:
        return "DATA COLLECTED:\n- No data gathered.\nREADY FOR FORMATTING."
    blocks = [
        format_tool_result(r["tool_name"], r.get("args", {}), r["result"])
        for r in tool_results
    ]
    return "DATA COLLECTED:\n" + "\n\n".join(blocks) + "\nREADY FOR FORMATTING."


_REPLANNER_PROMPT = """You are a travel advisor planning assistant reviewing execution progress.

Your job: return the remaining plan steps, pruning any that are now redundant.
Return an EMPTY steps list when the collected data is already sufficient.

PRUNE a remaining step when:
- It would return data already covered by what was collected
- It is clearly irrelevant given what was already found

KEEP a remaining step when:
- It adds new, distinct information needed to answer the question
- It covers a different city, season, or aspect not yet collected

Return EMPTY steps when:
- All planned steps have been executed
- The data is complete enough to answer the question without the remaining steps

NEVER add steps that were not in the remaining plan.
"""


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class AdvisorReplannerNode:
    """Review execution progress and update the plan or signal completion."""

    def __init__(self, extraction_model: BaseChatModel) -> None:
        self.pruner = extraction_model.with_structured_output(
            _RemainingPlan, method="function_calling"
        )

    def __call__(self, state: AgentState) -> dict:
        plan = list(state.get("advisor_plan") or [])
        past_results = list(state.get("advisor_last_tool_results") or [])
        replan_count = state.get("advisor_replan_count") or 0
        messages = list(state.get("messages", []))

        # plan[0] was just executed — compute remaining steps
        remaining_plan = plan[1:] if plan else []

        # --- Forced finish: plan exhausted or step-count guard triggered ---
        if not remaining_plan or replan_count >= MAX_REPLAN_STEPS:
            if replan_count >= MAX_REPLAN_STEPS:
                print(f"\n[Replanner] Step limit reached ({MAX_REPLAN_STEPS}). Forcing finish.")
            else:
                print(f"\n[Replanner] Plan complete. Proceeding to formatter.")
            return {
                "advisor_plan": [],
                "messages": [AIMessage(content=build_data_collected(past_results))],
                "advisor_replan_count": replan_count + 1,
            }

        # --- LLM-based pruning decision for remaining steps ---
        last_human = next(
            (m for m in reversed(messages) if getattr(m, "type", "") == "human"), None
        )
        user_question = last_human.content if last_human else ""

        past_summary = "\n".join(
            f"- {r['tool_name']}({r.get('args', {})}): "
            f"{'empty — no results' if _is_empty(r['result']) else 'data returned'}"
            for r in past_results
        )
        remaining_summary = "\n".join(
            f"- {s['tool_name']}({s.get('args', {})})"
            for s in remaining_plan
        )

        prompt = (
            f"Original question: {user_question}\n\n"
            f"Steps already executed:\n{past_summary}\n\n"
            f"Remaining steps:\n{remaining_summary}\n\n"
            "Return the remaining steps to execute (pruned of redundant ones), "
            "or an empty list if the data collected is already sufficient."
        )

        try:
            decision: _RemainingPlan = self.pruner.invoke([
                {"role": "system", "content": _REPLANNER_PROMPT},
                {"role": "user", "content": prompt},
            ])
        except Exception:  # noqa: BLE001
            # On failure, default to continuing with all remaining steps
            decision = _RemainingPlan(steps=[])
            for s in remaining_plan:
                from agent.nodes.advisor_planner import PlannedToolCall
                decision.steps.append(PlannedToolCall(tool_name=s["tool_name"], **s.get("args", {})))

        if not decision.steps:
            print(f"\n[Replanner] Sufficient data collected. Proceeding to formatter.")
            return {
                "advisor_plan": [],
                "messages": [AIMessage(content=build_data_collected(past_results))],
                "advisor_replan_count": replan_count + 1,
            }

        # Continue with (possibly pruned) remaining steps
        new_plan = [
            {"tool_name": step.tool_name, "args": _step_to_args(step)}
            for step in decision.steps
        ]
        print(f"\n[Replanner] Continuing with: {[s['tool_name'] for s in new_plan]}")
        return {
            "advisor_plan": new_plan,
            "advisor_replan_count": replan_count + 1,
        }
