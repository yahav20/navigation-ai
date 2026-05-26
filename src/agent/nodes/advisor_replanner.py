"""Replanner node — reviews execution results and decides to continue or finish."""
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from agent.state import AgentState
from agent.nodes.advisor_planner import AdvisorPlan, PlannedToolCall, _step_to_args
from agent.nodes.advisor_executor import _is_empty, format_tool_result
from ui import render_node, render_node_status

MAX_REPLAN_STEPS = 5

_DISCOVERY_TOOLS = frozenset({"find_destinations_by_vibe", "find_destinations_by_tag"})


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

ADDING NEW STEPS — ONE EXCEPTION ONLY:
Never add steps that were not in the remaining plan, EXCEPT for this case:

Discovery tool substitution: if find_destinations_by_vibe OR find_destinations_by_tag
returned empty (no results), you MAY insert the sibling discovery tool as the next step.
  - find_destinations_by_vibe returned empty → add find_destinations_by_tag with a semantically close tag
  - find_destinations_by_tag returned empty → add find_destinations_by_vibe with a semantically close category

Available categories (find_destinations_by_vibe): Culture, Entertainment, Family, History, Nature, Nightlife, Sightseeing
Available tags (find_destinations_by_tag): alternative, art, beach, budget-friendly, canals, city-break,
  cultural, cycling, entertainment, expensive, fashion, foodie, historic, iconic, liberal, luxury,
  mediterranean, modern, multicultural, nature-nearby, nightlife, rainy, romantic, safe, shopping,
  student-friendly, sunny, technology, theatre, unique, walkable

Substitution examples:
  category="Nature"      → tag="nature-nearby"
  category="Family"      → tag="safe"
  category="Culture"     → tag="cultural"
  category="History"     → tag="historic"
  category="Nightlife"   → tag="nightlife"
  category="Sightseeing" → tag="iconic"
  tag="romantic"         → category="Sightseeing"
  tag="beach"            → category="Sightseeing"
  tag="foodie"           → category="Entertainment"
  tag="nightlife"        → category="Nightlife"
  tag="eco-friendly"     → category="Nature"

Only substitute once. If the substitute also returned empty, return an empty steps list — do not keep trying.
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
        render_node("advisor_replanner")
        plan = list(state.get("advisor_plan") or [])
        past_results = list(state.get("advisor_last_tool_results") or [])
        replan_count = state.get("advisor_replan_count") or 0
        messages = list(state.get("messages", []))

        # plan[0] was just executed — compute remaining steps
        remaining_plan = plan[1:] if plan else []

        # --- Check whether the just-executed step was a discovery tool that returned empty.
        # If so, the LLM must get a chance to substitute the sibling even if no steps remain.
        needs_substitution_check = False
        if plan and past_results:
            last_executed_name = plan[0]["tool_name"]
            last_result_entry  = past_results[-1]
            sibling = {"find_destinations_by_vibe": "find_destinations_by_tag",
                       "find_destinations_by_tag":  "find_destinations_by_vibe"}.get(last_executed_name)
            sibling_already_ran = any(r["tool_name"] == sibling for r in past_results) if sibling else True
            if (last_executed_name in _DISCOVERY_TOOLS
                    and last_result_entry["tool_name"] == last_executed_name
                    and _is_empty(last_result_entry["result"])
                    and not sibling_already_ran):
                needs_substitution_check = True
                render_node_status(f"[Replanner] '{last_executed_name}' returned empty — "
                                   f"checking whether to substitute sibling tool.")

        # --- Forced finish: plan exhausted with no substitution needed, or step-count guard ---
        if (not remaining_plan and not needs_substitution_check) or replan_count >= MAX_REPLAN_STEPS:
            if replan_count >= MAX_REPLAN_STEPS:
                render_node_status(f"[Replanner] Step limit reached ({MAX_REPLAN_STEPS}). Forcing finish.")
            else:
                render_node_status(f"[Replanner] Plan complete. Proceeding to formatter.")
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

        substitution_instruction = (
            "\n\nIMPORTANT: The last discovery tool returned empty and no remaining steps exist. "
            "You MUST insert the sibling discovery tool with a semantically appropriate argument. "
            "Do NOT return an empty list here — that would leave the user with no results."
            if needs_substitution_check and not remaining_plan else ""
        )

        prompt = (
            f"Original question: {user_question}\n\n"
            f"Steps already executed:\n{past_summary}\n\n"
            f"Remaining steps:\n{remaining_summary if remaining_summary else '(none)'}\n\n"
            "Return the remaining steps to execute (pruned of redundant ones), "
            "or an empty list if the data collected is already sufficient."
            f"{substitution_instruction}"
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
            render_node_status(f"[Replanner] Sufficient data collected. Proceeding to formatter.")
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
        render_node_status(f"[Replanner] Continuing with: {[s['tool_name'] for s in new_plan]}")
        return {
            "advisor_plan": new_plan,
            "advisor_replan_count": replan_count + 1,
        }
