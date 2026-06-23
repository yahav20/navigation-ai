"""Replanner node — reviews execution results and decides to continue or finish."""
from typing import Any

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from agent.core.llm import silent
from agent.core.state import AgentState
from agent.advisor.planner import AdvisorPlan, PlannedToolCall, _step_to_args, _ALL_DESTINATION_TOOLS, _LIVE_EVENT_TOOLS
from agent.advisor.executor import _is_empty, format_tool_result
from ui import render_node, render_node_status

# Maximum total tool executions allowed in one advisor turn.
# Counts actual tool results, not replanner calls, so a legitimately large
# plan (e.g. 6 tools from the planner) is never cut short. The cap only
# fires when runaway replanning keeps adding new steps beyond what was planned.
MAX_TOOL_EXECUTIONS = 12

_DISCOVERY_TOOLS = frozenset({"find_destinations_by_vibe", "find_destinations_by_tag"})

# How many discovered cities to run concert searches for in mixed-intent queries
_MAX_CONCERT_CITIES = 3


def _inject_per_city_events(
    last_tool_name: str,
    remaining_plan: list[dict],
    past_results: list[dict],
) -> list[dict] | None:
    """Replace cityless live-event placeholders with per-city searches.

    Handles both search_concerts and search_special_events placeholders.
    Waits until ALL destination tools have run so multi-filter intersection
    is complete before selecting cities. Returns the updated plan or None.
    """
    if last_tool_name not in _ALL_DESTINATION_TOOLS:
        return None

    # Don't inject yet if more destination tools are still pending.
    if any(s["tool_name"] in _ALL_DESTINATION_TOOLS for s in remaining_plan):
        return None

    cityless_events = [
        s for s in remaining_plan
        if s["tool_name"] in _LIVE_EVENT_TOOLS and not s.get("args", {}).get("city")
    ]
    if not cityless_events:
        return None

    # Collect cities from all destination results that ran this turn.
    dest_results = [
        r for r in past_results
        if r["tool_name"] in _ALL_DESTINATION_TOOLS and not _is_empty(r.get("result"))
    ]
    if not dest_results:
        return None

    def _city_set(result: Any) -> set[str]:
        if not isinstance(result, list):
            return set()
        return {item["city"] for item in result if isinstance(item, dict) and "city" in item}

    if len(dest_results) == 1:
        cities = list(_city_set(dest_results[0]["result"]))[:_MAX_CONCERT_CITIES]
    else:
        # Multi-filter: intersect so only cities matching ALL criteria get events.
        sets = [_city_set(r["result"]) for r in dest_results]
        intersection = sets[0].intersection(*sets[1:])
        cities = (sorted(intersection)[:_MAX_CONCERT_CITIES]
                  if intersection else
                  list(_city_set(dest_results[0]["result"]))[:_MAX_CONCERT_CITIES])

    if not cities:
        return None

    # Expand each cityless placeholder into per-city searches.
    injected: list[dict] = []
    for placeholder in cityless_events:
        base_args = {k: v for k, v in placeholder.get("args", {}).items() if k != "city"}
        injected.extend(
            {"tool_name": placeholder["tool_name"], "args": {"city": city, **base_args}}
            for city in cities
        )

    # Keep non-event remaining steps (e.g. city-dive/practical) ahead of the new event steps.
    other = [s for s in remaining_plan
             if s["tool_name"] not in _LIVE_EVENT_TOOLS or s.get("args", {}).get("city")]
    return other + injected

# Only these two filter-discovery tools participate in intersection logic
_FILTER_DISCOVERY_TOOLS = frozenset({"find_destinations_by_vibe", "find_destinations_by_tag"})


def _extract_city_names(result: Any) -> set[str]:
    """Extract city names from a discovery tool result list."""
    if not isinstance(result, list):
        return set()
    return {item["city"] for item in result if isinstance(item, dict) and "city" in item}


def _build_intersection_note(tool_results: list[dict]) -> str:
    """When 2+ filter-discovery tools ran, return an INTERSECTION line for the data block.

    Returns an empty string when no intersection is applicable (< 2 filter tools ran).
    """
    filter_results = [
        r for r in tool_results
        if r["tool_name"] in _FILTER_DISCOVERY_TOOLS and not _is_empty(r["result"])
    ]
    if len(filter_results) < 2:
        return ""

    city_sets = [_extract_city_names(r["result"]) for r in filter_results]
    intersection = city_sets[0].intersection(*city_sets[1:])

    if not intersection:
        return (
            "INTERSECTION: No cities matched all selected filters — "
            "present the best partial matches from each list and note the trade-off to the user."
        )
    return (
        f"INTERSECTION (cities matching ALL user filters — present ONLY these): "
        f"{', '.join(sorted(intersection))}"
    )


class _RemainingPlan(BaseModel):
    """Remaining steps after pruning. Return an empty list to signal completion."""
    steps: list[PlannedToolCall] = Field(
        description=(
            "Steps still worth executing. Prune any that became redundant given "
            "what was already collected. Return an EMPTY list when the data collected "
            "so far is sufficient to answer the user's question."
        )
    )


def build_data_collected(tool_results: list[dict]) -> str:
    """Build the combined DATA COLLECTED block from all accumulated tool results."""
    usable = [r for r in tool_results if not _is_empty(r["result"])]
    if not usable:
        return "DATA COLLECTED:\n- No data gathered.\nREADY FOR FORMATTING."
    blocks = [
        format_tool_result(r["tool_name"], r.get("args", {}), r["result"])
        for r in usable
    ]
    data_block = "DATA COLLECTED:\n" + "\n\n".join(blocks)

    intersection_note = _build_intersection_note(usable)
    if intersection_note:
        data_block += f"\n\n{intersection_note}"

    return data_block + "\nREADY FOR FORMATTING."


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


class AdvisorReplannerNode:
    """Review execution progress and update the plan or signal completion."""

    def __init__(self, extraction_model: BaseChatModel) -> None:
        self.pruner = silent(extraction_model.with_structured_output(
            _RemainingPlan, method="function_calling"
        ))

    def __call__(self, state: AgentState) -> dict:
        render_node("advisor_replanner")
        plan = list(state.get("advisor_plan") or [])
        past_results = list(state.get("advisor_last_tool_results") or [])
        replan_count = state.get("advisor_replan_count") or 0
        messages = list(state.get("messages", []))

        remaining_plan = plan[1:] if plan else []

        # Mixed-intent injection: once all destination tools have run, replace any cityless
        # event placeholders with per-city searches — before any other replanning logic.
        if plan and past_results:
            injected = _inject_per_city_events(
                plan[0]["tool_name"],
                remaining_plan,
                past_results,
            )
            if injected is not None:
                event_steps = [s for s in injected if s["tool_name"] in _LIVE_EVENT_TOOLS]
                cities      = [s["args"].get("city") for s in event_steps]
                kinds       = sorted({s["tool_name"] for s in event_steps})
                render_node_status(f"[Replanner] Injecting {kinds} searches for: {cities}")
                return {
                    "advisor_plan": injected,
                    "advisor_replan_count": replan_count + 1,
                }

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

        tool_count = len(past_results)
        hit_cap    = tool_count >= MAX_TOOL_EXECUTIONS

        if (not remaining_plan and not needs_substitution_check) or hit_cap:
            if hit_cap:
                render_node_status(
                    f"[Replanner] Safety cap reached ({tool_count}/{MAX_TOOL_EXECUTIONS} tools). Forcing finish."
                )
            else:
                render_node_status("[Replanner] Plan complete. Proceeding to formatter.")
            return {
                "advisor_plan": [],
                "advisor_data_collected": build_data_collected(past_results),
                "advisor_replan_count": replan_count + 1,
            }

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
            f"<user_question>{user_question}</user_question>\n\n"
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
            decision = _RemainingPlan(steps=[])
            for s in remaining_plan:
                from agent.advisor.planner import PlannedToolCall
                decision.steps.append(PlannedToolCall(tool_name=s["tool_name"], **s.get("args", {})))

        if not decision.steps:
            render_node_status(f"[Replanner] Sufficient data collected. Proceeding to formatter.")
            return {
                "advisor_plan": [],
                "advisor_data_collected": build_data_collected(past_results),
                "advisor_replan_count": replan_count + 1,
            }

        new_plan = [
            {"tool_name": step.tool_name, "args": _step_to_args(step)}
            for step in decision.steps
        ]
        render_node_status(f"[Replanner] Continuing with: {[s['tool_name'] for s in new_plan]}")
        return {
            "advisor_plan": new_plan,
            "advisor_replan_count": replan_count + 1,
        }
