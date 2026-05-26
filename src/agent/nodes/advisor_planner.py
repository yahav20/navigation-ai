"""Planner node — decides which tools to call and in what order for an advisor query."""
from typing import Literal

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from agent.state import AgentState


# ---------------------------------------------------------------------------
# Follow-up detection
# ---------------------------------------------------------------------------

class _RelevanceCheck(BaseModel):
    is_followup: bool


_RELEVANCE_PROMPT = """Determine if the new user question is a follow-up to the previous
conversation, or a completely fresh, unrelated question.

Follow-up (is_followup=True) — ALL of the following must be true:
- References a SPECIFIC CITY or topic already discussed (e.g. "tell me more about Paris", "when is Berlin cold?")
- Uses explicit back-references: "tell me more", "what about...", "also", "and what about X", "of those", "from the list"
- Clearly continues the EXACT SAME travel scenario with no change in goal

New question (is_followup=False) — any of these is enough:
- A DESTINATION DISCOVERY question ("where should I go?", "recommend me somewhere", "where can I travel?",
  "where should I fly?", "what's a good destination?") → ALWAYS False, even if origin/budget were mentioned before
- Changes the TYPE of query (e.g. previous was city info; now asking for destination recommendations)
- No explicit reference to anything discussed before
- Could stand completely alone without the previous context

When in doubt, return False."""


def _is_followup(extraction_model: BaseChatModel, summary: str, new_question: str) -> bool:
    """Return True only if the new question clearly continues the previous topic."""
    if not summary:
        return False
    result: _RelevanceCheck = extraction_model.with_structured_output(_RelevanceCheck).invoke([
        {"role": "system", "content": _RELEVANCE_PROMPT},
        {"role": "user", "content": f"Previous conversation summary:\n{summary}\n\nNew question: {new_question}"},
    ])
    return result.is_followup


# ---------------------------------------------------------------------------
# Plan models — explicit fields so the LLM schema is unambiguous
# ---------------------------------------------------------------------------

ToolName = Literal[
    "find_destinations_by_vibe",
    "find_destinations_by_tag",
    "get_reachable_destinations",
    "find_destinations_within_budget_auto",
    "find_destinations_within_budget",
    "get_city_overview",
    "get_trip_duration_advisor",
    "fetch_activities",
    "get_best_time_to_visit",
    "get_average_weather",
]


class PlannedToolCall(BaseModel):
    tool_name: ToolName = Field(description="Exact name of the tool to call")
    city: str | None = Field(default=None, description=(
        "City name — required for get_city_overview, get_trip_duration_advisor, "
        "fetch_activities, get_best_time_to_visit, get_average_weather"
    ))
    tag: str | None = Field(default=None, description=(
        "City tag — required for find_destinations_by_tag. "
        "Must be one of the AVAILABLE CITY TAGS."
    ))
    category: str | None = Field(default=None, description=(
        "Activity category — required for find_destinations_by_vibe. "
        "Must be one of the AVAILABLE ACTIVITY CATEGORIES."
    ))
    origin: str | None = Field(default=None, description=(
        "Origin city — required for get_reachable_destinations, "
        "find_destinations_within_budget, find_destinations_within_budget_auto"
    ))
    total_budget: float | None = Field(default=None, description=(
        "Total trip budget in USD — required for find_destinations_within_budget "
        "and find_destinations_within_budget_auto"
    ))
    trip_days: int | None = Field(default=None, description=(
        "Explicit trip duration in days — required for find_destinations_within_budget only"
    ))
    max_flight_hours: float | None = Field(default=None, description=(
        "Max flight hours for get_reachable_destinations. "
        "Omit when no flight-length constraint. short=2.5, medium=5, long=9"
    ))
    season: str | None = Field(default=None, description=(
        "Season for get_average_weather. Must be exactly: Spring, Summer, Autumn, or Winter"
    ))


class AdvisorPlan(BaseModel):
    steps: list[PlannedToolCall] = Field(description="Ordered list of tool calls to execute, maximum 3")


# ---------------------------------------------------------------------------
# Convert a PlannedToolCall to the args dict the executor expects
# ---------------------------------------------------------------------------

_ARG_FIELDS = ("city", "tag", "category", "origin", "total_budget", "trip_days",
               "max_flight_hours", "season")


def _step_to_args(step: PlannedToolCall) -> dict:
    return {f: getattr(step, f) for f in _ARG_FIELDS if getattr(step, f) is not None}


_BUDGET_TOOLS = frozenset({"find_destinations_within_budget", "find_destinations_within_budget_auto"})
_DISCOVERY_TOOLS = frozenset({"find_destinations_by_tag", "find_destinations_by_vibe"})


def _sanitize_plan(steps: list[PlannedToolCall]) -> list[PlannedToolCall]:
    """Enforce hard architectural constraints the LLM occasionally ignores.

    Rule: when a budget tool is present, discovery tools are redundant and misleading
    (they return cities without cost/reachability filtering). Remove them.
    """
    tool_names = {s.tool_name for s in steps}
    if tool_names & _BUDGET_TOOLS:
        steps = [s for s in steps if s.tool_name not in _DISCOVERY_TOOLS]
    return steps


# ---------------------------------------------------------------------------
# Constants (same values used by the tool implementations)
# ---------------------------------------------------------------------------

_AVAILABLE_CATEGORIES = "Culture, Entertainment, Family, History, Nature, Nightlife, Sightseeing"

_AVAILABLE_TAGS = (
    "alternative, art, beach, budget-friendly, canals, city-break, cultural, cycling, "
    "entertainment, expensive, fashion, foodie, historic, iconic, liberal, luxury, "
    "mediterranean, modern, multicultural, nature-nearby, nightlife, rainy, romantic, "
    "safe, shopping, student-friendly, sunny, technology, theatre, unique, walkable"
)

_VIBE_MAPPING = """
Use find_destinations_by_TAG for atmosphere/lifestyle words — set the `tag` field:
  romantic, beach, luxury, budget-friendly, foodie, nightlife, historic, modern,
  cycling, shopping, scenic, unique, safe, sunny, walkable, canals, rainy, etc.

Use find_destinations_by_VIBE for activity-type words — set the `category` field:
  Nature, Culture, History, Family, Nightlife, Entertainment, Sightseeing

Examples:
  "beach destination"      -> find_destinations_by_tag,  tag="beach"
  "romantic getaway"       -> find_destinations_by_tag,  tag="romantic"
  "museum / cultural trip" -> find_destinations_by_vibe, category="Culture"
  "family with kids"       -> find_destinations_by_vibe, category="Family"
  "budget travel"          -> find_destinations_by_tag,  tag="budget-friendly"
  "nature / hiking"        -> find_destinations_by_vibe, category="Nature"
  "nightlife"              -> both find_destinations_by_tag (tag="nightlife")
                              AND find_destinations_by_vibe (category="Nightlife")
"""

# ---------------------------------------------------------------------------
# Helper: format last-turn tool results for the planner prompt
# ---------------------------------------------------------------------------

def _format_last_results(results: list[dict]) -> str:
    """Summarise advisor_last_tool_results into a compact, human-readable string."""
    lines = []
    for r in results:
        tool_name = r.get("tool_name", "?")
        args = r.get("args", {})
        result = r.get("result")
        args_str = ", ".join(f'{k}="{v}"' for k, v in args.items())

        if isinstance(result, list) and result:
            if "message" in result[0]:
                lines.append(f"- {tool_name}({args_str}) -> no results")
            else:
                cities = [item["city"] for item in result if "city" in item]
                if cities:
                    sample = ", ".join(cities[:6])
                    lines.append(f"- {tool_name}({args_str}) -> {sample}")
                else:
                    lines.append(f"- {tool_name}({args_str}) -> {len(result)} results")
        elif isinstance(result, dict):
            if "message" in result:
                lines.append(f"- {tool_name}({args_str}) -> no results")
            else:
                lines.append(f"- {tool_name}({args_str}) -> data fetched")
        else:
            lines.append(f"- {tool_name}({args_str}) -> no results")

    return "\n".join(lines) if lines else "None."


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM_PROMPT = """You are a travel advisor planning assistant.
Your ONLY job is to decide which tools to call — and in what order — to answer the user's travel question.
You do NOT answer the question yourself. An executor will run the tools and a formatter will write the response.

KNOWN USER CONTEXT (treat as confirmed facts — use directly in tool fields):
{state_context}

CONTEXT FROM PREVIOUS EXCHANGES:
{summary}

CITIES ALREADY SHOWN TO THE USER:
{shown_cities}

TOOLS ALREADY CALLED LAST TURN (avoid re-fetching unless the question specifically asks for fresh or different data):
{last_tool_results}

AVAILABLE ACTIVITY CATEGORIES (valid values for the `category` field):
{categories}

AVAILABLE CITY TAGS (valid values for the `tag` field):
{tags}

VIBE -> TOOL MAPPING:
{vibe_mapping}

TOOL REFERENCE — WHEN TO USE EACH AND WHICH FIELDS TO SET:

- find_destinations_by_vibe
    Set: category = one of the AVAILABLE ACTIVITY CATEGORIES
    -> For activity-type queries: Nature, Culture, History, Family, Nightlife, Sightseeing
    -> Pick the single closest category

- find_destinations_by_tag
    Set: tag = one of the AVAILABLE CITY TAGS
    -> For atmosphere, lifestyle, or vibe queries: romantic, beach, foodie, budget-friendly, etc.
    -> Pick the single closest tag

- find_destinations_within_budget_auto
    Set: origin = <city>, total_budget = <number>
    -> Use ONLY when trip_days is NOT in KNOWN USER CONTEXT and the user has NOT stated a duration
    -> Automatically uses each city's recommended minimum stay to calculate cost

- find_destinations_within_budget
    Set: origin = <city>, total_budget = <number>, trip_days = <int>
    -> Use ONLY when the user has explicitly stated a trip duration (or it's in KNOWN USER CONTEXT)

- get_reachable_destinations
    Set: origin = <city>, max_flight_hours = <number or omit>
    -> Use ONLY when the user explicitly states an origin AND has no budget
    -> DO NOT use if the user mentioned a budget — use a budget tool instead
    -> max_flight_hours values: short haul=2.5, medium haul=5, long haul=9; omit for no constraint

- get_city_overview
    Set: city = <city>
    -> When the user asks about a specific destination in general ("what is Tokyo like?", "when to visit Paris?")
    -> Returns activity types, best visit months, and seasonal temperatures in one call

- get_trip_duration_advisor
    Set: city = <city>
    -> When the user asks "how many days should I spend in X?" or "is N days enough for X?"

- fetch_activities
    Set: city = <city>
    -> ONLY when the user names a specific city AND explicitly asks what to do there
    -> NEVER plan this for "where should I go?" type questions

- get_best_time_to_visit
    Set: city = <city>
    -> When the user asks specifically about the best time to visit a known destination

- get_average_weather
    Set: city = <city>, season = <Spring|Summer|Autumn|Winter>
    -> When the user asks about weather in a specific season for a specific city

COMBINING TOOLS — WHICH TOOLS GO TOGETHER:

RULE 0 — BUDGET WITHOUT ORIGIN (check this FIRST before any other rule):
   If the user mentions a budget but NO origin city is known:
   -> The budget tools CANNOT work — they require an origin and will fail without one
   -> Plan ONLY: find_destinations_by_tag or find_destinations_by_vibe based on any vibe mentioned
   -> If no vibe mentioned: find_destinations_by_tag (tag="city-break")
   -> NEVER call find_destinations_within_budget_auto or find_destinations_within_budget
   -> This rule overrides everything else. A budget mention without an origin triggers vibe/tag tools.

1. User mentions origin + budget (regardless of any vibe preference), NO trip_days:
   -> Plan EXACTLY ONE tool: find_destinations_within_budget_auto (origin=<city>, total_budget=<n>)
   -> This applies EVEN IF the user also says "romantic", "beach", or any other vibe — budget overrides vibe
   -> PROHIBITED: find_destinations_by_tag, find_destinations_by_vibe, get_reachable_destinations
   -> The budget tool already filters by reachability and cost — adding vibe tools introduces out-of-budget cities

2. User mentions origin + budget + explicit trip_days:
   -> Plan EXACTLY ONE tool: find_destinations_within_budget (origin=<city>, total_budget=<n>, trip_days=<d>)
   -> Same absolute prohibitions as case 1 above

3. User mentions origin + a clear destination vibe/tag AND no budget:
   -> Plan TWO tools: get_reachable_destinations (origin=<city>) AND find_destinations_by_tag or find_destinations_by_vibe
   -> Use this ONLY when the user states a vibe preference (romantic, beach, foodie, etc.) with NO budget

4. User mentions origin + flight length preference only (no budget, no specific destination vibe):
   -> Plan EXACTLY ONE tool: get_reachable_destinations (origin=<city>, max_flight_hours=<n>)
   -> "Short flights", "hate long flights", "close destinations" all map here — no vibe tool needed
   -> DO NOT add find_destinations_by_tag or find_destinations_by_vibe

5. User mentions origin only, no budget, no specific vibe:
   -> Plan EXACTLY ONE tool: get_reachable_destinations (origin=<city>)
   -> DO NOT add find_destinations_by_tag or find_destinations_by_vibe

6. User mentions vibe/tag only, NO origin, NO budget:
   -> Plan EXACTLY ONE tool: find_destinations_by_tag or find_destinations_by_vibe
   -> Do NOT plan get_reachable_destinations — you have no origin to use

7. General destination ideas with no vibe, no origin, no budget:
   -> Plan: find_destinations_by_tag (tag="city-break") OR find_destinations_by_vibe (category="Sightseeing")

8. User asks about a specific city (overview / general info / "tell me everything"):
   -> Plan EXACTLY ONE tool: get_city_overview (city=<city>)
   -> get_city_overview already returns activity types, best visit months, AND seasonal temperatures
   -> DO NOT also call get_best_time_to_visit — it is completely redundant when get_city_overview is called
   -> Add fetch_activities as a second tool ONLY if the user explicitly asks "what can I do there?"
   -> Maximum 2 tools total for city-overview questions

9. User asks how many days for a city:
   -> Plan: get_trip_duration_advisor (city=<city>) for each city mentioned, no other tools

10. User asks "what budget do I need?" or "how much would it cost?":
    -> If trip_days is known: find_destinations_within_budget (origin=<city>, total_budget=99999, trip_days=<d>)
    -> If trip_days unknown: find_destinations_within_budget_auto (origin=<city>, total_budget=99999)

11. User asks about family trip:
    -> Plan: find_destinations_by_vibe (category="Family") — exactly 1 tool is usually enough

COUNTRY -> CITY RESOLUTION:
Apply this mapping to both KNOWN USER CONTEXT and the user's message:
  "Israel" -> "Tel Aviv"
  "UK" / "England" / "Britain" -> "London"
  "Japan" -> "Tokyo"
  "France" -> "Paris"
  "Germany" -> "Berlin"
  "Netherlands" / "Holland" -> "Amsterdam"

PLANNING DISCIPLINE — CRITICAL:
- Maximum 3 tool calls. Usually 1-2 is enough.
- A simple question (weather, duration, single-city overview) needs exactly 1 tool call.
- Do NOT plan fetch_activities for destination advisor questions.
- Do NOT plan get_reachable_destinations if you have no origin city to use.
- Do NOT plan tools whose required fields (origin, city) are completely unknown — except when a
  sensible default exists (e.g. find_destinations_by_tag with tag="city-break" needs no origin).
- You MUST always plan at least one tool call.
- Never plan to ask the user a question — just choose the best tool(s) with the available context.
"""


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class AdvisorPlannerNode:
    """Generate an ordered tool-call plan from the user's advisory question."""

    def __init__(self, extraction_model: BaseChatModel) -> None:
        self.planner = extraction_model.with_structured_output(AdvisorPlan, method="function_calling")
        self.extraction_model = extraction_model

    def __call__(self, state: AgentState) -> dict:
        summary = state.get("summary", "")
        messages = list(state.get("messages", []))

        last_human = next(
            (m for m in reversed(messages) if getattr(m, "type", "") == "human"), None
        )
        if not last_human:
            return {"advisor_plan": []}

        # Fresh questions should not bleed context from previous tool turns
        is_fresh = not _is_followup(self.extraction_model, summary, last_human.content)

        ctx_parts = []
        if state.get("current_city"):
            ctx_parts.append(f"Origin: {state['current_city']}")
        if state.get("total_budget") is not None:
            ctx_parts.append(f"Budget: ${state['total_budget']:.0f}")
        if state.get("trip_days") is not None:
            ctx_parts.append(f"Trip duration: {state['trip_days']} days")
        state_context = ", ".join(ctx_parts) if ctx_parts else "None established yet."

        shown_cities = state.get("advisor_shown_cities") or []
        shown_cities_str = ", ".join(shown_cities) if shown_cities else "None yet."

        last_results = state.get("advisor_last_tool_results") or []
        last_results_str = _format_last_results(last_results) if last_results else "None."

        system_prompt = _PLANNER_SYSTEM_PROMPT.format(
            state_context=state_context,
            summary=summary if not is_fresh else "No previous context. This is a new conversation.",
            shown_cities=shown_cities_str if not is_fresh else "None yet.",
            last_tool_results=last_results_str if not is_fresh else "None.",
            categories=_AVAILABLE_CATEGORIES,
            tags=_AVAILABLE_TAGS,
            vibe_mapping=_VIBE_MAPPING,
        )

        plan: AdvisorPlan = self.planner.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": last_human.content},
        ])

        sanitized = _sanitize_plan(plan.steps)
        plan_steps = [
            {"tool_name": step.tool_name, "args": _step_to_args(step)}
            for step in sanitized
        ]
        print(f"\n[Planner] Created plan: {[s['tool_name'] for s in plan_steps]}")
        return {
            "advisor_plan": plan_steps,
            "advisor_last_tool_results": [],   # reset accumulated results for new turn
            "advisor_replan_count": 0,          # reset replan counter for new turn
        }
