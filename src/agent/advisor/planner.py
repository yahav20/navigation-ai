"""Planner node — decides which tools to call and in what order for an advisor query."""
from typing import Literal

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from agent.core.llm import silent
from agent.core.state import AgentState
from ui import render_node, render_node_status


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
    result: _RelevanceCheck = silent(extraction_model.with_structured_output(_RelevanceCheck)).invoke([
        {"role": "system", "content": _RELEVANCE_PROMPT},
        {"role": "user", "content": f"Previous conversation summary:\n{summary}\n\nNew question: {new_question}"},
    ])
    return result.is_followup


# ---------------------------------------------------------------------------
# Travel relevance check — gates non-travel questions before planning
# ---------------------------------------------------------------------------

class _TravelRelevance(BaseModel):
    is_travel_related: bool


_TRAVEL_RELEVANCE_PROMPT = """You are a classifier for a travel assistant.
Determine whether the user's question is related to travel or this assistant's capabilities.

Return is_travel_related=True for ANY of these:
- Destinations, cities, countries, regions
- Flights, hotels, accommodations, transport
- Travel safety, visa/passport requirements
- Currency exchange (for travel context)
- Packing for trips
- Local customs, etiquette, tipping
- Trip planning, itineraries, activities
- Tourist attractions, landmarks, museums
- Weather for travel purposes
- Greetings to the assistant ("hello", "hi", "thanks")
- Questions about what the assistant can do ("what can you help me with?")

Return is_travel_related=False ONLY when the question has zero travel connection:
- Coding / programming questions
- Math or science homework
- Food recipes unrelated to travel
- Sports scores or results
- Medical / legal advice
- Other topics a travel assistant should not touch"""


def _is_travel_related(extraction_model: BaseChatModel, question: str) -> bool:
    """Return False only when the question is clearly unrelated to travel."""
    # silent(): keep this classifier's structured-output tokens off the chat
    # stream, otherwise `{"is_travel_related": ...}` flashes in the UI before the
    # real answer (matches the other silenced calls in this file).
    result: _TravelRelevance = silent(
        extraction_model.with_structured_output(_TravelRelevance)
    ).invoke([
        {"role": "system", "content": _TRAVEL_RELEVANCE_PROMPT},
        {"role": "user", "content": question},
    ])
    return result.is_travel_related


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
    "get_currency_exchange",
    "get_visa_requirements",
    "get_travel_safety_info",
    "get_packing_list",
    "get_local_customs",
    "get_wikipedia_summary",
    "search_concerts",
]


class PlannedToolCall(BaseModel):
    tool_name: ToolName = Field(description="Exact name of the tool to call")
    city: str | list[str] | None = Field(default=None, description=(
        "City name(s). Pass a plain string for single-city tools. "
        "For get_trip_duration_advisor only: pass a LIST of city names when the user asks "
        "about multiple cities at once (e.g. ['Rome', 'Amsterdam']) — this fetches all "
        "durations in one query instead of separate calls. "
        "Required for: get_city_overview, get_trip_duration_advisor, "
        "fetch_activities, get_best_time_to_visit, get_average_weather, "
        "get_travel_safety_info, get_visa_requirements, get_local_customs, get_packing_list."
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
        "Explicit trip duration in days — required for find_destinations_within_budget "
        "and get_packing_list"
    ))
    max_flight_hours: float | None = Field(default=None, description=(
        "Max flight hours for get_reachable_destinations. "
        "Omit when no flight-length constraint. short=2.5, medium=5, long=9"
    ))
    season: str | None = Field(default=None, description=(
        "Season for get_average_weather and get_packing_list. "
        "Must be exactly: Spring, Summer, Autumn, or Winter"
    ))
    passport_nationality: str | None = Field(default=None, description=(
        "User's passport nationality — required for get_visa_requirements. "
        "E.g. 'Israeli', 'American', 'British'. Use 'Unknown' if not mentioned."
    ))
    from_currency: str | None = Field(default=None, description=(
        "Source currency code or name — required for get_currency_exchange. "
        "E.g. 'USD', 'dollar', 'EUR', 'euro'"
    ))
    to_currency: str | None = Field(default=None, description=(
        "Target currency code or name — required for get_currency_exchange. "
        "E.g. 'ILS', 'shekel', 'GBP', 'pound'"
    ))
    amount: float | None = Field(default=None, description=(
        "Amount to convert — optional for get_currency_exchange. Defaults to 1.0 if omitted."
    ))
    trip_type: str | None = Field(default=None, description=(
        "Trip type for get_packing_list. "
        "One of: city, beach, nature, cultural, business. Default: city"
    ))
    topic: str | None = Field(default=None, description=(
        "Topic name for get_wikipedia_summary — a city, attraction, landmark, or place. "
        "Use the most specific name (e.g. 'Colosseum', 'Eiffel Tower', 'Kyoto')."
    ))
    artist: str | None = Field(default=None, description=(
        "Artist or performer name — used by search_concerts. "
        "E.g. 'John Legend', 'Hardwell', 'Coldplay'."
    ))
    month: str | None = Field(default=None, description=(
        "Month and year for search_concerts — e.g. 'August 2026', 'September 2026'. "
        "Omit if the user did not specify a time period."
    ))


class AdvisorPlan(BaseModel):
    steps: list[PlannedToolCall] = Field(description="Ordered list of tool calls to execute, maximum 3")


# ---------------------------------------------------------------------------
# Convert a PlannedToolCall to the args dict the executor expects
# ---------------------------------------------------------------------------

_ARG_FIELDS = (
    "city", "tag", "category", "origin", "total_budget", "trip_days",
    "max_flight_hours", "season", "passport_nationality", "from_currency",
    "to_currency", "amount", "trip_type", "topic", "artist", "month",
)

# These tools use 'destination' as their parameter name instead of 'city'
_CITY_AS_DESTINATION_TOOLS = frozenset({
    "get_travel_safety_info",
    "get_visa_requirements",
    "get_local_customs",
    "get_packing_list",
})


def _step_to_args(step: PlannedToolCall) -> dict:
    args = {f: getattr(step, f) for f in _ARG_FIELDS if getattr(step, f) is not None}
    if step.tool_name in _CITY_AS_DESTINATION_TOOLS and "city" in args:
        args["destination"] = args.pop("city")
    return args


_BUDGET_TOOLS = frozenset({"find_destinations_within_budget", "find_destinations_within_budget_auto"})
_DISCOVERY_TOOLS = frozenset({"find_destinations_by_tag", "find_destinations_by_vibe"})


def _sanitize_plan(steps: list[PlannedToolCall]) -> list[PlannedToolCall]:
    """Enforce hard architectural constraints the LLM occasionally ignores.

    Rule A: when a budget tool is present, discovery tools are redundant and misleading
    (they return cities without cost/reachability filtering). Remove them.

    Rule B: when search_concerts is present, it must be the only tool.
    Adding get_best_time_to_visit or any other tool causes the formatter to ignore
    the concert results and answer with irrelevant general-travel data instead.
    """
    tool_names = {s.tool_name for s in steps}
    if tool_names & _BUDGET_TOOLS:
        steps = [s for s in steps if s.tool_name not in _DISCOVERY_TOOLS]
    if "search_concerts" in tool_names:
        steps = [s for s in steps if s.tool_name == "search_concerts"]
    return steps


# ---------------------------------------------------------------------------
# Constants (same values used by the tool implementations)
# ---------------------------------------------------------------------------

_AVAILABLE_CATEGORIES = "Culture, Entertainment, Family, History, Nature, Nightlife, Sightseeing"

_AVAILABLE_TAGS = (
    "alternative, art, beach, budget-friendly, canals, city-break, cultural, cycling, "
    "eco-friendly, entertainment, expensive, fashion, foodie, historic, iconic, liberal, luxury, "
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

SPECIAL CASES — handle BEFORE planning any tools:

GREETINGS & META QUESTIONS:
If the user is greeting you ("hello", "hi", "hey", "thanks") or asking what you can do
("what can you do?", "how do you work?", "what are you?"): return an EMPTY steps list.
The formatter will generate a warm welcome response. Do NOT plan any tool calls.

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

DESTINATION DISCOVERY TOOLS:

- find_destinations_by_vibe
    Set: category = one of the AVAILABLE ACTIVITY CATEGORIES
    -> For activity-type queries: Nature, Culture, History, Family, Nightlife, Sightseeing
    -> Pick the single closest category (or one call per filter for multi-filter queries — see MULTI-FILTER RULE)

- find_destinations_by_tag
    Set: tag = one of the AVAILABLE CITY TAGS
    -> For atmosphere, lifestyle, or vibe queries: romantic, beach, foodie, budget-friendly, etc.
    -> Pick the single closest tag (or one call per filter for multi-filter queries — see MULTI-FILTER RULE)

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

PRACTICAL TRAVEL TOOLS (informational — each needs exactly ONE call, no combinations needed):

- get_currency_exchange
    Set: from_currency = <currency code or name>, to_currency = <currency code or name>, amount = <number or omit for rate only>
    -> Trigger: any exchange rate or currency conversion question
    -> Examples: "how much is $500 in euros?", "what's the euro to shekel rate?", "convert 100 USD to JPY"
    -> Always one tool call; never combine with other tools

- get_visa_requirements
    Set: city = <destination country or city>, passport_nationality = <user's nationality>
    -> Trigger: visa / passport requirement questions for a specific destination
    -> Examples: "do I need a visa for Japan?", "can I enter Thailand visa-free?", "visa for Schengen?"
    -> If passport nationality is unknown, use "Unknown"

- get_travel_safety_info
    Set: city = <destination>
    -> Trigger: safety, risk level, or travel advisory for a destination
    -> Examples: "is Bangkok safe?", "travel warnings for Mexico?", "how safe is Cairo?"

- get_packing_list
    Set: city = <destination>, season = <Spring|Summer|Autumn|Winter>, trip_days = <int>, trip_type = <city|beach|nature|cultural|business>
    -> Trigger: packing advice or what to bring for a trip
    -> Examples: "what should I pack for Tokyo in summer?", "packing list for a beach trip to Bali"
    -> Infer trip_type from context; default to "city" if unclear
    -> If trip_days unknown, use 7 as a reasonable default

- get_local_customs
    Set: city = <destination or country>
    -> Trigger: etiquette, tipping norms, local customs, or useful phrases for a destination
    -> Examples: "what are the customs in Japan?", "should I tip in France?", "useful phrases for Italy"

- search_concerts
    Set: city = <city>, artist = <performer name>, month = <"Month YYYY">
    -> For questions about live concerts, shows, gigs, or DJ sets
    -> At least one parameter must be provided; combine all that are known
    -> Trigger phrases: "concerts in X", "shows in X in [month]", "where is [artist] touring",
                        "when is [artist] in [city]", "where can I see [artist]"
    -> Searches exclusively: Songkick, Bandsintown, Ticketmaster, Live Nation, Resident Advisor
    -> Returns live web results — the formatter will extract event names, dates, venues, and cities
    -> ALWAYS ONE call only — NEVER combine with ANY other tool (not get_best_time_to_visit,
       not get_city_overview, not fetch_activities, not any discovery tool).
       Concert date data IS the answer; general travel timing is irrelevant here.

    Query-mode examples:
      "concerts in London in August 2026"           → city="London",       month="August 2026"
      "where is John Legend touring September 2026" → artist="John Legend", month="September 2026"
      "when is Hardwell in Amsterdam"               → artist="Hardwell",    city="Amsterdam"
      "upcoming Coldplay shows"                     → artist="Coldplay"

- get_wikipedia_summary
    Set: topic = <specific place, attraction, or landmark name>
    -> For questions about a SPECIFIC ATTRACTION, LANDMARK, or HISTORICAL SITE (NOT a city as a whole)
    -> Trigger phrases: "tell me about X", "what is X?", "history of X", "what's the X?", "describe X"
    -> Examples: "tell me about the Colosseum" → topic="Colosseum"
                 "what is the Sagrada Familia?" → topic="Sagrada Familia"
                 "history of the Acropolis" → topic="Acropolis"
                 "tell me about the Great Wall" → topic="Great Wall of China"
    -> Do NOT use for city-level questions ("tell me about Rome" → use get_city_overview instead)
    -> Use the most specific name (e.g. "Colosseum", not "that famous place in Rome")
    -> One call only

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
   -> Exception: if the user specifies MULTIPLE vibes/filters, see MULTI-FILTER RULE below

MULTI-FILTER RULE — User specifies TWO or more distinct destination preferences:
   When the user combines two or more distinct filters (e.g. "cheap AND family", "beach AND romantic",
   "cultural AND nightlife"), plan ONE discovery tool per filter. Results will be automatically
   intersected so that only destinations satisfying ALL filters are shown.

   Examples:
     "cheap family destination"      → find_destinations_by_tag(tag="budget-friendly")
                                        + find_destinations_by_vibe(category="Family")
     "romantic beach getaway"        → find_destinations_by_tag(tag="romantic")
                                        + find_destinations_by_tag(tag="beach")
     "cultural city with nightlife"  → find_destinations_by_vibe(category="Culture")
                                        + find_destinations_by_tag(tag="nightlife")
     "safe, family-friendly nature"  → find_destinations_by_tag(tag="safe")
                                        + find_destinations_by_vibe(category="Family")
                                        (2 tools already — do not add a third discovery tool)

   With origin and NO budget:
     "cheap family trip from Paris"  → get_reachable_destinations(origin="Paris")
                                        + find_destinations_by_tag(tag="budget-friendly")
                                        + find_destinations_by_vibe(category="Family")

   IMPORTANT constraints:
   - Maximum 2 filter-discovery tools even when 3+ vibes are mentioned (pick the 2 most important)
   - Budget tools (RULES 1 & 2) ALWAYS override this rule: if user has origin + budget, use budget tool only
   - If one filter maps to both a tag and a vibe (e.g. "nightlife"), count it as ONE filter tool

7. General destination ideas with no vibe, no origin, no budget:
   -> Plan: find_destinations_by_tag (tag="city-break") OR find_destinations_by_vibe (category="Sightseeing")

8. User asks about a specific city (overview / general info / "tell me everything"):
   -> Plan EXACTLY ONE tool: get_city_overview (city=<city>)
   -> get_city_overview already returns activity types, best visit months, AND seasonal temperatures
   -> DO NOT also call get_best_time_to_visit — it is completely redundant when get_city_overview is called
   -> Add fetch_activities as a second tool ONLY if the user explicitly asks "what can I do there?"
   -> Maximum 2 tools total for city-overview questions

9. User asks how many days for one or more cities:
   -> Plan EXACTLY ONE tool: get_trip_duration_advisor with city=<city> for a single city,
      or city=["Rome", "Amsterdam"] (a list) when multiple cities are mentioned.
   -> Never plan multiple separate get_trip_duration_advisor calls — use a single call with a list.

10. User asks "what budget do I need?" or "how much would it cost?":
    -> If trip_days is known: find_destinations_within_budget (origin=<city>, total_budget=99999, trip_days=<d>)
    -> If trip_days unknown: find_destinations_within_budget_auto (origin=<city>, total_budget=99999)

11. User asks about family trip:
    -> Plan: find_destinations_by_vibe (category="Family") — exactly 1 tool is usually enough

12. Practical travel questions (currency, visa, safety, packing, customs, Wikipedia):
    -> Plan EXACTLY ONE tool from the PRACTICAL TRAVEL TOOLS section above
    -> Never combine practical tools with destination discovery tools in the same plan
    -> Never combine two practical tools unless the user explicitly asked two separate questions

    Quick trigger reference:
      "how much is X in Y / convert X to Y / exchange rate"  → get_currency_exchange
      "do I need a visa / visa requirements for X"           → get_visa_requirements
      "is X safe / travel warning / safety in X"            → get_travel_safety_info
      "what should I pack / packing list for X"             → get_packing_list
      "customs in X / tipping in X / etiquette / phrases"   → get_local_customs
      "tell me about [landmark] / what is [landmark]"       → get_wikipedia_summary
      (city-level: "tell me about Rome" → get_city_overview, NOT wikipedia)
      "concerts / shows / gigs / events / touring / DJ set" → search_concerts

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
- A simple question (weather, duration, single-city overview, currency, visa, safety) needs exactly 1 tool call.
- Do NOT plan fetch_activities for destination advisor questions.
- Do NOT plan get_reachable_destinations if you have no origin city to use.
- Do NOT plan tools whose required fields (origin, city) are completely unknown — except when a
  sensible default exists (e.g. find_destinations_by_tag with tag="city-break" needs no origin).
- Never plan to ask the user a question — just choose the best tool(s) with the available context.
"""


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class AdvisorPlannerNode:
    """Generate an ordered tool-call plan from the user's advisory question."""

    def __init__(self, extraction_model: BaseChatModel) -> None:
        self.planner = silent(extraction_model.with_structured_output(AdvisorPlan, method="function_calling"))
        self.extraction_model = extraction_model

    def __call__(self, state: AgentState) -> dict:
        render_node("advisor_planner")
        summary = state.get("summary", "")
        messages = list(state.get("messages", []))

        last_human = next(
            (m for m in reversed(messages) if getattr(m, "type", "") == "human"), None
        )
        if not last_human:
            return {"advisor_plan": []}

        # Gate: reject non-travel questions before planning
        if not _is_travel_related(self.extraction_model, last_human.content):
            render_node_status("[Planner] Non-travel question detected — flagging as out-of-scope.")
            return {
                "advisor_plan": [],
                "advisor_out_of_scope": True,
                "advisor_last_tool_results": [],
                "advisor_replan_count": 0,
            }

        # Fresh questions should not bleed context from previous tool turns
        is_fresh = not _is_followup(self.extraction_model, summary, last_human.content)

        ctx_parts = []
        if state.get("current_city"):
            ctx_parts.append(f"Origin: {state['current_city']}")
        if state.get("destination_city"):
            ctx_parts.append(f"Destination: {state['destination_city']}")
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
        render_node_status(f"[Planner] Created plan: {[s['tool_name'] for s in plan_steps]}")
        return {
            "advisor_plan": plan_steps,
            "advisor_last_tool_results": [],   # reset accumulated results for new turn
            "advisor_replan_count": 0,          # reset replan counter for new turn
            "advisor_out_of_scope": False,
        }
