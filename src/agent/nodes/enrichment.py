from langchain_core.messages import ToolMessage
from langgraph.graph import END
from agent.state import AgentState
from agent.models import UserPreferences, RefusalDetection
from tools.tools import create_data_provider
from tools.enrichment_tools import enrichment_tools, enrichment_tool_map

OPTION_THRESHOLD = 2


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _count_travel_options(origin: str, destination: str):
    """Return real (non-error) flights and hotels from the active provider."""
    provider = create_data_provider()
    flights = [f for f in provider.fetch_flights(origin, destination) if "message" not in f]
    hotels  = [h for h in provider.fetch_hotels(destination)           if "message" not in h]
    return flights, hotels


def _get_country_cities(destination: str, origin: str = None) -> list:
    """Return cities in the country if destination is a country name, else empty list."""
    return create_data_provider().get_cities_in_country(destination, origin)


def _apply_pref_filter(flights: list, hotels: list, prefs: dict):
    """Filter flights and hotels against the user's stated preferences."""
    f = flights[:]
    h = hotels[:]
    if prefs.get("max_flight_price"):
        f = [x for x in f if x.get("price", float("inf")) <= prefs["max_flight_price"]]
    if prefs.get("preferred_airline"):
        f = [x for x in f if prefs["preferred_airline"].lower() in x.get("airline", "").lower()]
    if prefs.get("min_hotel_stars"):
        h = [x for x in h if x.get("stars", 0) >= prefs["min_hotel_stars"]]
    if prefs.get("max_hotel_price_per_night"):
        h = [x for x in h if x.get("price_per_night", float("inf")) <= prefs["max_hotel_price_per_night"]]
    return f, h


# ---------------------------------------------------------------------------
# Phase functions
# Each returns either a terminal state-update dict (node should return it) or None (continue).
# Phase 1 also returns extra_updates to propagate through later phases.
# ---------------------------------------------------------------------------

def _detect_refusals(state, extraction_model, asked):
    """Run a single RefusalDetection call if we asked for fields that are still missing."""
    fields_pending = {
        f for f in asked
        if f in ("current_city", "destination_city", "total_budget")
        and not state.get(f)
        and not (f == "total_budget" and state.get("budget_optional"))
    }
    if not fields_pending:
        return None
    last_user = [m for m in state["messages"] if getattr(m, "type", "") == "human"]
    if not last_user:
        return None
    return extraction_model.with_structured_output(RefusalDetection).invoke([
        {
            "role": "system",
            "content": (
                "Analyse the user's message and determine whether they are explicitly "
                "refusing or claiming to be unable to provide their origin city, "
                "destination city, and/or travel budget. "
                "Only return True for fields that are clearly and intentionally refused."
            ),
        },
        {"role": "user", "content": last_user[-1].content},
    ])


def _phase1_required_fields(state, extraction_model, asked):
    """
    Check that current_city, destination_city, and total_budget are present.
    Mandatory city fields: refusal → firm message. Budget: refusal → budget_optional=True.

    Returns (terminal_dict or None, extra_updates).
    extra_updates carries budget_optional if the user just declined budget.
    """
    refusal = _detect_refusals(state, extraction_model, asked)
    extra = {}
    missing_labels = []
    missing_keys   = set()
    mandatory_refused = []

    if not state.get("current_city"):
        if refusal and refusal.refusing_origin_city:
            mandatory_refused.append("origin city")
        else:
            missing_labels.append("origin city")
            missing_keys.add("current_city")

    if not state.get("destination_city"):
        if refusal and refusal.refusing_destination_city:
            mandatory_refused.append("destination city")
        else:
            missing_labels.append("destination city")
            missing_keys.add("destination_city")

    if mandatory_refused:
        msg = extraction_model.invoke([
            {
                "role": "system",
                "content": (
                    "You are a travel assistant. Inform the user that the requested information "
                    "is mandatory — without it, a trip cannot be planned. "
                    "Ask them to provide it, or to type 'exit' to quit. "
                    "Be firm but polite and keep it brief."
                ),
            },
            {"role": "user", "content": f"User refused to provide: {' and '.join(mandatory_refused)}."},
        ])
        return (
            {"messages": [msg], "enrichment_complete": False, "enrichment_asked_fields": list(asked)},
            {},
        )

    if not state.get("total_budget") and not state.get("budget_optional"):
        if refusal and refusal.refusing_budget:
            extra["budget_optional"] = True
        else:
            missing_labels.append("travel budget")
            missing_keys.add("total_budget")

    if not state.get("trip_days"):
        if "trip_days" in asked:
            extra["trip_days"] = 3  # user was asked but didn't answer — use default
        else:
            missing_labels.append("number of days for the trip")
            missing_keys.add("trip_days")

    if missing_labels:
        msg = extraction_model.invoke([
            {
                "role": "system",
                "content": (
                    "You are a friendly travel assistant. "
                    "The user wants travel help but their request is missing some details. "
                    "Ask for the missing information in a warm, conversational way. Keep it to 1-2 sentences."
                ),
            },
            {"role": "user", "content": f"Missing information: {', '.join(missing_labels)}. Please ask the user for it."},
        ])
        return (
            {**extra, "messages": [msg], "enrichment_complete": False,
             "enrichment_asked_fields": list(asked | missing_keys)},
            {},
        )

    return None, extra


def _phase2_country_destination(state, extraction_model, origin):
    """
    If the destination looks like a country, ask the user to pick a city.
    Returns (terminal_dict or None, resolved_destination_string).
    """
    destination = state["destination_city"]
    cities = _get_country_cities(destination, origin)

    if len(cities) > 1:
        msg = extraction_model.invoke([
            {
                "role": "system",
                "content": (
                    "You are a friendly travel assistant. "
                    "The user specified a country as their destination. "
                    "Let them know the available destination cities and ask which one they'd like to visit. "
                    "Keep it friendly and brief."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User wants to travel to {destination} (a country). "
                    f"Available destination cities with flights: {', '.join(cities)}. "
                    "Ask them to choose one."
                ),
            },
        ])
        return {"messages": [msg], "enrichment_complete": False}, destination

    if len(cities) == 1:
        return None, cities[0]

    return None, destination


def _phase4_extract_preferences(state, extraction_model, flights, hotels):
    """
    Scan conversation history for stated preferences and apply them as a filter.
    Returns a terminal dict if preferences resolve the option count, else None.
    """
    extracted: UserPreferences = extraction_model.with_structured_output(UserPreferences).invoke([
        {
            "role": "system",
            "content": (
                "Extract any travel preferences the user has explicitly expressed. "
                "Only extract clearly stated preferences. Return null for anything not mentioned."
            ),
        },
        *state["messages"],
    ])

    prefs = {k: v for k, v in extracted.model_dump().items() if v is not None}
    if not prefs:
        return None

    filtered_flights, filtered_hotels = _apply_pref_filter(flights, hotels, prefs)
    if filtered_flights or filtered_hotels:
        return {"user_preferences": prefs, "enrichment_complete": True}
    # Preferences matched nothing — proceed with all options
    return {"enrichment_complete": True}


def _phase5_ask_question(origin, destination, too_many, enrichment_question_model):
    """
    Call dimension tools and ask the user one targeted preference question.
    Always halts the graph.
    """
    system_msg = {
        "role": "system",
        "content": (
            "You are a travel assistant helping a user narrow down their options. "
            "Call the provided tools to check what hotel and/or flight options are available for the trip, "
            "then ask the user ONE clear, targeted question based on what you find. "
            "Only ask about dimensions that have real variety "
            "(e.g., if all hotels share the same star rating, do not mention stars). "
            "Be friendly and keep it to 2-3 sentences."
        ),
    }
    user_msg = {
        "role": "user",
        "content": (
            f"The user wants to travel from {origin} to {destination}. "
            f"There are too many {' and '.join(too_many)} to present without filtering. "
            "Use the tools to check what options exist, then ask the user a targeted preference question."
        ),
    }

    first_response = enrichment_question_model.invoke([system_msg, user_msg])

    if first_response.tool_calls:
        tool_results = [
            ToolMessage(content=str(fn.invoke(tc["args"])), tool_call_id=tc["id"], name=tc["name"])
            for tc in first_response.tool_calls
            if (fn := enrichment_tool_map.get(tc["name"]))
        ]
        final_response = enrichment_question_model.invoke(
            [system_msg, user_msg, first_response, *tool_results]
        )
    else:
        final_response = first_response

    return {"messages": [final_response], "enrichment_complete": False}


# ---------------------------------------------------------------------------
# Node class — replaces the make_check_enrichment factory + nested function
# ---------------------------------------------------------------------------

def after_enrichment(state: AgentState) -> str:
    """Conditional edge: route to agent when enrichment is complete, else surface question to user."""
    return "agent" if state.get("enrichment_complete", False) else END


class EnrichmentNode:
    """
    LangGraph node that runs the five-phase enrichment gate.
    Accepts only extraction_model at construction time; binds enrichment tools internally.
    """

    def __init__(self, extraction_model):
        self.extraction_model = extraction_model
        self.enrichment_question_model = extraction_model.bind_tools(enrichment_tools)

    def __call__(self, state: AgentState):
        asked = set(state.get("enrichment_asked_fields") or [])

        # Phase 1 — required fields
        terminal, extra = _phase1_required_fields(state, self.extraction_model, asked)
        if terminal is not None:
            return terminal

        origin = state.get("current_city")

        # Phase 2 — country destination
        terminal, destination = _phase2_country_destination(state, self.extraction_model, origin)
        if terminal is not None:
            return {**extra, **terminal}

        dest_update = {"destination_city": destination} if destination != state.get("destination_city") else {}

        # Phase 3 — option count
        flights, hotels = _count_travel_options(origin, destination)
        too_many = [label for label, lst in [("flights", flights), ("hotels", hotels)]
                    if len(lst) > OPTION_THRESHOLD]
        if not too_many:
            return {**extra, **dest_update, "enrichment_complete": True}

        # Phase 4 — preference extraction
        terminal = _phase4_extract_preferences(state, self.extraction_model, flights, hotels)
        if terminal is not None:
            return {**extra, **dest_update, **terminal}

        # Phase 5 — enrichment question mini-agent
        return {**extra, **dest_update,
                **_phase5_ask_question(origin, destination, too_many, self.enrichment_question_model)}
