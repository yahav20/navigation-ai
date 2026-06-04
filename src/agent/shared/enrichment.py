"""Enrichment gate node and its phase helpers for the travel agent."""
from datetime import date, timedelta

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.runnables import Runnable

from agent.core.models import RefusalDetection, UserPreferences
from agent.core.state import AgentState
from providers import SQLiteDataProvider
from tools import enrichment_tool_map, enrichment_tools

# Deterministic fallback: well-known country → primary city for flight resolution.
_COUNTRY_TO_CITY: dict[str, str] = {
    "israel": "Tel Aviv",
    "france": "Paris",
    "uk": "London",
    "england": "London",
    "britain": "London",
    "great britain": "London",
    "usa": "New York City",
    "united states": "New York City",
    "united states of america": "New York City",
    "america": "New York City",
    "japan": "Tokyo",
    "germany": "Berlin",
    "netherlands": "Amsterdam",
    "holland": "Amsterdam",
    "spain": "Madrid",
    "italy": "Rome",
    "portugal": "Lisbon",
    "australia": "Sydney",
    "canada": "Toronto",
    "china": "Beijing",
    "india": "Mumbai",
    "brazil": "São Paulo",
    "mexico": "Mexico City",
    "russia": "Moscow",
    "turkey": "Istanbul",
    "greece": "Athens",
    "thailand": "Bangkok",
    "uae": "Dubai",
    "united arab emirates": "Dubai",
    "south korea": "Seoul",
    "korea": "Seoul",
    "egypt": "Cairo",
    "jordan": "Amman",
    "morocco": "Casablanca",
}

data_provider = SQLiteDataProvider()

OPTION_THRESHOLD = 2
DEFAULT_TRIP_DAYS = 3


def _default_trip_start() -> str:
    """Fall back to next month (YYYY-MM) if the user skips picking a date."""
    return (date.today() + timedelta(days=30)).strftime("%Y-%m")


def _count_travel_options(origin: str, destination: str) -> tuple[list, list]:
    flights = [f for f in data_provider.fetch_flights(origin, destination) if "message" not in f]
    hotels  = [h for h in data_provider.fetch_hotels(destination)           if "message" not in h]
    return flights, hotels


def _get_country_cities(destination: str, origin: str | None = None) -> list:
    return data_provider.get_cities_in_country(destination, origin)


def _get_origin_cities_from_country(origin: str, destination: str | None = None) -> list:
    return data_provider.get_origin_cities_in_country(origin, destination)


def _apply_pref_filter(flights: list, hotels: list, prefs: dict) -> tuple[list, list]:
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


def _detect_refusals(state: AgentState, extraction_model: BaseChatModel, asked: set) -> RefusalDetection | None:
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


def _classify_city_fields(
    state: AgentState,
    refusal: RefusalDetection | None,
) -> tuple[list[str], set[str], list[str]]:
    missing_labels: list[str] = []
    missing_keys: set[str] = set()
    mandatory_refused: list[str] = []

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

    return missing_labels, missing_keys, mandatory_refused


def _classify_optional_fields(
    state: AgentState,
    refusal: RefusalDetection | None,
    asked: set,
    missing_labels: list[str],
    missing_keys: set[str],
) -> dict:
    extra: dict = {}

    if not state.get("total_budget") and not state.get("budget_optional"):
        if refusal and refusal.refusing_budget:
            extra["budget_optional"] = True
        else:
            missing_labels.append("travel budget")
            missing_keys.add("total_budget")

    if not state.get("trip_days"):
        if "trip_days" in asked:
            extra["trip_days"] = DEFAULT_TRIP_DAYS
        else:
            missing_labels.append("number of days for the trip")
            missing_keys.add("trip_days")

    if not state.get("trip_start"):
        if "trip_start" in asked:
            extra["trip_start"] = _default_trip_start()
        else:
            missing_labels.append("approximate trip start (a month is fine, e.g. 'June')")
            missing_keys.add("trip_start")

    return extra


def _build_mandatory_refusal_message(
    extraction_model: BaseChatModel,
    mandatory_refused: list[str],
    asked: set,
) -> dict:
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
    return {"messages": [msg], "enrichment_complete": False, "enrichment_asked_fields": list(asked)}


def _build_missing_info_message(
    extraction_model: BaseChatModel,
    missing_labels: list[str],
    missing_keys: set[str],
    asked: set,
    extra: dict,
) -> dict:
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
    return {
        **extra,
        "messages": [msg],
        "enrichment_complete": False,
        "enrichment_asked_fields": list(asked | missing_keys),
    }


def _phase1_required_fields(
    state: AgentState,
    extraction_model: BaseChatModel,
    asked: set,
) -> tuple[dict | None, dict]:
    refusal = _detect_refusals(state, extraction_model, asked)
    missing_labels, missing_keys, mandatory_refused = _classify_city_fields(state, refusal)

    if mandatory_refused:
        return _build_mandatory_refusal_message(extraction_model, mandatory_refused, asked), {}

    extra = _classify_optional_fields(state, refusal, asked, missing_labels, missing_keys)

    if missing_labels:
        return (
            _build_missing_info_message(extraction_model, missing_labels, missing_keys, asked, extra),
            {},
        )

    return None, extra


def _phase2a_origin_country(
    state: AgentState,
    extraction_model: BaseChatModel,
    destination: str | None,
) -> tuple[dict | None, str | None]:
    origin = state.get("current_city")
    cities = _get_origin_cities_from_country(origin, destination)

    if len(cities) > 1:
        msg = extraction_model.invoke([
            {
                "role": "system",
                "content": (
                    "You are a friendly travel assistant. "
                    "The user specified a country as their origin. "
                    "Let them know the available departure cities and ask which one they'd like to fly from. "
                    "Keep it friendly and brief."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User wants to travel from {origin} (a country). "
                    f"Available departure cities with flights: {', '.join(cities)}. "
                    "Ask them to choose one."
                ),
            },
        ])
        return {"messages": [msg], "enrichment_complete": False}, origin

    if len(cities) == 1:
        return None, cities[0]

    if origin:
        fallback_city = _COUNTRY_TO_CITY.get(origin.strip().lower())
        if fallback_city:
            return None, fallback_city

    return None, origin


def _phase2_country_destination(
    state: AgentState,
    extraction_model: BaseChatModel,
    origin: str | None,
) -> tuple[dict | None, str]:
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

    if destination:
        fallback_city = _COUNTRY_TO_CITY.get(destination.strip().lower())
        if fallback_city:
            return None, fallback_city

    return None, destination


import re

_TYPO_MAP: list[tuple[str, str]] = [
    (r"\bwanbt\b", "want"),
    (r"\bwannt\b", "want"),
    (r"\bwnat\b",  "want"),
    (r"\bcosher\b", "kosher"),
    (r"\bkashre\b", "kosher"),
    (r"\bkasher\b", "kosher"),
    (r"\bkoshre\b", "kosher"),
    (r"\bkoshe\b",  "kosher"),
    (r"\bvegn\b",       "vegan"),
    (r"\bvgen\b",       "vegan"),
    (r"\bvegeterian\b", "vegetarian"),
    (r"\bhalall\b",     "halal"),
    (r"\bhalel\b",      "halal"),
]


def _normalise_typos(text: str) -> str:
    for pattern, replacement in _TYPO_MAP:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def _extract_general_preferences(
    state: AgentState,
    extraction_model: BaseChatModel,
) -> dict:
    messages = state.get("messages", [])

    last_user_message = next((m for m in reversed(messages) if getattr(m, "type", "") == "human"), None)

    if not last_user_message:
        return {}

    raw_content: str = last_user_message.content
    normalised_content = _normalise_typos(raw_content)

    extracted: UserPreferences = extraction_model.with_structured_output(UserPreferences).invoke([
        {
            "role": "system",
            "content": (
                "Extract any travel preferences the user has explicitly expressed in their latest message. "
                "Look for dietary restrictions, preferred airlines, mobility/accessibility needs, or hotel vibes. "
                "IMPORTANT — handle common typos and misspellings gracefully:\n"
                "  - 'wanbt', 'wannt', 'wan't' → 'want'\n"
                "  - 'cosher', 'kasher', 'koshre', 'koshe' → 'kosher' (dietary_restrictions = 'kosher')\n"
                "  - 'vegn', 'vgen' → 'vegan'\n"
                "  - 'vegeterian', 'vegetarian' → 'vegetarian'\n"
                "  - 'halall', 'halel' → 'halal'\n"
                "Treat 'kosher hotel', 'cosher hotel', 'kasher hotel' as dietary_restrictions = 'kosher'. "
                "CRITICAL: Do NOT guess, split, or calculate flight/hotel budgets from the user's total budget. "
                "Return null for anything not explicitly mentioned."
            ),
        },
        {"role": "user", "content": normalised_content},
    ])

    new_prefs = {k: v for k, v in extracted.model_dump().items() if v is not None}

    if not new_prefs:
        return {}

    current_prefs = state.get("user_preferences") or {}
    updated_prefs = {**current_prefs, **new_prefs}

    return {"user_preferences": updated_prefs}


def _phase5_ask_question(
    origin: str | None,
    destination: str | None,
    too_many: list[str],
    enrichment_question_model: Runnable,
) -> dict:
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
            [system_msg, user_msg, first_response, *tool_results],
        )
    else:
        final_response = first_response

    return {"messages": [final_response], "enrichment_complete": False}


class EnrichmentNode:
    """Run the multi-phase enrichment gate before delegating to the main agent."""

    def __init__(self, extraction_model: BaseChatModel) -> None:
        """Bind enrichment tools to the extraction model and store both."""
        self.extraction_model = extraction_model
        self.enrichment_question_model = extraction_model.bind_tools(enrichment_tools)

    def __call__(self, state: AgentState) -> dict:
        """Run the enrichment phases and return the next state update."""
        asked = set(state.get("enrichment_asked_fields") or [])

        terminal, extra = _phase1_required_fields(state, self.extraction_model, asked)
        if terminal is not None:
            return terminal

        origin = state.get("current_city")
        destination = state.get("destination_city")

        terminal, origin = _phase2a_origin_country(state, self.extraction_model, destination)
        if terminal is not None:
            return {**extra, **terminal}
        origin_update = {"current_city": origin} if origin != state.get("current_city") else {}

        terminal, destination = _phase2_country_destination(state, self.extraction_model, origin)
        if terminal is not None:
            return {**extra, **origin_update, **terminal}

        dest_update = {"destination_city": destination} if destination != state.get("destination_city") else {}

        pref_update = _extract_general_preferences(state, self.extraction_model)

        return {**extra, **origin_update, **dest_update, **pref_update, "enrichment_complete": True}
