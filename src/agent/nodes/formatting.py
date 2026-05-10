"""Format the agent's gathered travel data into a Markdown response."""
import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from agent.state import AgentState


def _parse_tool_payload(msg: BaseMessage) -> object | None:
    """Return the parsed JSON payload of a tool message, or None on failure."""
    try:
        return json.loads(msg.content)
    except json.JSONDecodeError:
        return None


def _ingest_flights(data: object, flights_dict: dict) -> None:
    """Merge fetch_flights tool output into the flights dictionary."""
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return
    for f in data:
        flight_num = f.get("flight_number", "unknown")
        flights_dict[flight_num] = f


def _ingest_hotels(data: object, hotels_dict: dict) -> None:
    """Merge fetch_hotels tool output into the hotels dictionary."""
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return
    for h in data:
        hotel_name = h.get("name", "unknown")
        hotels_dict[hotel_name] = h


def _ingest_trip_cost(data: object, trip_cost_calculations: list) -> None:
    """Append calculate_trip_cost output to the running list."""
    if isinstance(data, dict) and "total_estimate" in data:
        trip_cost_calculations.append(data)


def extract_travel_data(state: AgentState) -> dict:
    """Parse tool executions and deduplicate flights and hotels for the formatter."""
    flights_dict: dict = {}
    hotels_dict: dict = {}
    trip_cost_calculations: list = []

    for msg in state.get("messages", []):
        if msg.type != "tool":
            continue
        tool_name = getattr(msg, "name", "")
        data = _parse_tool_payload(msg)
        if data is None:
            continue
        if tool_name == "fetch_flights":
            _ingest_flights(data, flights_dict)
        elif tool_name == "fetch_hotels":
            _ingest_hotels(data, hotels_dict)
        elif tool_name == "calculate_trip_cost":
            _ingest_trip_cost(data, trip_cost_calculations)

    exclude_keys = {"messages", "step_count"}
    travel_data = {
        key: value
        for key, value in state.items()
        if key not in exclude_keys
    }

    travel_data["flights"] = list(flights_dict.values())
    travel_data["hotels"] = list(hotels_dict.values())
    travel_data["trip_cost_calculations"] = trip_cost_calculations

    return travel_data


class FormatterNode:
    """Render the final travel itinerary as Markdown."""

    def __init__(self, extraction_model: BaseChatModel) -> None:
        """Store the chat model used to render the formatted response."""
        self.extraction_model = extraction_model

    def __call__(self, state: AgentState) -> dict:
        """Format the gathered travel data into a Markdown response."""
        if not state.get("messages"):
            return {}

        last_msg = state["messages"][-1]
        content = last_msg.content if hasattr(last_msg, "content") else "No content"

        if isinstance(content, list):
            last_agent_message = "".join(
                part.get("text", "") for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        else:
            last_agent_message = str(content)

        if "let me know" in last_agent_message.lower():
            return {}

        has_origin = bool(state.get("current_city"))
        has_dest = bool(state.get("destination_city"))
        has_budget = bool(state.get("total_budget")) or bool(state.get("budget_optional"))

        if not (has_origin and has_dest and has_budget):
            return {}

        travel_data = extract_travel_data(state)

        # Filter hotels to only those affordable with at least one available flight
        budget = state.get("total_budget")
        trip_days = state.get("trip_days") or 3
        if budget and travel_data.get("flights") and travel_data.get("hotels"):
            affordable = [
                h for h in travel_data["hotels"]
                if any(
                    f.get("price", float("inf")) + h.get("price_per_night", float("inf")) * trip_days <= budget
                    for f in travel_data["flights"]
                )
            ]
            travel_data["hotels"] = affordable

        system_prompt = """
        You are a strict data formatter. Your ONLY job is to output the provided <data> into the EXACT Markdown template below.

        CRITICAL RULES:
        1. DO NOT add conversational filler.
        2. FORCE CURRENCY: You MUST use the '$' symbol for ALL prices and budgets. DO NOT use '€' or '£'.
        3. DO NOT ask the user any questions.
        4. If any section has no data, use the appropriate fallback text provided in the template.
        5. Do not invent flights or hotels. Use ONLY what is in the <data>.
        6. STRICT CONDITIONAL LOGIC: Follow the IF/ELSE logic in the template perfectly. Do not output the fallback text if data exists.
        7. TOTAL PRICE RULE: The <data> includes a "trip_cost_calculations" list with pre-computed totals from the calculate_trip_cost tool.
           Use the lowest total_estimate from that list as "Total Price". DO NOT recompute the total yourself.
           If trip_cost_calculations is empty, write "N/A" for Total Price.
        8. YOU MUST USE THIS EXACT TEMPLATE:

        [IF NO FLIGHTS ARE FOUND, USE THIS EXACT TEXT AND DO NOT ADD ANYTHING]
        Based on our search, we unfortunately could not find any available flights from your origin to [Destination City] at this time.

        ---
        [IF FLIGHTS ARE FOUND IN THE DATA, USE THIS FORMAT:]
        [Greeting tailored to the destination]
        ### ✨ **Your [Destination City] Escape** ✨

        **Destination:** [City Name, Country]
        **Total Budget:** [Budget with correct currency symbol]

        **Total Price:** [Use the lowest total_estimate from trip_cost_calculations, with $ symbol]

        ---

        ### ✈️ **Your Flight Details**
        Based on our search, we have found the following flight option:
        * **Airline:** [Airline Name]
        * **Flight Number:** [Flight Number]
        * **Price:** [Price with correct currency symbol]
        * **Status:** Available

        ---

        ### 🏨 **Accommodation Options in [Destination City]**

        [IF HOTELS ARE FOUND IN THE DATA, USE THIS FORMAT:]
        Based on our search, we've found excellent options to suit different preferences:

        **1. [Hotel Name]**
            * [Star Emojis corresponding to rating, e.g., ⭐ ⭐ ⭐] ([Number] Stars)
            * **Price Per Night:** [Price with correct currency symbol]
            * **Highlights:** [Brief, engaging sentence summarizing amenities]

        [Repeat numbered list for additional hotels]
        [Appropriate closing sign-off]
        """

        messages_to_pass = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"<data>\n{travel_data}\n</data>"},
        ]

        response = self.extraction_model.invoke(messages_to_pass)
        response.name = "formatter_output"

        return {"messages": [response]}
