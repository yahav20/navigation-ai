"""Format the agent's gathered travel data into a Markdown response."""
import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.messages import AIMessage

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

        has_origin = bool(state.get("current_city"))
        has_dest = bool(state.get("destination_city"))
        budget = state.get("total_budget")
        trip_days = state.get("trip_days")

        is_adjustment = state.get("is_adjustment", False)

        if not (has_origin and has_dest and bool(budget) and bool(trip_days)):
            return {}   

        travel_data = extract_travel_data(state)

        # Filter hotels by budget
        if budget and travel_data.get("flights") and travel_data.get("hotels"):
            affordable = [
                h for h in travel_data["hotels"]
                if any(
                    f.get("price", float("inf")) + h.get("price_per_night", float("inf")) * trip_days <= budget
                    for f in travel_data["flights"]
                )
            ]
            travel_data["hotels"] = affordable

        # Check in the code if we have the necessary data
        has_flights = bool(travel_data.get("flights"))
        has_hotels = bool(travel_data.get("hotels"))

        # Determine the texts based on the state (adjustment or initial search)
        if is_adjustment:
            success_greeting = "✅ **Trip Updated Successfully!** Here are the new details based on your requested changes:"
            no_flights_text = "⚠️ **Update Failed:** I tried to update your trip, but unfortunately, I couldn't find any available flights matching your new request."
            no_hotels_text = "⚠️ **Update Failed:** I found flights for your new request, but I couldn't find any hotels that fit your newly adjusted budget constraints."
        else:
            success_greeting = "Here is your perfect travel plan!"
            no_flights_text = f"Based on our search, we unfortunately could not find any available flights from {state.get('current_city')} to {state.get('destination_city')} at this time."
            no_hotels_text = "Based on our search, we found flights but could not find any hotels within your specified budget."

        # --- Solution: Handle failure cases directly in code, without LLM ---
        if not has_flights:
            return {"messages": [AIMessage(content=no_flights_text, name="formatter_output")]}
        
        if not has_hotels:
            return {"messages": [AIMessage(content=no_hotels_text, name="formatter_output")]}

        # --- Call LLM only in case of success with a clean prompt ---
        system_prompt = f"""
        You are a strict data formatter. Your ONLY job is to output the provided <data> into the EXACT Markdown template below.

        CRITICAL RULES:
        1. DO NOT add conversational filler.
        2. FORCE CURRENCY: You MUST use the '$' symbol for ALL prices and budgets.
        3. DO NOT ask the user any questions.
        4. Do not invent flights or hotels. Use ONLY what is in the <data>.
        5. TOTAL PRICE RULE: Use the lowest total_estimate from trip_cost_calculations.

        YOU MUST USE THIS EXACT TEMPLATE:

        {success_greeting}
        ### ✨ **Your [{state.get("destination_city")}] Escape** ✨

        **Destination:** [{state.get("destination_city")}]
        **Total Budget:** ${budget}
        **Trip Days:** {trip_days}

        **Total Price:** [Use the lowest total_estimate, with $ symbol]

        ---

        ### ✈️ **Your Flight Details**
        Based on our search, we have found the following flight option:
        * **Airline:** [Airline Name]
        * **Flight Number:** [Flight Number]
        * **Price:** [Price with correct currency symbol]

        ---

        ### 🏨 **Accommodation Options**
        Based on our search, we've found excellent options to suit different preferences:

        **1. [Hotel Name]**
            * [Star Emojis corresponding to rating]
            * **Price Per Night:** [Price with correct currency symbol]

        [Repeat numbered list for additional hotels]
        """

        messages_to_pass = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"<data>\n{travel_data}\n</data>"},
        ]

        response = self.extraction_model.invoke(messages_to_pass)
        response.name = "formatter_output"

        return {"messages": [response]}