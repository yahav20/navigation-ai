"""Format the agent's gathered travel data into a Markdown response."""
import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from agent.state import AgentState


def _parse_tool_payload(msg: BaseMessage) -> object | None:
    try:
        return json.loads(msg.content)
    except json.JSONDecodeError:
        return None


def _ingest_flights(data: object, flights_dict: dict) -> None:
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return
    for f in data:
        if "route" in f:
            flight_num = "-".join([leg.get("flight", "unk") for leg in f["route"]])
        else:
            flight_num = f.get("flight_number", "unknown")
            
        flights_dict[flight_num] = f


def _ingest_hotels(data: object, hotels_dict: dict) -> None:
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return
    for h in data:
        hotel_name = h.get("name", "unknown")
        hotels_dict[hotel_name] = h


def _ingest_trip_cost(data: object, trip_cost_calculations: list) -> None:
    if isinstance(data, dict) and "total_estimate" in data:
        trip_cost_calculations.append(data)


def extract_travel_data(state: AgentState) -> dict:
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
        if tool_name in ["fetch_flights", "find_connecting_flights"]:
            _ingest_flights(data, flights_dict)
        elif tool_name == "fetch_hotels":
            _ingest_hotels(data, hotels_dict)
        elif tool_name == "calculate_trip_cost":
            _ingest_trip_cost(data, trip_cost_calculations)

    exclude_keys = {"messages", "step_count"}
    travel_data = {
        key: value for key, value in state.items() if key not in exclude_keys
    }

    travel_data["flights"] = list(flights_dict.values())
    travel_data["hotels"] = list(hotels_dict.values())
    travel_data["trip_cost_calculations"] = trip_cost_calculations

    return travel_data


class FormatterNode:
    def __init__(self, extraction_model: BaseChatModel) -> None:
        self.extraction_model = extraction_model

    def __call__(self, state: AgentState) -> dict:
        if not state.get("messages"):
            return {}

        last_msg = state["messages"][-1]
        content = last_msg.content if hasattr(last_msg, "content") else "No content"

        if isinstance(content, list):
            last_agent_message = "".join(
                part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"
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

        budget = state.get("total_budget")
        trip_days = state.get("trip_days") or 3
        
        if budget and travel_data.get("flights") and travel_data.get("hotels"):
            affordable = []
            for h in travel_data["hotels"]:
                hotel_total = h.get("price_per_night", float("inf")) * trip_days
                for f in travel_data["flights"]:
                    flight_price = f.get("total_price") if "total_price" in f else f.get("price", float("inf"))
                    if flight_price + hotel_total <= budget:
                        affordable.append(h)
                        break 
            travel_data["hotels"] = affordable

        flights = travel_data.get("flights", [])
        hotels = travel_data.get("hotels", [])

        if not flights:
            flight_section = "Based on our search, we unfortunately could not find any available flights from your origin to [Destination City] at this time."
        elif "route" in flights[0]:
            # connecting flight template
            flight_section = """
            Based on our search, we have found the following connecting flight option:
            **Total Flight Price:** [total_price with correct currency symbol]
            * **Leg 1:** [from] ➔ [to] | **Airline:** [airline] (**Flight:** [flight]) | **Dep:** [departure_time] **Arr:** [arrival_time]
            * **Leg 2:** [from] ➔ [to] | **Airline:** [airline] (**Flight:** [flight]) | **Dep:** [departure_time] **Arr:** [arrival_time]
            """
        else:
        #  direct flight template
            flight_section = """
            Based on our search, we have found the following flight option:
            * **Airline:** [Airline Name]
            * **Flight Number:** [Flight Number]
            * **Departure:** [departure_time] | **Arrival:** [arrival_time]
            * **Price:** [Price with correct currency symbol]
            """

        if not hotels:
            hotel_section = "No affordable hotels were found within your budget."
        else:
            hotel_section = """
            Based on our search, we've found excellent options to suit different preferences:

            **1. [Hotel Name]**
                * [Star Emojis] ([Number] Stars) | **Type:** [hotel_type]
                * **Price Per Night:** [Price with correct currency symbol]
                * **Distance from Center:** [distance_from_center_km] km
            
            [Repeat numbered list for additional hotels]
            """

        system_prompt = f"""
        You are a strict data formatter. Your ONLY job is to output the provided <data> into the EXACT Markdown template below.

        CRITICAL RULES:
        1. DO NOT add conversational filler.
        2. FORCE CURRENCY: You MUST use the '$' symbol for ALL prices and budgets.
        3. Do not invent flights or hotels. Use ONLY what is in the <data>.
        4. TOTAL PRICE RULE: Use the lowest total_estimate from trip_cost_calculations. If missing, write "N/A".
        
        YOU MUST USE THIS EXACT TEMPLATE:

        [Greeting tailored to the destination]
        ### ✨ **Your [Destination City] Escape** ✨

        **Destination:** [City Name, Country]
        **Total Budget:** [Budget with correct currency symbol]
        **Total Price:** [Lowest total_estimate with $ symbol]

        ---

        ### ✈️ **Your Flight Details**
        {flight_section}

        ---

        ### 🏨 **Accommodation Options in [Destination City]**
        {hotel_section}

        [Appropriate closing sign-off]
        """

        messages_to_pass = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"<data>\n{travel_data}\n</data>"},
        ]

        response = self.extraction_model.invoke(messages_to_pass)
        response.name = "formatter_output"

        return {"messages": [response]}