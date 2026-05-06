from langchain_core.tools import tool
from tools.tools import create_data_provider


@tool
def get_hotel_filter_options(city: str) -> dict:
    """
    Fetch distinct hotel filtering dimensions for a destination city:
    which star ratings exist and the price range per night.
    Use this to decide what preference question to ask the user about hotels.
    """
    return create_data_provider().get_hotel_dimensions(city)


@tool
def get_flight_filter_options(origin: str, destination: str) -> dict:
    """
    Fetch distinct flight filtering dimensions for a route:
    which airlines operate it and the ticket price range.
    Use this to decide what preference question to ask the user about flights.
    """
    return create_data_provider().get_flight_dimensions(origin, destination)


enrichment_tools = [get_hotel_filter_options, get_flight_filter_options]
enrichment_tool_map = {t.name: t for t in enrichment_tools}
