"""LangChain tools for fetching filter dimensions used during enrichment."""
from langchain_core.tools import tool

from providers import SQLiteDataProvider

data_provider = SQLiteDataProvider()


@tool
def get_hotel_filter_options(city: str) -> dict:
    """Fetch distinct hotel filtering dimensions for a destination city, including available star ratings and the price-per-night range."""
    return data_provider.get_hotel_dimensions(city)


@tool
def get_flight_filter_options(origin: str, destination: str) -> dict:
    """Fetch distinct flight filtering dimensions for a route, including operating airlines and the ticket price range."""
    return data_provider.get_flight_dimensions(origin, destination)


enrichment_tools = [get_hotel_filter_options, get_flight_filter_options]
enrichment_tool_map = {t.name: t for t in enrichment_tools}
