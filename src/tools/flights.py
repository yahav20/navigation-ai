"""LangChain tools for fetching flight options and filter dimensions."""
from langchain_core.tools import tool

from tools.dependencies import data_provider


@tool
def fetch_flights(origin: str, destination: str) -> list[dict]:
    """Fetch available flights between two cities with prices and availability status."""
    return data_provider.fetch_flights(origin, destination)


@tool
def get_flight_filter_options(origin: str, destination: str) -> dict:
    """Fetch distinct flight filtering dimensions for a route, including operating airlines and the ticket price range."""
    return data_provider.get_flight_dimensions(origin, destination)
