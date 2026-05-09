from langchain_core.tools import tool
from .dependencies import data_provider

@tool
def fetch_flights(origin: str, destination: str):
    """
    Fetch available flights between two cities from the local database.
    Returns a list of flights with prices and availability status.
    """
    return data_provider.fetch_flights(origin, destination)

@tool
def get_flight_filter_options(origin: str, destination: str) -> dict:
    """
    Fetch distinct flight filtering dimensions for a route:
    which airlines operate it and the ticket price range.
    Use this to decide what preference question to ask the user about flights.
    """
    return data_provider.get_flight_dimensions(origin, destination)
