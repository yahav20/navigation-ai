from langchain_core.tools import tool
from .dependencies import data_provider

@tool
def fetch_hotels(city: str, max_price: int = None):
    """
    Fetch available hotels in a city from the local database.
    Optionally filter by maximum price per night.
    Returns a list of hotels with prices and availability status.
    """
    return data_provider.fetch_hotels(city, max_price)

@tool
def get_hotel_filter_options(city: str) -> dict:
    """
    Fetch distinct hotel filtering dimensions for a destination city:
    which star ratings exist and the price range per night.
    Use this to decide what preference question to ask the user about hotels.
    """
    return data_provider.get_hotel_dimensions(city)
