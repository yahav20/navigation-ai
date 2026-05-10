"""LangChain tools for fetching hotel options and filter dimensions."""
from langchain_core.tools import tool

from tools.dependencies import data_provider


@tool
def fetch_hotels(city: str, max_price: int | None = None) -> list[dict]:
    """Fetch available hotels in a city with prices and availability, optionally filtered by maximum price per night."""
    return data_provider.fetch_hotels(city, max_price)


@tool
def get_hotel_filter_options(city: str) -> dict:
    """Fetch distinct hotel filtering dimensions for a destination city, including available star ratings and the price-per-night range."""
    return data_provider.get_hotel_dimensions(city)
