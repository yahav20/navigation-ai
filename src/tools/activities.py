"""LangChain tool for fetching activities and attractions."""
from langchain_core.tools import tool

from tools.dependencies import data_provider


@tool
def fetch_activities(city: str) -> list[dict]:
    """Fetch activities, museums, tours, and attractions for a city, including category, price, duration, and operating days."""
    return data_provider.fetch_activities(city)
