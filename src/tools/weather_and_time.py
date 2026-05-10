"""LangChain tools for weather and best-time-to-visit lookups."""
from langchain_core.tools import tool

from tools.dependencies import data_provider


@tool
def get_best_time_to_visit(city: str) -> dict:
    """Find the recommended months to visit a city and the reasons such as weather or festivals."""
    return data_provider.get_best_time_to_visit(city)


@tool
def get_average_weather(city: str, season: str) -> dict:
    """Get the average temperature for a city in a given season ('Spring', 'Summer', 'Autumn', 'Winter')."""
    return data_provider.get_average_weather(city, season)
