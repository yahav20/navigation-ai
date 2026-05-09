from langchain_core.tools import tool
from .dependencies import data_provider

@tool
def get_best_time_to_visit(city: str):
    """
    Find the recommended months to visit a specific city and the underlying reasons (e.g., weather, festivals).
    Use this when a user is unsure about when to travel to a destination.
    """
    return data_provider.get_best_time_to_visit(city)

@tool
def get_average_weather(city: str, season: str):
    """
    Get the average temperature for a specific city during a specific season.
    Valid seasons are: 'Spring', 'Summer', 'Autumn', 'Winter'.
    """
    return data_provider.get_average_weather(city, season)
