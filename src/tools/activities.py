from langchain_core.tools import tool
from .dependencies import data_provider

@tool
def fetch_activities(city: str):
    """
    Fetch available activities, museums, tours, and attractions for a specific city from the local database.
    Returns a list of activities including their category, price, duration, operational days, and closed dates.
    Crucial for building itineraries or checking if a specific venue is open on a given day.
    """
    return data_provider.fetch_activities(city)
