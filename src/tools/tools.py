from langchain_core.tools import tool
from providers.json_provider import JSONDataProvider
from providers.sqlite_provider import SQLiteDataProvider
from providers.base import BaseDataProvider

def create_data_provider(provider_type: str = "sqlite") -> BaseDataProvider:
    """
    Factory method to create a data provider instance.

    Args:
        provider_type: Type of provider to create ("sqlite" or "json")

    Returns:
        An instance of the requested data provider
    """
    if provider_type == "sqlite":
        return SQLiteDataProvider()
    elif provider_type == "json":
        return JSONDataProvider()
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")


# Initialize the data provider
data_provider = create_data_provider()

@tool
def fetch_flights(origin: str, destination: str):
    """
    Fetch available flights between two cities from the local database.
    Returns a list of flights with prices and availability status.
    """
    return data_provider.fetch_flights(origin, destination)

@tool
def fetch_hotels(city: str, max_price: int = None):
    """
    Fetch available hotels in a city from the local database.
    Optionally filter by maximum price per night.
    Returns a list of hotels with prices and availability status.
    """
    return data_provider.fetch_hotels(city, max_price)

@tool
def calculate_trip_cost(flight_price: float, hotel_price_per_night: float, duration_days: int):
    """
    Calculates the total cost for a trip including flight and hotel stay.
    Input: flight_price (float), hotel_price_per_night (float), duration_days (int).
    """
    try:
        total_hotel = float(hotel_price_per_night) * int(duration_days)
        total_grand = float(flight_price) + total_hotel
        return {
            "breakdown": {
                "flight": flight_price,
                "hotel_total": total_hotel,
                "days": duration_days,
            },
            "total_estimate": total_grand,
            "currency": "USD",
        }
    except (ValueError, TypeError):
        return "Error: Please provide valid numbers for prices and duration."

@tool
def fetch_activities(city: str):
    """
    Fetch available activities, museums, tours, and attractions for a specific city from the local database.
    Returns a list of activities including their category, price, duration, operational days, and closed dates.
    Crucial for building itineraries or checking if a specific venue is open on a given day.
    """
    return data_provider.fetch_activities(city)

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


tools = [fetch_flights, fetch_hotels, calculate_trip_cost, fetch_activities, get_best_time_to_visit, get_average_weather]
