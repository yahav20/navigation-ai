from langchain_core.tools import tool
from providers.json_provider import JSONDataProvider
from providers.base import BaseDataProvider

def create_data_provider(provider_type: str = "json") -> BaseDataProvider:
    """
    Factory method to create a data provider instance.
    
    Args:
        provider_type: Type of provider to create (default: "json")
        
    Returns:
        An instance of the requested data provider
    """
    if provider_type == "json":
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
def fetch_hotels(city: str):
    """
    Fetch available hotels in a city from the local database.
    Returns a list of hotels with prices and availability status.
    """
    return data_provider.fetch_hotels(city)

@tool
def calculate_trip_cost(flight_price: float, hotel_price_per_night: float, nights: int):
    """
    Calculate the total cost of a trip based on flight price and hotel price per night.
    """
    total_cost = flight_price + (hotel_price_per_night * nights)
    # Add 10% service fee
    total_cost = total_cost * 1.10
    
    return f"The total trip cost, including a 10% service fee, is ${total_cost:.2f}."

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
