"""Aggregated LangChain tools exposed to the agent."""
from langchain_core.tools import tool

from providers import SQLiteDataProvider

data_provider = SQLiteDataProvider()

@tool
def fetch_flights(origin: str, destination: str) -> list[dict]:
    """Fetch available flights between two cities with prices and availability status."""
    return data_provider.fetch_flights(origin, destination)

@tool
def fetch_hotels(city: str, max_price: int | None = None) -> list[dict]:
    """Fetch available hotels in a city with prices and availability, optionally filtered by maximum price per night."""
    return data_provider.fetch_hotels(city, max_price)

@tool
def calculate_trip_cost(flight_price: float, hotel_price_per_night: float, duration_days: int) -> dict | str:
    """Calculate the total trip cost from flight price, hotel price per night, and duration in days."""
    try:
        total_hotel = float(hotel_price_per_night) * int(duration_days)
        total_grand = float(flight_price) + total_hotel
    except (ValueError, TypeError):
        return "Error: Please provide valid numbers for prices and duration."
    else:
        return {
            "breakdown": {
                "flight": flight_price,
                "hotel_total": total_hotel,
                "days": duration_days,
            },
            "total_estimate": total_grand,
            "currency": "USD",
        }

@tool
def fetch_activities(city: str) -> list[dict]:
    """Fetch activities, museums, tours, and attractions for a city, including category, price, duration, and operating days."""
    return data_provider.fetch_activities(city)

@tool
def get_best_time_to_visit(city: str) -> dict:
    """Find the recommended months to visit a city and the reasons such as weather or festivals."""
    return data_provider.get_best_time_to_visit(city)

@tool
def get_average_weather(city: str, season: str) -> dict:
    """Get the average temperature for a city in a given season ('Spring', 'Summer', 'Autumn', 'Winter')."""
    return data_provider.get_average_weather(city, season)


tools = [fetch_flights, fetch_hotels, calculate_trip_cost, fetch_activities, get_best_time_to_visit, get_average_weather]
