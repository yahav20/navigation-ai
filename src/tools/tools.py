"""Aggregated LangChain tools exposed to the agent."""
from langchain_core.tools import tool

from providers import SQLiteDataProvider
from providers.flights import search_flights_with_fallback
from security import validate_city, validate_positive_number, validate_season

# Import new API-backed tools
from tools.google_maps_hotels import fetch_hotels_gm, get_hotel_filter_options_gm
from tools.google_maps_attractions import fetch_attractions, fetch_restaurants, fetch_landmarks
from tools.xotelo_hotels import fetch_hotels_xotelo, fetch_hotels_with_ratings_xotelo
from tools.geocoding import geocode_location
from tools.attraction_enrichment import fetch_attraction_details, lookup_wikidata

data_provider = SQLiteDataProvider()


@tool
def fetch_flights(origin: str, destination: str, departure_at: str | None = None) -> list[dict]:
    """Fetch available flights between two cities for an approximate departure date.

    `departure_at` is YYYY-MM-DD for a specific day or YYYY-MM for a whole month.
    When omitted, defaults to roughly one month from today (month-wide query).
    API-first with SQLite fallback.
    """
    origin = validate_city(origin)
    destination = validate_city(destination)
    return search_flights_with_fallback(origin, destination, departure_at)

@tool
def fetch_hotels(city: str, max_price: int | None = None) -> list[dict]:
    """Fetch available hotels in a city with prices and availability, optionally filtered by maximum price per night."""
    city = validate_city(city)
    if max_price is not None:
        validate_positive_number(max_price, "max_price")
    return data_provider.fetch_hotels(city, max_price)

@tool
def calculate_trip_cost(flight_price: float, hotel_price_per_night: float, duration_days: int) -> dict | str:
    """Calculate the total trip cost from flight price, hotel price per night, and duration in days."""
    try:
        validate_positive_number(flight_price, "flight_price")
        validate_positive_number(hotel_price_per_night, "hotel_price_per_night")
        validate_positive_number(duration_days, "duration_days")
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
    city = validate_city(city)
    return data_provider.fetch_activities(city)

@tool
def get_best_time_to_visit(city: str) -> dict:
    """Find the recommended months to visit a city and the reasons such as weather or festivals."""
    city = validate_city(city)
    return data_provider.get_best_time_to_visit(city)

@tool
def get_average_weather(city: str, season: str) -> dict:
    """Get the average temperature for a city in a given season ('Spring', 'Summer', 'Autumn', 'Winter')."""
    city = validate_city(city)
    season = validate_season(season)
    return data_provider.get_average_weather(city, season)

@tool
def find_connecting_flights(origin: str, destination: str, departure_at: str | None = None) -> list[dict]:
    """
    Fetch connecting flights (1+ stops) between an origin and destination.
    Use this tool ONLY when direct flights (fetch_flights) are unavailable or exceed the user's budget.
    Returns connecting offers with total price and per-leg route detail. API-first with SQLite fallback.
    """
    origin = validate_city(origin)
    destination = validate_city(destination)
    offers = search_flights_with_fallback(origin, destination, departure_at)
    return [
        o for o in offers
        if isinstance(o, dict) and ((o.get("transfers") or 0) >= 1 or o.get("route"))
    ]

tools = [
    # Original SQLite-backed tools (kept for backward compatibility & fallback)
    fetch_flights,
    fetch_hotels,
    calculate_trip_cost,
    fetch_activities,
    get_best_time_to_visit,
    get_average_weather,
    find_connecting_flights,
    # New Google Maps API-backed tools
    fetch_hotels_gm,
    get_hotel_filter_options_gm,
    fetch_attractions,
    fetch_restaurants,
    fetch_landmarks,
    # New Xotelo API-backed tools
    fetch_hotels_xotelo,
    fetch_hotels_with_ratings_xotelo,
    # New geocoding tool
    geocode_location,
    # New Wikipedia/Wikivoyage enrichment tools
    fetch_attraction_details,
    lookup_wikidata,
]
