"""Aggregated LangChain tools exposed to the agent."""
from langchain_core.tools import tool

from providers import SQLiteDataProvider

data_provider = SQLiteDataProvider()

# @tool
# def fetch_flights(origin: str, destination: str) -> list[dict]:
#     """Fetch available flights between two cities with prices and availability status."""
#     return data_provider.fetch_flights(origin, destination)
@tool
def fetch_flights(origin: str, destination: str) -> list[dict]:
    """Fetch available flights between two cities. Automatically finds direct flights, and if none exist, finds connecting flights."""
    
    # 1. ניסיון למצוא טיסות ישירות
    direct_flights = data_provider.fetch_flights(origin, destination)
    
    # בדיקה אם חזרו טיסות ישירות אמיתיות (יש להן flight_number)
    has_direct = any("flight_number" in f for f in direct_flights)
    
    if not has_direct:
        # 2. אם אין טיסות ישירות, מחפשים קונקשיינים מאחורי הקלעים!
        connecting_flights = data_provider.find_connecting_flights(origin, destination)
        
        # אם מצאנו קונקשיינים (יש להם 'route'), נחזיר אותם לסוכן כאילו כלום לא קרה
        if connecting_flights and any("route" in f for f in connecting_flights):
            return connecting_flights
            
    # 3. החזרת טיסות ישירות (או הודעת שגיאה אם כלום לא עבד)
    return direct_flights

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

@tool
def find_connecting_flights(origin: str, destination: str) -> list[dict]:
    """
    Fetch connecting flights (1 or 2 stops) between an origin and destination.
    Use this tool ONLY when direct flights (fetch_flights) are unavailable or exceed the user's budget.
    Returns flight routes including total price, intermediate connecting cities, and operating airlines.
    """
    return data_provider.find_connecting_flights(origin, destination)

tools = [fetch_flights, fetch_hotels, calculate_trip_cost, fetch_activities, get_best_time_to_visit, get_average_weather]
