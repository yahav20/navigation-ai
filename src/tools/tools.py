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
@tool
def fetch_connecting_flights(origin: str, destination: str, max_budget: float):
    """
    Search for connecting flights (1 or 2 stops) between two cities within a specific budget.
    Returns the route breakdown, flight numbers, layover cities, and total combined price.
    Use this when direct flights are unavailable or when the user wants to find cheaper alternatives within their budget.
    """
    return data_provider.fetch_connecting_flights(origin, destination, max_budget)

@tool
def search_best_flight_route(origin: str, destination: str, max_budget: float = None):
    """
    Search for the best flight route between two cities.
    It checks for direct flights first. If none are found, or if they exceed the optional budget,
    it automatically searches for connecting flights (up to 2 stops).
    It also suggests alternative cities in the same country if the exact destination is unreachable.
    """
    # 1. Search direct flights
    direct_results = data_provider.fetch_flights(origin, destination)
    
    valid_direct = []
    is_direct_found = False
    
    # Check whether these are exact direct results or an alternative fallback
    if direct_results and isinstance(direct_results[0], dict):
        if "alternatives" in direct_results[0]:
            # Returned direct-flight fallback to the country
            valid_direct = direct_results[0]["alternatives"]
            if max_budget is not None:
                valid_direct = [f for f in valid_direct if f.get('price', float('inf')) <= max_budget]
        elif "message" not in direct_results[0]:
             # Returned normal direct flights
             is_direct_found = True
             for flight in direct_results:
                 if max_budget is None or flight.get('price', float('inf')) <= max_budget:
                     valid_direct.append(flight)
    
    if valid_direct:
        valid_direct.sort(key=lambda x: x.get('price', 0))
        return {
            "route_type": "Direct Flights (Exact or Alternatives)", 
            "options": valid_direct[:5]
        }
        
    # 2. Search connecting flights (used if no direct flights or direct options are too expensive)
    connections = data_provider.fetch_connecting_flights(origin, destination, max_budget)
        
    if connections and isinstance(connections[0], dict):
         reason = "Direct flights exceeded budget." if (is_direct_found and max_budget) else "No direct flights available."
         
         # Check whether connecting flights are an alternative fallback
         if "alternatives" in connections[0]:
             return {
                 "route_type": "Connecting Flights to Alternative Cities",
                 "note": f"{reason} " + connections[0].get("message", ""),
                 "options": connections[0]["alternatives"]
             }
         # Regular connecting flights to the exact city
         elif "message" not in connections[0]:
             return {
                 "route_type": "Connecting Flights (1 or 2 stops)", 
                 "note": reason,
                 "options": connections
             }
             
    # 3. If everything fails
    if is_direct_found and max_budget:
         return {"message": f"Found direct flights, but all exceed ${max_budget}. No connecting flights under budget either."}
         
    msg = f"No flights (direct or connecting) found from {origin} to {destination}"
    msg += f" within ${max_budget}." if max_budget else "."
    return {"message": msg}

tools = [fetch_flights, fetch_hotels, calculate_trip_cost, fetch_activities, get_best_time_to_visit, get_average_weather,search_best_flight_route]
