"""LangChain tools for fetching flight options and filter dimensions."""
from langchain_core.tools import tool
from tools.dependencies import data_provider


# @tool
# def fetch_flights(origin: str, destination: str) -> list[dict]:
#     """Fetch available flights between two cities with prices and availability status."""
#     return data_provider.fetch_flights(origin, destination)
@tool
def fetch_flights(origin: str, destination: str) -> list[dict]:
    """Fetch available flights between two cities. Automatically finds direct flights, and if none exist, finds connecting flights."""
    
    direct_flights = data_provider.fetch_flights(origin, destination)
    
    has_direct = any("flight_number" in f for f in direct_flights)
    
    if not has_direct:
        connecting_flights = data_provider.find_connecting_flights(origin, destination)
        
        if connecting_flights and any("route" in f for f in connecting_flights):
            return connecting_flights
     
    return direct_flights

@tool
def find_connecting_flights(origin: str, destination: str) -> list[dict]:
    """
    Fetch connecting flights (1 or 2 stops) between an origin and destination.
    Use this tool ONLY when direct flights (fetch_flights) are unavailable or exceed the user's budget.
    Returns flight routes including total price, intermediate connecting cities, and operating airlines.
    """
    return data_provider.find_connecting_flights(origin, destination)


@tool
def get_flight_filter_options(origin: str, destination: str) -> dict:
    """Fetch distinct flight filtering dimensions for a route, including operating airlines and the ticket price range."""
    return data_provider.get_flight_dimensions(origin, destination)