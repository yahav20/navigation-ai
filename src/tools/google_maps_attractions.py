"""LangChain tools for Google Maps attractions and restaurants."""
from langchain_core.tools import tool

from providers.google_maps import geocode_city, search_places, search_nearby_places
from security import validate_city


@tool
def fetch_attractions(city: str, query: str = "museums and attractions") -> list[dict]:
    """Fetch attractions, museums, and tourist sites in a city using Google Maps.

    Returns attractions with ratings, locations, and price levels.
    """
    city = validate_city(city)
    coords = geocode_city(city)
    if not coords:
        # Fallback to text search if geocoding fails
        return search_places(f"{query} in {city}", limit=15)

    return search_nearby_places(coords, place_type="museum", radius=15000, limit=15)


@tool
def fetch_restaurants(city: str, cuisine: str | None = None) -> list[dict]:
    """Fetch restaurants in a city using Google Maps Places API.

    Returns restaurants with ratings, locations, and price levels.
    Optionally filter by cuisine type (e.g., "Italian", "Japanese").
    """
    city = validate_city(city)
    coords = geocode_city(city)

    if cuisine:
        query = f"{cuisine} restaurants in {city}"
    else:
        query = f"restaurants in {city}"

    if coords:
        results = search_nearby_places(coords, place_type="restaurant", radius=10000, limit=20)
    else:
        results = search_places(query, limit=20)

    return results


@tool
def fetch_landmarks(city: str) -> list[dict]:
    """Fetch landmarks and historic sites in a city using Google Maps."""
    city = validate_city(city)
    coords = geocode_city(city)

    if not coords:
        return search_places(f"landmarks in {city}", limit=15)

    return search_nearby_places(coords, place_type="point_of_interest", radius=15000, limit=15)
