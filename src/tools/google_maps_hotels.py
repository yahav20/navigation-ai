"""LangChain tools for Google Maps hotels API."""
from langchain_core.tools import tool

from providers.google_maps import geocode_city, search_places, search_nearby_places
from security import validate_city, validate_positive_number


@tool
def fetch_hotels_gm(city: str, max_price: int | None = None) -> list[dict]:
    """Fetch hotels in a city using Google Maps Places API.

    Returns available hotels with ratings, price levels, and locations.
    Optionally filter by maximum price level (1-4).
    """
    city = validate_city(city)
    if max_price is not None:
        validate_positive_number(max_price, "max_price")

    coords = geocode_city(city)
    if not coords:
        return []

    results = search_nearby_places(coords, place_type="lodging", radius=15000, limit=15)

    if max_price is not None:
        results = [h for h in results if (h.get("price_level") or 0) <= max_price]

    return results


@tool
def get_hotel_filter_options_gm(city: str) -> dict:
    """Get available price levels and ratings for hotels in a city via Google Maps."""
    city = validate_city(city)
    coords = geocode_city(city)
    if not coords:
        return {}

    hotels = search_nearby_places(coords, place_type="lodging", radius=15000, limit=20)

    price_levels = set()
    ratings = []
    for hotel in hotels:
        if hotel.get("price_level") is not None:
            price_levels.add(hotel["price_level"])
        if hotel.get("rating"):
            ratings.append(hotel["rating"])

    avg_rating = sum(ratings) / len(ratings) if ratings else None

    return {
        "available_price_levels": sorted(list(price_levels)),
        "price_level_range": (min(price_levels), max(price_levels)) if price_levels else None,
        "average_rating": avg_rating,
        "total_hotels": len(hotels),
    }
