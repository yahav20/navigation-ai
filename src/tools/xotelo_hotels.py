"""LangChain tools for Xotelo hotel API."""
from langchain_core.tools import tool

from providers.google_maps import geocode_city
from providers.xotelo import xotelo_list_hotels, xotelo_get_rates
from security import validate_city


# Common TripAdvisor location keys for popular destinations
TRIPADVISOR_LOCATION_KEYS = {
    "paris": "g187147",
    "london": "g186338",
    "new york": "g60763",
    "tokyo": "g298184",
    "sydney": "g255060",
    "barcelona": "g187497",
    "rome": "g187791",
    "amsterdam": "g188590",
    "dubai": "g295424",
    "los angeles": "g32655",
}


@tool
def fetch_hotels_xotelo(city: str, check_in: str, check_out: str, currency: str = "USD") -> list[dict]:
    """Fetch hotels with live OTA rates from Xotelo (TripAdvisor-backed).

    Args:
        city: Destination city
        check_in: Check-in date (YYYY-MM-DD)
        check_out: Check-out date (YYYY-MM-DD)
        currency: Currency code (default USD)

    Returns: Hotels with available OTA rates and prices
    """
    city = validate_city(city)

    location_key = TRIPADVISOR_LOCATION_KEYS.get(city.lower())
    if not location_key:
        return []

    hotels = xotelo_list_hotels(location_key, limit=10)

    for hotel in hotels:
        hotel_key = hotel.get("hotel_key")
        if hotel_key:
            rates = xotelo_get_rates(hotel_key, check_in, check_out, currency)
            hotel["rates"] = rates

    return hotels


@tool
def fetch_hotels_with_ratings_xotelo(city: str, limit: int = 10) -> list[dict]:
    """Fetch top-rated hotels in a city from Xotelo.

    Returns hotels without rates (faster, no date filtering needed).
    """
    city = validate_city(city)
    location_key = TRIPADVISOR_LOCATION_KEYS.get(city.lower())
    if not location_key:
        return []

    return xotelo_list_hotels(location_key, limit=limit, sort="rating")
