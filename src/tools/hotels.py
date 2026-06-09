"""LangChain tools for fetching hotel options — SQLite, Google Maps, and Xotelo."""
from langchain_core.tools import tool

from providers.google_maps import geocode_city, search_nearby_places
from providers.tripadvisor_lookup import scrape_ta_location_key
from providers.xotelo import xotelo_get_rates, xotelo_list_hotels
from security import validate_city, validate_positive_number
from tools.dependencies import data_provider

# Hardcoded TripAdvisor location keys for the most common destinations.
# Used as a fast first-check before falling back to the dynamic TypeAhead lookup.
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
    "tel aviv": "g293984",
    "berlin": "g187323",
    "bangkok": "g293916",
    "singapore": "g294265",
    "istanbul": "g298650",
    "prague": "g274707",
    "vienna": "g190454",
    "lisbon": "g189158",
    "madrid": "g187514",
    "athens": "g189400",
}


def _resolve_ta_key(city: str) -> str | None:
    """Return a TripAdvisor location key for any city.

    Checks the hardcoded dict first (instant), then falls back to the
    DAL's DB cache, and finally to the TypeAhead API lookup.
    """
    key = city.lower().strip()

    # 1. Hardcoded fast path
    if key in TRIPADVISOR_LOCATION_KEYS:
        return TRIPADVISOR_LOCATION_KEYS[key]

    # 2. DAL DB cache (covers cities previously looked up at runtime)
    city_id = data_provider._get_city_id(city)
    if city_id:
        cached = data_provider.sqlite._query(
            "SELECT ta_key FROM cities WHERE id = ?", (city_id,)
        )
        if cached and cached[0].get("ta_key"):
            return cached[0]["ta_key"]

    # 3. Live TypeAhead lookup → persist to DB for next time
    ta_key = scrape_ta_location_key(city)
    if ta_key and city_id:
        data_provider._save_ta_key(city_id, ta_key)
    return ta_key


# ── SQLite/DAL-backed tools ──────────────────────────────────────────────────

@tool
def fetch_hotels(city: str, max_price: int | None = None) -> list[dict]:
    """Fetch available hotels in a city with prices and availability, optionally filtered by maximum price per night."""
    return data_provider.fetch_hotels(city, max_price)


@tool
def get_hotel_filter_options(city: str) -> dict:
    """Fetch distinct hotel filtering dimensions for a destination city, including available star ratings and the price-per-night range."""
    return data_provider.get_hotel_dimensions(city)


# ── Google Maps-backed tools ─────────────────────────────────────────────────

@tool
def fetch_hotels_gm(city: str, max_price: int | None = None) -> list[dict]:
    """Fetch hotels in a city using Google Maps Places API.

    Returns available hotels with ratings, price levels, and locations.
    Falls back to the hybrid DAL (Xotelo + local fixtures) when no Google
    Maps API key is configured or the geocoding fails.
    Optionally filter by maximum price level (1-4).
    """
    city = validate_city(city)
    if max_price is not None:
        validate_positive_number(max_price, "max_price")

    coords = geocode_city(city)
    if not coords:
        return data_provider.fetch_hotels(city, max_price)

    results = search_nearby_places(coords, place_type="lodging", radius=15000, limit=15)
    if not results:
        return data_provider.fetch_hotels(city, max_price)

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


# ── Xotelo (TripAdvisor-backed) tools ───────────────────────────────────────

@tool
def fetch_hotels_xotelo(city: str, check_in: str, check_out: str, currency: str = "USD") -> list[dict]:
    """Fetch hotels with live OTA rates from Xotelo (TripAdvisor-backed).

    Works for any city — resolves the TripAdvisor location key dynamically
    when the city is not in the built-in lookup table.
    Falls back to the hybrid DAL (without per-date rates) if no key can be found.

    Args:
        city: Destination city
        check_in: Check-in date (YYYY-MM-DD)
        check_out: Check-out date (YYYY-MM-DD)
        currency: Currency code (default USD)
    """
    city = validate_city(city)

    location_key = _resolve_ta_key(city)
    if not location_key:
        return data_provider.fetch_hotels(city)

    hotels = xotelo_list_hotels(location_key, limit=10)
    if not hotels:
        return data_provider.fetch_hotels(city)

    for hotel in hotels:
        hotel_key = hotel.get("hotel_key")
        if hotel_key:
            rates = xotelo_get_rates(hotel_key, check_in, check_out, currency)
            hotel["rates"] = rates

    return hotels


@tool
def fetch_hotels_with_ratings_xotelo(city: str, limit: int = 10) -> list[dict]:
    """Fetch top-rated hotels in a city from Xotelo.

    Works for any city — resolves the TripAdvisor location key dynamically.
    Falls back to the hybrid DAL when no key can be found.
    Returns hotels without per-date rates (faster).
    """
    city = validate_city(city)

    location_key = _resolve_ta_key(city)
    if not location_key:
        return data_provider.fetch_hotels(city)

    hotels = xotelo_list_hotels(location_key, limit=limit, sort="rating")
    return hotels if hotels else data_provider.fetch_hotels(city)
