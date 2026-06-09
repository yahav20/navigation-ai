"""LangChain tools for fetching activities, attractions, and attraction enrichment."""
from langchain_core.tools import tool

from providers.google_maps import geocode_city, search_places, search_nearby_places
from providers.wikipedia import fetch_wiki_summary, fetch_wikidata_id
from security import validate_city
from tools.dependencies import data_provider


# ── SQLite-backed tool ───────────────────────────────────────────────────────

@tool
def fetch_activities(city: str) -> list[dict]:
    """Fetch activities, museums, tours, and attractions for a city, including category, price, duration, and operating days."""
    return data_provider.fetch_activities(city)


# ── Google Maps-backed tools ─────────────────────────────────────────────────

@tool
def fetch_attractions(city: str, query: str = "museums and attractions") -> list[dict]:
    """Fetch attractions, museums, and tourist sites in a city using Google Maps.

    Returns attractions with ratings, locations, and price levels.
    """
    city = validate_city(city)
    coords = geocode_city(city)
    if not coords:
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


# ── Wikipedia/Wikidata enrichment tools ──────────────────────────────────────

@tool
def fetch_attraction_details(attraction_name: str) -> dict:
    """Fetch Wikipedia details about an attraction.

    Returns a one-liner description, Wikipedia URL, and image URL if available.
    """
    if not attraction_name:
        return {}

    wiki_data = fetch_wiki_summary(attraction_name)
    if not wiki_data:
        return {"error": f"No Wikipedia article found for '{attraction_name}'"}

    return {
        "name": attraction_name,
        "description": wiki_data.get("extract"),
        "wikipedia_url": wiki_data.get("url"),
        "image_url": wiki_data.get("image_url"),
        "wikidata_id": fetch_wikidata_id(attraction_name),
    }


@tool
def lookup_wikidata(attraction_name: str) -> dict:
    """Look up a Wikidata Q-ID for an attraction by Wikipedia title.

    Useful for cross-referencing with Wikivoyage and other Wikidata-backed systems.
    """
    if not attraction_name:
        return {}

    qid = fetch_wikidata_id(attraction_name)
    if not qid:
        return {"error": f"No Wikidata entry found for '{attraction_name}'"}

    return {
        "name": attraction_name,
        "wikidata_id": qid,
        "wikidata_url": f"https://www.wikidata.org/wiki/{qid}",
    }
