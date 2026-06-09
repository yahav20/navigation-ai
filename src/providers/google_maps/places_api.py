"""Google Maps Places API wrapper for searching attractions, restaurants, and hotels."""
from __future__ import annotations

import os
from typing import Any

import requests
from providers.http_client import get as _http_get

_BASE = "https://maps.googleapis.com/maps/api"
_TIMEOUT = 15

_place_search_cache: dict[str, list[dict]] = {}


def _api_key() -> str | None:
    return os.getenv("GOOGLE_MAPS_API_KEY")


def search_places(query: str, location: tuple[float, float] | None = None, radius: int = 50000) -> list[dict]:
    """Search for places using Google Places Text Search API.

    Args:
        query: Search query (e.g., "museums in Paris")
        location: Optional (lat, lng) to bias results toward a location
        radius: Search radius in meters (default 50km)

    Returns: List of normalized place dicts with name, rating, price_level, lat, lng, types
    """
    api_key = _api_key()
    if not api_key:
        return []

    cache_key = f"{query}:{location}:{radius}"
    if cache_key in _place_search_cache:
        return _place_search_cache[cache_key]

    params: dict[str, Any] = {
        "query": query,
        "key": api_key,
    }
    if location:
        params["location"] = f"{location[0]},{location[1]}"
        params["radius"] = radius

    try:
        r = _http_get(f"{_BASE}/place/textsearch/json", params=params, timeout=_TIMEOUT)
        r.raise_for_status()
        data = r.json() or {}
    except (requests.RequestException, ValueError):
        return []

    if data.get("status") != "OK":
        return []

    results = []
    for result in data.get("results") or []:
        normalized = _normalize_place(result)
        if normalized:
            results.append(normalized)

    _place_search_cache[cache_key] = results
    return results


def search_nearby_places(
    location: tuple[float, float],
    place_type: str,
    radius: int = 5000,
    limit: int = 20,
) -> list[dict]:
    """Search for nearby places of a specific type using Nearby Search.

    Args:
        location: (lat, lng) center point
        place_type: Place type (e.g., "museum", "restaurant", "hotel")
        radius: Search radius in meters
        limit: Maximum results to return

    Returns: List of normalized place dicts
    """
    api_key = _api_key()
    if not api_key:
        return []

    params = {
        "location": f"{location[0]},{location[1]}",
        "type": place_type,
        "radius": radius,
        "key": api_key,
        "rankby": "prominence",
    }

    try:
        r = _http_get(f"{_BASE}/place/nearbysearch/json", params=params, timeout=_TIMEOUT)
        r.raise_for_status()
        data = r.json() or {}
    except (requests.RequestException, ValueError):
        return []

    if data.get("status") != "OK":
        return []

    results = []
    for result in data.get("results") or []:
        normalized = _normalize_place(result)
        if normalized:
            results.append(normalized)
        if len(results) >= limit:
            break

    return results


def get_place_details(place_id: str) -> dict | None:
    """Fetch detailed information about a place.

    Returns dict with hours, phone, website, etc., or None on error.
    """
    api_key = _api_key()
    if not api_key:
        return None

    params = {
        "place_id": place_id,
        "fields": "name,rating,price_level,formatted_phone_number,website,opening_hours,photos",
        "key": api_key,
    }

    try:
        r = _http_get(f"{_BASE}/place/details/json", params=params, timeout=_TIMEOUT)
        r.raise_for_status()
        data = r.json() or {}
    except (requests.RequestException, ValueError):
        return None

    if data.get("status") != "OK":
        return None

    result = data.get("result")
    if not result:
        return None

    return {
        "name": result.get("name"),
        "rating": result.get("rating"),
        "price_level": result.get("price_level"),
        "phone": result.get("formatted_phone_number"),
        "website": result.get("website"),
        "hours": result.get("opening_hours", {}).get("weekday_text"),
    }


def normalize_rating(api_rating: float | None) -> float | None:
    """Convert Google rating (0-5) to 1-5 scale to match SQLite.
    
    Formula: max(1.0, (api_rating * 4 / 5) + 1)
    """
    if api_rating is None:
        return None
    return max(1.0, (api_rating * 4 / 5) + 1)


def _normalize_place(place: dict) -> dict | None:
    """Normalize a Google Places result to agent schema."""
    name = place.get("name")
    if not name:
        return None

    geo = place.get("geometry", {}).get("location") or {}
    rating = place.get("rating")
    price_level = place.get("price_level")

    return {
        "name": name,
        "place_id": place.get("place_id"),
        "rating": normalize_rating(rating),
        "review_count": place.get("user_ratings_total"),
        "price_level": price_level,
        "price_level_text": _price_level_text(price_level),
        "lat": geo.get("lat"),
        "lng": geo.get("lng"),
        "types": place.get("types") or [],
        "formatted_address": place.get("formatted_address"),
        "photo_reference": (place.get("photos") or [{}])[0].get("photo_reference"),
    }


def _price_level_text(level: int | None) -> str:
    """Convert Google price_level (0-4) to text description."""
    if level is None:
        return "Unknown"
    price_map = {0: "Free", 1: "$", 2: "$$", 3: "$$$", 4: "$$$$"}
    return price_map.get(level, "Unknown")
