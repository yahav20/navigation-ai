"""Google Maps Geocoding API wrapper for resolving city names to coordinates."""
from __future__ import annotations

import os
from typing import Any

import requests
from providers.http_client import get as _http_get

_BASE = "https://maps.googleapis.com/maps/api"
_TIMEOUT = 15

_geocode_cache: dict[str, tuple[float, float]] = {}


def _api_key() -> str | None:
    return os.getenv("GOOGLE_MAPS_API_KEY")


def geocode_city(city: str) -> tuple[float, float] | None:
    """Resolve a city name to latitude/longitude coordinates.

    Returns (lat, lng) tuple or None if not found. Results are cached.
    """
    if not city:
        return None

    key = city.strip().lower()
    if key in _geocode_cache:
        return _geocode_cache[key]

    api_key = _api_key()
    if not api_key:
        return None

    try:
        r = _http_get(
            f"{_BASE}/geocode/json",
            params={"address": city, "key": api_key},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json() or {}
    except (requests.RequestException, ValueError):
        return None

    if not data.get("results"):
        return None

    location = data["results"][0].get("geometry", {}).get("location")
    if not location:
        return None

    coords = (location.get("lat"), location.get("lng"))
    if coords[0] and coords[1]:
        _geocode_cache[key] = coords
        return coords

    return None
