"""Xotelo API wrapper for fetching hotel listings and rates from TripAdvisor."""
from __future__ import annotations

import requests

_XOTELO_BASE = "https://data.xotelo.com/api"
_TIMEOUT = 20


def xotelo_list_hotels(location_key: str, limit: int = 10, sort: str = "popularity") -> list[dict]:
    """Fetch hotel listings for a TripAdvisor location.

    Args:
        location_key: TripAdvisor location key (e.g., "g187147" for Paris)
        limit: Maximum hotels to return
        sort: Sort order ("popularity", "rating", "price")

    Returns: List of hotel dicts with name, rating, address, price_ranges, key
    """
    try:
        r = requests.get(
            f"{_XOTELO_BASE}/list",
            params={
                "location_key": location_key,
                "limit": limit,
                "sort": sort,
            },
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json() or {}
    except (requests.RequestException, ValueError):
        return []

    return _normalize_hotels((data.get("result") or {}).get("list") or [])


def xotelo_get_rates(hotel_key: str, check_in: str, check_out: str, currency: str = "USD") -> list[dict]:
    """Fetch live OTA rates for a hotel on specific dates.

    Args:
        hotel_key: Xotelo hotel key (from xotelo_list_hotels)
        check_in: Check-in date (YYYY-MM-DD)
        check_out: Check-out date (YYYY-MM-DD)
        currency: Currency code (default USD)

    Returns: List of rate dicts with OTA, price_per_night, availability
    """
    try:
        r = requests.get(
            f"{_XOTELO_BASE}/rates",
            params={
                "hotel_key": hotel_key,
                "chk_in": check_in,
                "chk_out": check_out,
                "currency": currency,
            },
            timeout=_TIMEOUT,
        )
        if r.status_code != 200:
            return []
        data = r.json() or {}
    except (requests.RequestException, ValueError):
        return []

    return _normalize_rates((data.get("result") or {}).get("rates") or [])


def normalize_rating(api_rating: float | None) -> float | None:
    """Convert Xotelo/TripAdvisor rating (0-5) to 1-5 scale.
    
    Formula: max(1.0, (api_rating * 4 / 5) + 1)
    """
    if api_rating is None:
        return None
    return max(1.0, (api_rating * 4 / 5) + 1)


# Fixed USD estimates per star tier (used when Xotelo has no price range)
_STARS_TO_PRICE = {
    (4.6, 5.1): 300.0,
    (4.0, 4.6): 180.0,
    (3.5, 4.0): 120.0,
    (0.0, 3.5):  80.0,
}


def _stars_to_price_estimate(stars: float | None) -> float:
    """Return a fixed USD/night estimate based on normalized star rating."""
    if stars is None:
        return 120.0
    for (lo, hi), price in _STARS_TO_PRICE.items():
        if lo <= stars < hi:
            return price
    return 120.0


def _normalize_hotels(hotels: list) -> list[dict]:
    """Normalize Xotelo hotel listing to agent schema."""
    results = []
    for hotel in hotels:
        if not isinstance(hotel, dict):
            continue

        name = hotel.get("name")
        if not name:
            continue

        review = hotel.get("review_summary") or {}
        # Xotelo uses "minimum"/"maximum" field names (not "min"/"max")
        price_range = hotel.get("price_ranges") or {}
        geo = hotel.get("geo") or {}

        p_min = price_range.get("minimum") or price_range.get("min")
        p_max = price_range.get("maximum") or price_range.get("max")

        if p_min is not None and p_max is not None:
            avg_price = round((float(p_min) + float(p_max)) / 2, 2)
        elif p_min is not None:
            avg_price = round(float(p_min), 2)
        elif p_max is not None:
            avg_price = round(float(p_max), 2)
        else:
            # No price range from API — derive fixed estimate from star rating
            stars = normalize_rating(review.get("rating"))
            avg_price = _stars_to_price_estimate(stars)

        normalized = {
            "name": name,
            "hotel_key": hotel.get("key"),
            "stars": normalize_rating(review.get("rating")),
            "review_count": review.get("review_count"),
            "address": hotel.get("address"),
            "lat": geo.get("lat"),
            "lng": geo.get("lng"),
            "price_per_night": avg_price,
            "currency": price_range.get("currency", "USD"),
            "tripadvisor_url": hotel.get("url"),
        }
        results.append(normalized)

    return results


def _normalize_rates(rates: list) -> list[dict]:
    """Normalize Xotelo rate data to agent schema."""
    results = []
    for rate in rates:
        if not isinstance(rate, dict):
            continue

        ota = rate.get("ota_name")
        price = rate.get("price")
        if not ota or price is None:
            continue

        normalized = {
            "ota": ota,
            "price_per_night": price,
            "total_price": rate.get("total_price"),
            "currency": rate.get("currency", "USD"),
            "availability": "available" if rate.get("available") else "unavailable",
            "url": rate.get("url"),
        }
        results.append(normalized)

    return results
