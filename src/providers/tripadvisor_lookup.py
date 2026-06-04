"""TripAdvisor location key resolver.

Uses TripAdvisor's internal TypeAhead JSON API (the same endpoint the browser
uses for autocomplete) to resolve a city name to a gXXXXXX geographic ID used
by both TripAdvisor and the Xotelo API as a location_key.
"""
from __future__ import annotations

import logging
import warnings

import requests
import urllib3

_logger = logging.getLogger(__name__)
_TIMEOUT = 15
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/javascript, */*",
    "Referer": "https://www.tripadvisor.com/",
    "X-Requested-With": "XMLHttpRequest",
    "Accept-Language": "en-US,en;q=0.9",
}


def scrape_ta_location_key(city_name: str) -> str | None:
    """Return the TripAdvisor gXXXXXX location key for a city.

    Queries the TypeAhead JSON API that powers TripAdvisor's browser autocomplete.
    Returns None if the city cannot be resolved or the request fails.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", urllib3.exceptions.InsecureRequestWarning)
            r = requests.get(
                "https://www.tripadvisor.com/TypeAheadJson",
                params={
                    "action": "API",
                    "uiOrigin": "GEOSEARCH_TESTTAB_HOTEL_TAB",
                    "query": city_name,
                    "types": "geo",
                    "max": 6,
                    "lang": "en",
                },
                headers=_HEADERS,
                timeout=_TIMEOUT,
                verify=False,
            )
        r.raise_for_status()
        data = r.json() or {}
    except Exception as e:
        _logger.warning("TripAdvisor TypeAhead failed for %r: %s", city_name, e)
        return None

    results = data.get("results") or []
    for result in results:
        geo_id = result.get("value") or result.get("document_id")
        if geo_id:
            key = f"g{int(geo_id)}"
            _logger.info("Resolved TripAdvisor key for %r: %s", city_name, key)
            return key

    _logger.warning("No geo result found in TypeAhead for %r", city_name)
    return None
