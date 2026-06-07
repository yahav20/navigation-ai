"""Hybrid Data Access Layer (DAL) for the navigation-ai project.

This layer coordinates between cached API data, live API calls, and local fixtures.
All public fetch methods return a unified column schema regardless of source.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any

from providers.city_metadata import CITY_METADATA
from providers.google_maps.places_api import get_place_details, search_nearby_places
from providers.sqlite.provider import SQLiteDataProvider
from providers.tripadvisor_lookup import scrape_ta_location_key
from providers.wikipedia.enrichment import fetch_wiki_summary
from providers.xotelo.hotels_api import xotelo_get_rates, xotelo_list_hotels

_logger = logging.getLogger(__name__)

# Rough USD estimate per Google price_level tier (0-4)
_PRICE_LEVEL_TO_USD = {0: 0.0, 1: 15.0, 2: 40.0, 3: 100.0, 4: 250.0}


class HybridDAL:
    """Coordinates data access across multiple sources with caching and fallbacks."""

    def __init__(self, db_path: str | None = None) -> None:
        self.sqlite = SQLiteDataProvider(db_path) if db_path else SQLiteDataProvider()
        self.db_path = self.sqlite.db_path

    def __getattr__(self, name: str) -> Any:
        """Delegate any unknown methods to the underlying SQLite provider."""
        return getattr(self.sqlite, name)

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_hotel(row: dict, source: str = "local") -> dict:
        """Return a unified hotel dict with consistent keys and no unexpected None values."""
        amenities = row.get("amenities") or "[]"
        if isinstance(amenities, str):
            try:
                amenities = json.loads(amenities)
            except (ValueError, TypeError):
                amenities = []

        raw_stars = row.get("stars")
        stars = round(float(raw_stars), 2) if raw_stars is not None else None

        raw_price_min = row.get("price_min") or row.get("price_per_night")
        raw_price_max = row.get("price_max")
        price_per_night = round(float(raw_price_min), 2) if raw_price_min is not None else None
        price_max = round(float(raw_price_max), 2) if raw_price_max is not None else None

        return {
            "name": row.get("name") or "",
            "stars": stars,
            "price_per_night": price_per_night,
            "price_min": price_per_night,
            "price_max": price_max,
            "latitude": row.get("latitude") or row.get("lat"),
            "longitude": row.get("longitude") or row.get("lng"),
            "address": row.get("address") or row.get("formatted_address") or "",
            "amenities": amenities if isinstance(amenities, list) else [],
            "min_age": int(row.get("min_age") or 0),
            "hotel_type": row.get("hotel_type") or "",
            "is_kosher": bool(row.get("is_kosher", False)),
            "distance_from_center_km": row.get("distance_from_center_km"),
            "review_count": row.get("review_count"),
            "phone": row.get("phone") or "",
            "website": row.get("website") or "",
            "image_urls": row.get("image_urls") or "",
            "booking_url": row.get("booking_url") or row.get("tripadvisor_url") or "",
            # Preserve the Xotelo/TripAdvisor key so callers can fetch rates later
            "hotel_key": row.get("hotel_key") or row.get("tripadvisor_key"),
            "source": source,
        }

    @staticmethod
    def _normalize_activity(row: dict, source: str = "local") -> dict:
        """Return a unified activity dict with consistent keys and no unexpected None values.

        Both local activities and API attractions produce the same key set so
        callers never need to know which source the data came from.
        """
        cats = row.get("categories") or row.get("types") or "[]"
        if isinstance(cats, str):
            try:
                cats = json.loads(cats)
            except (ValueError, TypeError):
                cats = []

        raw_rating = row.get("rating")
        rating = round(float(raw_rating), 2) if raw_rating is not None else None

        # Derive a USD price estimate when only a 0-4 price_level is available
        raw_price = row.get("price")
        price_level = row.get("price_level")
        if raw_price is not None:
            price = float(raw_price)
        elif price_level is not None:
            price = _PRICE_LEVEL_TO_USD.get(int(price_level), 0.0)
        else:
            price = 0.0

        return {
            "name": row.get("name") or "",
            "categories": cats if isinstance(cats, list) else [],
            "rating": rating,
            "price": price,
            "price_level": price_level,
            "min_age": int(row.get("min_age") or 0),
            "latitude": row.get("latitude") or row.get("lat"),
            "longitude": row.get("longitude") or row.get("lng"),
            "address": row.get("address") or row.get("formatted_address") or "",
            "avg_duration_minutes": row.get("avg_duration_minutes"),
            "opening_time": row.get("opening_time") or "",
            "closing_time": row.get("closing_time") or "",
            "operating_days": row.get("operating_days") or "",
            "best_time_of_day": row.get("best_time_of_day") or "",
            "food_available": bool(row.get("food_available", False)),
            "requires_booking": bool(row.get("requires_booking", False)),
            "wiki_description": row.get("wiki_description") or row.get("wiki_extract") or "",
            "review_count": row.get("review_count"),
            "phone": row.get("phone") or "",
            "website": row.get("website") or "",
            "hours": row.get("hours") or "",
            "source": source,
        }

    def _enrich_hotel_prices(self, hotels: list[dict]) -> list[dict]:
        """Call xotelo_get_rates for hotels that lack a price (max 5 rate calls).

        Accepts both raw Xotelo dicts (use 'hotel_key') and cached DB rows
        (use 'tripadvisor_key') — checks both field names.
        Tries two date windows: 7 days out then 30 days out.
        """
        from datetime import date, timedelta

        def _windows():
            today = date.today()
            yield (today + timedelta(days=7)).strftime("%Y-%m-%d"),  (today + timedelta(days=10)).strftime("%Y-%m-%d")
            yield (today + timedelta(days=30)).strftime("%Y-%m-%d"), (today + timedelta(days=33)).strftime("%Y-%m-%d")

        enriched = []
        calls = 0
        for hotel in hotels:
            key = hotel.get("hotel_key") or hotel.get("tripadvisor_key")
            if hotel.get("price_per_night") is None and key and calls < 5:
                for check_in, check_out in _windows():
                    try:
                        rates = xotelo_get_rates(key, check_in, check_out)
                        available = [r for r in rates if r.get("availability") == "available"]
                        if available:
                            hotel = dict(hotel)
                            hotel["price_per_night"] = round(
                                min(r["price_per_night"] for r in available), 2
                            )
                            break
                    except Exception as e:
                        _logger.warning("Xotelo rates failed for %s: %s", hotel.get("name"), e)
                calls += 1
            enriched.append(hotel)
        return enriched

    @staticmethod
    def _merge_activity(api_row: dict, local_row: dict) -> dict:
        """Merge API and local records for the same attraction.

        Local provides: price, opening/closing times, operating days, food_available,
        requires_booking, avg_duration_minutes, min_age, categories.
        API provides: rating (live), review_count, address, wiki_description, phone, website.
        """
        merged = dict(local_row)
        if api_row.get("rating") is not None:
            merged["rating"] = api_row["rating"]
        merged["review_count"] = api_row.get("review_count")
        if api_row.get("address"):
            merged["address"] = api_row["address"]
        if api_row.get("wiki_description"):
            merged["wiki_description"] = api_row["wiki_description"]
        if api_row.get("phone"):
            merged["phone"] = api_row["phone"]
        if api_row.get("website"):
            merged["website"] = api_row["website"]
        merged["price_level"] = api_row.get("price_level")
        if api_row.get("latitude") is not None:
            merged["latitude"] = api_row["latitude"]
        if api_row.get("longitude") is not None:
            merged["longitude"] = api_row["longitude"]
        merged["source"] = "hybrid"
        return merged

    @staticmethod
    def _normalize_restaurant(row: dict, source: str = "api") -> dict:
        """Return a unified restaurant dict with consistent keys."""
        raw_rating = row.get("rating")
        rating = round(float(raw_rating), 2) if raw_rating is not None else None

        raw_price = row.get("price")
        price_level = row.get("price_level")
        if raw_price is not None:
            price = float(raw_price)
        elif price_level is not None:
            price = _PRICE_LEVEL_TO_USD.get(int(price_level), 0.0)
        else:
            price = None

        return {
            "name": row.get("name") or "",
            "rating": rating,
            "price": price,
            "price_level": price_level,
            "review_count": row.get("review_count"),
            "latitude": row.get("latitude") or row.get("lat"),
            "longitude": row.get("longitude") or row.get("lng"),
            "address": row.get("address") or row.get("formatted_address") or "",
            "phone": row.get("phone") or "",
            "website": row.get("website") or "",
            "hours": row.get("hours") or "",
            "cuisine": row.get("cuisine") or "",
            "source": source,
        }

    def _enrich_places_with_details(self, places: list[dict], max_enrichments: int = 20) -> list[dict]:
        """Enrich places with phone, website, hours from Google Place Details API.

        Only called during initial cache population so the cost is paid once per city.
        Silently skips on API error or missing key.
        """
        enriched = []
        count = 0
        for place in places:
            place = dict(place)
            pid = place.get("place_id")
            if pid and count < max_enrichments:
                try:
                    details = get_place_details(pid)
                    if details:
                        if details.get("phone"):
                            place["phone"] = details["phone"]
                        if details.get("website"):
                            place["website"] = details["website"]
                        hours = details.get("hours")
                        if hours:
                            place["hours"] = json.dumps(hours) if isinstance(hours, list) else hours
                    count += 1
                except Exception as e:
                    _logger.warning("Place Details failed for %s: %s", place.get("name"), e)
            enriched.append(place)
        return enriched

    # ------------------------------------------------------------------
    # Internal city lookup
    # ------------------------------------------------------------------

    def _get_city_id(self, city_name: str) -> int | None:
        """Resolve city name to DB id, preferring cities that have seeded travel data."""
        name = city_name.lower().strip()
        # Prefer cities that have hotels (our seeded cities) so "Paris" → France, not Canada
        rows = self.sqlite._query(
            """SELECT c.id, COUNT(h.id) AS hotel_count
                 FROM cities c
                 LEFT JOIN hotels h ON h.city_id = c.id
                WHERE LOWER(c.name) = ?
                GROUP BY c.id
                ORDER BY hotel_count DESC, c.id ASC
                LIMIT 1""",
            (name,),
        )
        if rows:
            return rows[0]["id"]
        # Fuzzy prefix match as fallback
        rows = self.sqlite._query(
            """SELECT c.id, COUNT(h.id) AS hotel_count
                 FROM cities c
                 LEFT JOIN hotels h ON h.city_id = c.id
                WHERE LOWER(c.name) LIKE ?
                GROUP BY c.id
                ORDER BY hotel_count DESC, c.id ASC
                LIMIT 1""",
            (name + "%",),
        )
        return rows[0]["id"] if rows else None

    def _get_ta_key(self, city_name: str, city_id: int) -> str | None:
        """Resolve a TripAdvisor gXXXXXX location key for a city.

        Lookup order:
        1. cities.ta_key column in DB (fastest, covers pre-seeded and previously scraped)
        2. Hardcoded CITY_METADATA dict (redundant after first seed, but a safe fallback)
        3. Scrape TripAdvisor search page and persist the result for next time
        """
        # 1. DB column
        rows = self.sqlite._query("SELECT ta_key FROM cities WHERE id = ?", (city_id,))
        if rows and rows[0].get("ta_key"):
            return rows[0]["ta_key"]

        # 2. Hardcoded metadata
        key = city_name.lower().strip()
        meta = CITY_METADATA.get(key)
        if meta and meta.get("ta_key"):
            ta_key = meta["ta_key"]
            self._save_ta_key(city_id, ta_key)
            return ta_key

        # 3. Scrape TripAdvisor
        ta_key = scrape_ta_location_key(city_name)
        if ta_key:
            self._save_ta_key(city_id, ta_key)
        return ta_key

    def _save_ta_key(self, city_id: int, ta_key: str) -> None:
        import sqlite3 as _sqlite3
        conn = _sqlite3.connect(self.db_path)
        try:
            conn.execute("UPDATE cities SET ta_key = ? WHERE id = ?", (ta_key, city_id))
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Public fetch methods
    # ------------------------------------------------------------------

    def fetch_hotels(self, city: str, max_price: int | None = None, approach: str = "hybrid") -> list[dict]:
        """Fetch hotels for a city. Returns unified hotel schema regardless of source.

        Approaches:
            - "local":  Only local fixtures.
            - "hybrid": API hotels + local hotels merged (API listed first, local fills gaps).
            - "api":    Live API only (no local merge).
        """
        if approach == "local":
            raw = self.sqlite.fetch_hotels(city, max_price)
            if raw and isinstance(raw[0], dict) and "message" in raw[0]:
                return raw
            if max_price is not None:
                raw = [r for r in raw if (r.get("price_per_night") or 0) <= max_price]
            return [self._normalize_hotel(r, "local") for r in raw]

        city_id = self._get_city_id(city)
        if not city_id:
            return []

        api_results: list[dict] = []

        # 1. Check cache
        cache = self.sqlite._query("SELECT * FROM api_hotels WHERE city_id = ?", (city_id,))
        if cache:
            raw_cached = [dict(row) for row in cache]
            # Re-enrich any still-priceless cached hotels (at most 5 rate calls)
            if any(r.get("price_per_night") is None for r in raw_cached):
                raw_cached = self._enrich_hotel_prices(raw_cached)
            api_results = [self._normalize_hotel(h, "api") for h in raw_cached]

        # 2. Call Xotelo if cache is empty
        if not api_results:
            ta_key = self._get_ta_key(city, city_id)
            if ta_key:
                try:
                    hotels = xotelo_list_hotels(ta_key)
                    if hotels:
                        hotels = self._enrich_hotel_prices(hotels)
                        self._cache_hotels(city_id, hotels)
                        api_results = [self._normalize_hotel(h, "api") for h in hotels]
                except Exception as e:
                    _logger.error("Xotelo API failed for %s: %s", city, e)

        # For "api" approach: return API results only (no local merge)
        if approach == "api":
            if max_price is not None:
                api_results = [h for h in api_results if h.get("price_per_night") is None
                               or h["price_per_night"] <= max_price]
            return api_results

        # For "hybrid": merge API results with local, API listed first.
        # Local hotels not present in the API set are appended so the agent always
        # has priced options even when Xotelo omits price_ranges.
        local_raw = self.sqlite.fetch_hotels(city, None)
        local_hotels: list[dict] = []
        if not (local_raw and isinstance(local_raw[0], dict) and "message" in local_raw[0]):
            local_hotels = [self._normalize_hotel(r, "local") for r in local_raw]

        api_names = {h["name"].lower() for h in api_results}
        merged = api_results + [h for h in local_hotels if h["name"].lower() not in api_names]

        if max_price is not None:
            # Keep hotels with no price (can't prove unaffordable) or within budget
            merged = [h for h in merged if h.get("price_per_night") is None
                      or h["price_per_night"] <= max_price]

        return merged

    def get_hotel_dimensions(self, city: str) -> dict:
        """Forward to sqlite for now, or could be hybrid later."""
        return self.sqlite.get_hotel_dimensions(city)

    def fetch_activities(self, city: str, approach: str = "hybrid") -> list[dict]:
        """Fetch activities / attractions for a city. Returns unified schema regardless of source.

        Approaches:
            - "local":  Only local fixtures.
            - "hybrid": API attractions + local activities merged (API listed first).
            - "api":    Live API only (no local merge).
        """
        if approach == "local":
            raw = self.sqlite.fetch_activities(city)
            return [self._normalize_activity(r, "local") for r in raw]

        city_id = self._get_city_id(city)
        if not city_id:
            return []

        api_results: list[dict] = []

        # 1. Check cache
        cache = self.sqlite._query("SELECT * FROM api_attractions WHERE city_id = ?", (city_id,))
        if cache:
            api_results = [self._normalize_activity(dict(row), "api") for row in cache]

        # 2. Call Google Maps + Wikipedia if cache is empty
        if not api_results:
            rows = self.sqlite._query("SELECT lat, lng FROM cities WHERE id = ?", (city_id,))
            if rows:
                lat, lng = rows[0]["lat"], rows[0]["lng"]
                try:
                    attractions = search_nearby_places((lat, lng), "tourist_attraction", limit=15)
                    if attractions:
                        for attr in attractions:
                            wiki = fetch_wiki_summary(attr["name"])
                            if wiki:
                                attr["wiki_title"] = wiki.get("title")
                                attr["wiki_description"] = wiki.get("extract")
                                attr["wiki_url"] = wiki.get("url")
                        self._cache_attractions(city_id, attractions)
                        api_results = [self._normalize_activity(a, "api") for a in attractions]
                except Exception as e:
                    _logger.error("Google Attractions API failed for %s: %s", city, e)

        # For "api" approach: return API results only
        if approach == "api":
            return api_results

        # For "hybrid": smart merge — when an attraction exists in both sources,
        # combine API's live rating/wiki/address with local's price/hours/scheduling.
        local_results = [self._normalize_activity(r, "local")
                         for r in self.sqlite.fetch_activities(city)]

        api_by_name   = {a["name"].lower(): a for a in api_results}
        local_by_name = {a["name"].lower(): a for a in local_results}

        result: list[dict] = []
        for name, api_row in api_by_name.items():
            if name in local_by_name:
                result.append(self._merge_activity(api_row, local_by_name[name]))
            else:
                result.append(api_row)
        for name, local_row in local_by_name.items():
            if name not in api_by_name:
                result.append(local_row)
        return result

    def fetch_restaurants(self, city: str, approach: str = "hybrid") -> list[dict]:
        """Fetch restaurants for a city."""
        city_id = self._get_city_id(city)
        if not city_id:
            return []

        # 1. Check cache
        if approach == "hybrid":
            cache = self.sqlite._query(
                "SELECT * FROM api_restaurants WHERE city_id = ?",
                (city_id,),
            )
            if cache:
                return [self._normalize_restaurant(dict(row), "api") for row in cache]

        # 2. Call API
        rows = self.sqlite._query("SELECT lat, lng FROM cities WHERE id = ?", (city_id,))
        if rows:
            lat, lng = rows[0]["lat"], rows[0]["lng"]
            try:
                restaurants = search_nearby_places((lat, lng), "restaurant", limit=10)
                if restaurants:
                    self._cache_restaurants(city_id, restaurants)
                    return [self._normalize_restaurant(r, "api") for r in restaurants]
            except Exception as e:
                _logger.error("Google Restaurants API failed for %s: %s", city, e)

        return []

    # ------------------------------------------------------------------
    # Cache write helpers
    # ------------------------------------------------------------------

    def _cache_hotels(self, city_id: int, hotels: list[dict]) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            for hotel in hotels:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO api_hotels
                    (city_id, api_source, tripadvisor_key, name, stars, review_count,
                     price_per_night, price_min, price_max, currency, latitude, longitude,
                     address, image_urls, booking_url)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        city_id,
                        "xotelo",
                        hotel.get("hotel_key"),
                        hotel["name"],
                        hotel.get("stars"),
                        hotel.get("review_count"),
                        hotel.get("price_per_night") or hotel.get("price_min"),
                        hotel.get("price_min"),
                        hotel.get("price_max"),
                        hotel.get("currency", "USD"),
                        hotel.get("lat"),
                        hotel.get("lng"),
                        hotel.get("address"),
                        hotel.get("image_urls"),
                        hotel.get("booking_url") or hotel.get("tripadvisor_url"),
                    ),
                )
            conn.commit()
        finally:
            conn.close()

    def _cache_attractions(self, city_id: int, attractions: list[dict]) -> None:
        attractions = self._enrich_places_with_details(attractions)
        conn = sqlite3.connect(self.db_path)
        try:
            for attr in attractions:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO api_attractions
                    (city_id, place_id, name, rating, price_level, latitude, longitude,
                     address, types, wiki_title, wiki_description, wiki_url,
                     phone, website, hours)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        city_id,
                        attr.get("place_id"),
                        attr["name"],
                        attr.get("rating"),
                        attr.get("price_level"),
                        attr.get("lat"),
                        attr.get("lng"),
                        attr.get("formatted_address"),
                        json.dumps(attr.get("types", [])),
                        attr.get("wiki_title"),
                        attr.get("wiki_description"),
                        attr.get("wiki_url"),
                        attr.get("phone"),
                        attr.get("website"),
                        attr.get("hours"),
                    ),
                )
            conn.commit()
        finally:
            conn.close()

    def _cache_restaurants(self, city_id: int, restaurants: list[dict]) -> None:
        restaurants = self._enrich_places_with_details(restaurants)
        conn = sqlite3.connect(self.db_path)
        try:
            for rest in restaurants:
                price_level = rest.get("price_level")
                price = (
                    _PRICE_LEVEL_TO_USD.get(int(price_level), 0.0)
                    if price_level is not None else None
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO api_restaurants
                    (city_id, place_id, name, rating, review_count, price, price_level,
                     latitude, longitude, address, phone, website, hours)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        city_id,
                        rest.get("place_id"),
                        rest["name"],
                        rest.get("rating"),
                        rest.get("review_count"),
                        price,
                        price_level,
                        rest.get("lat"),
                        rest.get("lng"),
                        rest.get("formatted_address"),
                        rest.get("phone"),
                        rest.get("website"),
                        rest.get("hours"),
                    ),
                )
            conn.commit()
        finally:
            conn.close()
