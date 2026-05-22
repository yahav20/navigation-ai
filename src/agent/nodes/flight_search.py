from typing import Any
import logging
import json

from agent.state import AgentState
from tools.dependencies import data_provider

logger = logging.getLogger(__name__)


def _is_direct_flight(item: object) -> bool:
    return isinstance(item, dict) and bool(item.get("flight_number"))


def _is_connecting_route(item: object) -> bool:
    return isinstance(item, dict) and bool(item.get("route"))


def _usable_flights(items: object) -> list[dict]:
    if not isinstance(items, list):
        return []
    return [
        item for item in items
        if _is_direct_flight(item) or _is_connecting_route(item)
    ]


# ------------------------------------------------------------------
# NODE
# ------------------------------------------------------------------
class FlightSearchNode:
    """
    NOW: Full Data Preparation Node (NOT only flights)
    Builds the complete itinerary_data_bundle.
    """

    def __call__(self, state: AgentState) -> dict:
        origin = state.get("current_city")
        destination = state.get("destination_city")
        budget = state.get("total_budget", 0)
        trip_days = state.get("trip_days", 3)
        prefs = state.get("user_preferences", {})

        logger.info("📦 DATA NODE START | %s → %s", origin, destination)

        if not origin or not destination:
            logger.error("Missing origin/destination")
            return {
                "flight_options": [],
                "has_flights": False,
                "itinerary_data_bundle": {}
            }

        # -------------------------------------------------
        # 1. FLIGHTS
        # -------------------------------------------------
        raw_flights = _usable_flights(
            data_provider.fetch_flights(origin, destination)
        )

        if not raw_flights:
            raw_flights = _usable_flights(
                data_provider.find_connecting_flights(origin, destination)
            )

        flights = sorted(
            [
                f for f in raw_flights
                if str(f.get("availability", "")).lower() == "available"
            ],
            key=lambda f: f.get("price", 9999)
        )[:3]

        logger.info("✈️ flights=%d", len(flights))

        # -------------------------------------------------
        # 2. HOTELS
        # -------------------------------------------------
        raw_hotels = data_provider.get_hotels_by_city(destination) or []

        hotels = [
            h for h in raw_hotels
            if self._matches_preferences(h, prefs)
        ]

        hotels = sorted(
            hotels,
            key=lambda h: (-h.get("stars", 0), h.get("price_per_night", 9999))
        )[:5]

        logger.info("🏨 hotels=%d", len(hotels))

        # -------------------------------------------------
        # 3. ACTIVITIES
        # -------------------------------------------------
        raw_activities = data_provider.get_activities_by_city(destination) or []

        activities = [
            a for a in raw_activities
            if self._matches_preferences(a, prefs)
        ]

        activities = sorted(
            activities,
            key=lambda a: -a.get("rating", 0)
        )[:20]

        logger.info("🎡 activities=%d", len(activities))

        # -------------------------------------------------
        # 4. WEATHER + BEST TIME
        # -------------------------------------------------
        weather = data_provider.get_average_weather(destination) or []
        best_time = data_provider.get_best_time_to_visit(destination) or {}

        logger.info("🌤 weather=%s | best_time=%s",
                    bool(weather), bool(best_time))

        # -------------------------------------------------
        # 5. BUNDLE (THIS IS WHAT PLANNER NEEDS)
        # -------------------------------------------------
        data_bundle = {
            "flights": flights,
            "hotels": hotels,
            "activities": activities,
            "weather": weather,
            "best_time": best_time,
            "budget": budget,
            "trip_days": trip_days,
            "preferences": prefs,
        }

        logger.info("📦 BUNDLE READY | keys=%s", list(data_bundle.keys()))

        return {
            "flight_options": flights,
            "has_flights": bool(flights),
            "itinerary_data_bundle": data_bundle,
        }

    # -------------------------------------------------
    # Preferences filter (move inside class!)
    # -------------------------------------------------
    def _matches_preferences(self, item: dict, prefs: dict) -> bool:
        if not prefs:
            return True

        if prefs.get("kosher"):
            if not item.get("is_kosher"):
                return False

        if prefs.get("wheelchair") or prefs.get("accessibility"):
            amenities = item.get("amenities", [])
            features = item.get("features", [])
            text = " ".join(map(str, amenities + features)).lower()
            if "wheelchair" not in text and "accessible" not in text:
                return False

        if prefs.get("vegan") or prefs.get("vegetarian"):
            cats = str(item.get("categories", "")).lower()
            if "vegan" not in cats and "vegetarian" not in cats:
                return False

        if "min_age" in prefs:
            if item.get("min_age", 0) > prefs["min_age"]:
                return False

        return True