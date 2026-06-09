"""
itinerary_tools.py
==================
LangChain tools for the ItineraryExecutorNode.

Tools exposed:
  search_activities       — fetch + filter activities for a city
  get_weather             — seasonal weather for a city
  calculate_trip_cost     — total cost breakdown
  get_average_location_cost — estimated avg flight + hotel for standalone mode
"""
from __future__ import annotations

from typing import Optional
from langchain_core.tools import tool, ToolException

from agent.shared.pricing import activity_group_price, flight_group_price, hotel_group_price
from agent.shared.travelers import compute_default_rooms
from providers.flights import search_flights_with_fallback
from providers.xotelo import xotelo_list_hotels
from security import validate_city, validate_positive_number
from tools.dependencies import data_provider
from tools.hotels import TRIPADVISOR_LOCATION_KEYS


# ---------------------------------------------------------------------------
# Internal DB fetchers
# ---------------------------------------------------------------------------

def _fetch_activities(city: str) -> list[dict]:
    city = validate_city(city)
    return data_provider.fetch_activities_with_features(city)


def _fetch_weather(city: str) -> dict:
    city = validate_city(city)
    return data_provider.get_average_weather(city, "Spring")


# ---------------------------------------------------------------------------
# LangChain tools
# ---------------------------------------------------------------------------

@tool
def search_activities(
    city: str,
    kosher_only: bool = False,
    vegetarian_friendly: bool = False,
    vegan_friendly: bool = False,
    category: Optional[str] = None,
    max_price: Optional[float] = None,
) -> list[dict]:
    """
    Search activities/attractions in a city with optional dietary and category filters.
    Returns activities sorted by price ascending.
    """
    try:
        activities = _fetch_activities(city)

        if category:
            cat_lower = category.lower()
            activities = [
                a for a in activities
                if cat_lower in str(a.get("category", "")).lower()
            ]

        if kosher_only:
            activities = [
                a for a in activities
                if "kosher" in [f.lower() for f in a.get("features", [])]
            ]

        if vegetarian_friendly:
            activities = [
                a for a in activities
                if "vegetarian options" in [f.lower() for f in a.get("features", [])]
            ]

        if vegan_friendly:
            activities = [
                a for a in activities
                if "vegan options" in [f.lower() for f in a.get("features", [])]
            ]

        if max_price is not None:
            activities = [a for a in activities if a.get("price", 9999) <= max_price]

        return sorted(activities, key=lambda a: a.get("price", 0))
    except Exception as e:
        raise ToolException(f"search_activities failed: {e}")


@tool
def get_weather(city: str) -> dict:
    """Get average seasonal weather for a city."""
    try:
        return _fetch_weather(city)
    except Exception as e:
        raise ToolException(f"get_weather failed: {e}")


@tool
def calculate_trip_cost(
    flight_price: float,
    return_flight_price: float,
    hotel_price_per_night: float,
    trip_days: int,
    estimated_activities_budget: float,
    estimated_meals_budget_per_day: float,
    num_adults: int = 1,
    num_children: int = 0,
    num_rooms: int = 1,
) -> dict:
    """
    Calculate total estimated trip cost and verify against the user's budget.
    Returns a per-person breakdown plus a group total scaled by the party size.

    num_adults / num_children drive flight and activity/meal group costs.
    num_rooms drives the hotel group cost (billed per room, not per person).
    """
    try:
        validate_positive_number(flight_price)
        validate_positive_number(return_flight_price)
        validate_positive_number(hotel_price_per_night)
        validate_positive_number(trip_days)

        hotel_total = hotel_price_per_night * trip_days
        meals_total = estimated_meals_budget_per_day * trip_days

        # Per-person solo total (backwards-compatible)
        total = (
            flight_price
            + return_flight_price
            + hotel_total
            + estimated_activities_budget
            + meals_total
        )

        # Group totals via shared pricing formulas
        g_outbound   = flight_group_price(flight_price,        num_adults, num_children)
        g_return     = flight_group_price(return_flight_price, num_adults, num_children)
        g_hotel      = hotel_group_price(hotel_price_per_night, num_rooms, trip_days)
        g_activities = activity_group_price(estimated_activities_budget, num_adults, num_children)
        g_meals      = activity_group_price(meals_total,                  num_adults, num_children)
        group_total  = g_outbound + g_return + g_hotel + g_activities + g_meals

        return {
            # Per-person fields (unchanged for backward-compatibility)
            "outbound_flight":  round(flight_price, 2),
            "return_flight":    round(return_flight_price, 2),
            "hotel_total":      round(hotel_total, 2),
            "activities_total": round(estimated_activities_budget, 2),
            "meals_total":      round(meals_total, 2),
            "grand_total":      round(total, 2),
            # Group fields
            "group_outbound_flight":  round(g_outbound,   2),
            "group_return_flight":    round(g_return,     2),
            "group_hotel_total":      round(g_hotel,      2),
            "group_activities_total": round(g_activities, 2),
            "group_meals_total":      round(g_meals,      2),
            "group_grand_total":      round(group_total,  2),
            "num_adults":   num_adults,
            "num_children": num_children,
            "num_rooms":    num_rooms,
        }
    except Exception as e:
        raise ToolException(f"calculate_trip_cost failed: {e}")


@tool
def get_average_location_cost(destination: str, origin: str, trip_days: int) -> dict:
    """
    Estimate average flight and hotel costs for a destination.
    Used in standalone mode (no real booking data available) to produce
    a meaningful budget estimate in verify_budget.
    Returns avg_flight_price, avg_return_flight_price, avg_hotel_per_night.
    """
    try:
        destination = validate_city(destination)
        origin      = validate_city(origin)

        flights_out    = search_flights_with_fallback(origin, destination) or []
        flights_return = search_flights_with_fallback(destination, origin) or []

        def _avg_price(items: list, key: str, fallback: float) -> float:
            prices = [
                float(f.get(key) or f.get("total_price") or 0)
                for f in items
                if isinstance(f, dict) and (f.get(key) or f.get("total_price"))
            ]
            return round(sum(prices) / len(prices), 2) if prices else fallback

        avg_flight        = _avg_price(flights_out,    "price", 400.0)
        avg_return_flight = _avg_price(flights_return, "price", 400.0)

        # Use Xotelo (external) hotel data; fall back to SQLite for unknown cities
        location_key = TRIPADVISOR_LOCATION_KEYS.get(destination.lower())
        xotelo_hotels = xotelo_list_hotels(location_key) if location_key else []
        if xotelo_hotels:
            avg_hotel = _avg_price(xotelo_hotels, "price_per_night", 120.0)
        else:
            db_hotels = data_provider.fetch_hotels(destination) or []
            avg_hotel = _avg_price(db_hotels, "price_per_night", 120.0)

        return {
            "avg_flight_price":        avg_flight,
            "avg_return_flight_price": avg_return_flight,
            "avg_hotel_per_night":     avg_hotel,
            "note": "estimated averages — no booking confirmed",
        }
    except Exception as e:
        raise ToolException(f"get_average_location_cost failed: {e}")


@tool
def get_min_location_cost(destination: str, origin: str, trip_days: int) -> dict:
    """
    Find the minimum available flight and hotel prices for a destination.
    Used in standalone mode when average prices push the trip over budget,
    to check whether the cheapest bookable options would still fit the user's budget.
    Returns min_flight_price, min_return_flight_price, min_hotel_per_night.
    """
    try:
        destination = validate_city(destination)
        origin      = validate_city(origin)

        flights_out    = search_flights_with_fallback(origin, destination) or []
        flights_return = search_flights_with_fallback(destination, origin) or []

        def _min_price(items: list, key: str, fallback: float) -> float:
            prices = [
                float(f.get(key) or f.get("total_price") or 0)
                for f in items
                if isinstance(f, dict) and (f.get(key) or f.get("total_price"))
            ]
            return round(min(prices), 2) if prices else fallback

        min_flight        = _min_price(flights_out,    "price", 400.0)
        min_return_flight = _min_price(flights_return, "price", 400.0)

        location_key  = TRIPADVISOR_LOCATION_KEYS.get(destination.lower())
        xotelo_hotels = xotelo_list_hotels(location_key) if location_key else []
        if xotelo_hotels:
            min_hotel = _min_price(xotelo_hotels, "price_per_night", 120.0)
        else:
            db_hotels = data_provider.fetch_hotels(destination) or []
            min_hotel = _min_price(db_hotels, "price_per_night", 120.0)

        return {
            "min_flight_price":        min_flight,
            "min_return_flight_price": min_return_flight,
            "min_hotel_per_night":     min_hotel,
            "note": "minimum available prices — actual booking required",
        }
    except Exception as e:
        raise ToolException(f"get_min_location_cost failed: {e}")


# ---------------------------------------------------------------------------
# Tool registry — imported by executor.py
# ---------------------------------------------------------------------------

ITINERARY_TOOLS = [
    search_activities,
    get_weather,
    calculate_trip_cost,
    get_average_location_cost,
    get_min_location_cost,
]
