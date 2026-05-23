"""
itinerary_tools.py
==================
LangChain tools for the ItineraryExecutorNode.
Each tool maps to one DB concept. The Executor picks the right tool per step.

USAGE: connect to your DB by implementing the stub functions at the bottom.
All tools return JSON-serialisable dicts or raise ToolException on failure.
"""
from __future__ import annotations

from typing import Optional
from langchain_core.tools import tool, ToolException


# ---------------------------------------------------------------------------
# ── DB stubs ── replace these with your real data_provider calls ────────────
# ---------------------------------------------------------------------------

def _fetch_flights(origin: str, destination: str) -> list[dict]:
    """Return list of flight dicts for this route."""
    raise NotImplementedError("Connect to your data_provider here")


def _fetch_hotels(city: str) -> list[dict]:
    raise NotImplementedError


def _fetch_activities(city: str) -> list[dict]:
    raise NotImplementedError


def _fetch_weather(city: str) -> list[dict]:
    raise NotImplementedError


def _fetch_best_time(city: str) -> dict:
    raise NotImplementedError


def _fetch_return_flights(origin: str, destination: str) -> list[dict]:
    raise NotImplementedError


# ---------------------------------------------------------------------------
# ── Tools ───────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

@tool
def search_outbound_flights(origin: str, destination: str) -> list[dict]:
    """
    Search for available outbound flights from origin to destination.
    Returns a list of flights sorted cheapest first.
    Each flight has: flight_number, airline, price, departure_time, arrival_time,
    duration_minutes, availability.
    """
    try:
        flights = _fetch_flights(origin, destination)
        available = [
            f for f in flights
            if str(f.get("availability", "")).lower() == "available"
        ]
        return sorted(available, key=lambda f: f.get("price", 9999))
    except Exception as e:
        raise ToolException(f"search_outbound_flights failed: {e}")


@tool
def search_return_flights(origin: str, destination: str) -> list[dict]:
    """
    Search for return flights (destination → origin).
    Returns flights sorted cheapest first.
    """
    try:
        flights = _fetch_return_flights(origin, destination)
        available = [
            f for f in flights
            if str(f.get("availability", "")).lower() == "available"
        ]
        return sorted(available, key=lambda f: f.get("price", 9999))
    except Exception as e:
        raise ToolException(f"search_return_flights failed: {e}")


@tool
def search_hotels(
    city: str,
    max_price_per_night: Optional[float] = None,
    min_stars: Optional[int] = None,
    kosher_only: bool = False,
) -> list[dict]:
    """
    Search hotels in a city, optionally filtered by price, stars, and kosher.
    Returns hotels sorted by stars (desc) then price (asc).
    Each hotel has: name, stars, price_per_night, breakfast_available,
    is_kosher, latitude, longitude, hotel_type, amenities.
    """
    try:
        hotels = _fetch_hotels(city)
        if kosher_only:
            hotels = [h for h in hotels if h.get("is_kosher")]
        if min_stars:
            hotels = [h for h in hotels if h.get("stars", 0) >= min_stars]
        if max_price_per_night:
            hotels = [h for h in hotels if h.get("price_per_night", 9999) <= max_price_per_night]
        return sorted(hotels, key=lambda h: (-h.get("stars", 0), h.get("price_per_night", 9999)))
    except Exception as e:
        raise ToolException(f"search_hotels failed: {e}")


@tool
def search_activities(
    city: str,
    categories: Optional[str] = None,
    max_price: Optional[float] = None,
    kosher_only: bool = False,
    vegetarian_friendly: bool = False,
    vegan_friendly: bool = False,
) -> list[dict]:
    """
    Search activities/attractions in a city.
    Returns activities sorted by rating (desc).
    Each activity has: name, categories, price, avg_duration_minutes,
    opening_time, closing_time, operating_days, latitude, longitude,
    food_available, requires_booking, rating, min_age.
    """
    try:
        activities = _fetch_activities(city)

        if categories:
            cat_lower = categories.lower()
            activities = [
                a for a in activities
                if cat_lower in str(a.get("categories", "")).lower()
            ]
        if max_price is not None:
            activities = [a for a in activities if a.get("price", 9999) <= max_price]
        if kosher_only:
            activities = [
                a for a in activities
                if "kosher" in str(a.get("categories", "")).lower()
            ]
        if vegetarian_friendly:
            activities = [
                a for a in activities
                if not a.get("food_available")
                or "vegetarian" in str(a.get("categories", "")).lower()
                or "vegan" in str(a.get("categories", "")).lower()
            ]
        if vegan_friendly:
            activities = [
                a for a in activities
                if not a.get("food_available")
                or "vegan" in str(a.get("categories", "")).lower()
            ]

        return sorted(activities, key=lambda a: -a.get("rating", 0))
    except Exception as e:
        raise ToolException(f"search_activities failed: {e}")


@tool
def get_weather(city: str) -> list[dict]:
    """
    Get average seasonal weather for a city.
    Returns a list of {season, temperature} dicts.
    Use this to understand what to expect during the trip and to
    suggest appropriate clothing or activities.
    """
    try:
        return _fetch_weather(city)
    except Exception as e:
        raise ToolException(f"get_weather failed: {e}")


@tool
def get_best_time_to_visit(city: str) -> dict:
    """
    Get the best months to visit a city and the reason why.
    Returns {months, reason}.
    """
    try:
        return _fetch_best_time(city)
    except Exception as e:
        raise ToolException(f"get_best_time_to_visit failed: {e}")


@tool
def calculate_trip_cost(
    flight_price: float,
    return_flight_price: float,
    hotel_price_per_night: float,
    trip_days: int,
    estimated_activities_budget: float,
    estimated_meals_budget_per_day: float,
) -> dict:
    """
    Calculate total estimated trip cost.
    Returns a breakdown dict with individual components and total.
    """
    hotel_total    = hotel_price_per_night * trip_days
    meals_total    = estimated_meals_budget_per_day * trip_days
    total = (
        flight_price
        + return_flight_price
        + hotel_total
        + estimated_activities_budget
        + meals_total
    )
    return {
        "outbound_flight":  flight_price,
        "return_flight":    return_flight_price,
        "hotel_total":      round(hotel_total, 2),
        "activities_total": round(estimated_activities_budget, 2),
        "meals_total":      round(meals_total, 2),
        "grand_total":      round(total, 2),
    }


# ---------------------------------------------------------------------------
# Tool registry — import this in your graph
# ---------------------------------------------------------------------------

itinerary_tools = [
    search_outbound_flights,
    search_return_flights,
    search_hotels,
    search_activities,
    get_weather,
    get_best_time_to_visit,
    calculate_trip_cost,
]
