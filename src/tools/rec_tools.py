"""LangChain tools for the recommendation agent."""
from langchain_core.tools import tool

from providers.sqlite.provider import SQLiteDataProvider
from tools.activities import fetch_activities
from tools.weather_and_time import get_average_weather, get_best_time_to_visit

_provider = SQLiteDataProvider()


@tool
def find_destinations_by_vibe(category: str) -> list[dict]:
    """Find cities that have activities matching a given category.

    Available categories: Culture, Entertainment, Family, History, Nature, Nightlife, Sightseeing.
    Use this to answer vibe-based questions like 'where should I go for nightlife?'
    or 'I want a nature-focused trip, where should I go?'
    """
    return _provider.search_by_activity_category(category)


@tool
def get_reachable_destinations(origin: str, max_flight_hours: float | None = None) -> list[dict]:
    """Get all destinations with available flights from a given origin city.

    Use max_flight_hours to filter by actual flight duration:
    - Short haul (under ~2 hours):  max_flight_hours=2
    - Medium haul (under ~5 hours): max_flight_hours=5
    - Long haul (under ~8 hours):   max_flight_hours=8
    Leave max_flight_hours as None to return all reachable destinations.
    Returns city, country, price range, and shortest available flight duration.
    """
    return _provider.get_all_destinations_from_origin(origin, max_flight_hours)


@tool
def find_destinations_by_tag(tag: str) -> list[dict]:
    """Find cities that match a given travel style or vibe tag.

    Available tags: alternative, art, beach, budget-friendly, canals, city-break,
    cultural, cycling, entertainment, expensive, fashion, foodie, historic, iconic,
    liberal, luxury, mediterranean, modern, multicultural, nature-nearby, nightlife,
    rainy, romantic, safe, shopping, student-friendly, sunny, technology, theatre,
    unique, walkable.

    Use this when the user describes a vibe, atmosphere, or lifestyle preference
    (e.g. 'romantic', 'beach', 'budget-friendly', 'foodie').
    Prefer find_destinations_by_tag over find_destinations_by_vibe for style/atmosphere queries;
    use find_destinations_by_vibe for activity-category queries (Nature, Culture, Family, etc.).
    """
    return _provider.search_by_tag(tag)


@tool
def find_destinations_within_budget(origin: str, total_budget: float, trip_days: int) -> list[dict]:
    """Find destinations affordable from a given origin within a total budget for a given number of days.

    Calculates the minimum possible trip cost as: cheapest available flight + (cheapest hotel per night × days).
    Returns only destinations where this minimum total is within the budget, ordered cheapest first.
    Each result includes the cheapest flight price, cheapest hotel per night, estimated minimum total,
    and shortest available flight duration.

    Use this whenever the user mentions a budget alongside an origin, e.g.:
    'I have $1000 and I'm flying from Tel Aviv for 5 days — where can I go?'
    """
    return _provider.find_within_budget(origin, total_budget, trip_days)


@tool
def get_trip_duration_recommendation(city: str) -> dict:
    """Get a recommendation for how many days to spend in a city.

    Returns the suggested minimum and maximum number of days, with an explanation
    of what each range covers.
    Use this when the user asks 'how many days should I spend in X?'
    or 'is N days enough for X?' or wants help splitting time across multiple cities.
    """
    return _provider.get_recommended_duration(city)


@tool
def get_city_overview(city: str) -> dict:
    """Get a rich profile of a destination city.

    Returns:
    - Activity categories available and sample activity names
    - Best months to visit and the reason why
    - Average temperature for each season (Spring, Summer, Autumn, Winter)

    Use this to answer questions like 'what kind of trip is Tokyo?',
    'when should I visit Paris?', or 'what is there to do in Amsterdam?'
    """
    return _provider.get_city_profile(city)


rec_tools = [
    find_destinations_by_vibe,
    find_destinations_by_tag,
    get_reachable_destinations,
    find_destinations_within_budget,
    get_city_overview,
    get_trip_duration_recommendation,
    fetch_activities,
    get_best_time_to_visit,
    get_average_weather,
]
