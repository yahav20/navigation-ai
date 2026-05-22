from agent.state import AgentState
from tools.dependencies import data_provider


def _is_direct_flight(item: object) -> bool:
    return isinstance(item, dict) and bool(item.get("flight_number"))


def _is_connecting_route(item: object) -> bool:
    return isinstance(item, dict) and bool(item.get("route"))


def _usable_flights(items: object) -> list[dict]:
    if not isinstance(items, list):
        return []
    return [item for item in items if _is_direct_flight(item) or _is_connecting_route(item)]


class FlightSearchNode:
    """Fetch flights once and persist only route options needed by the graph."""

    def __call__(self, state: AgentState) -> dict:
        """Return flight_options, has_flights, and the itinerary data bundle."""
        origin = state.get("current_city")
        destination = state.get("destination_city")

        if not origin or not destination:
            return {
                "flight_options": [], 
                "has_flights": False,
                "itinerary_data_bundle": {}
            }

        # 1. Fetch Flights (Direct first, fallback to connecting)
         # 1. Fetch Flights (Direct first, fallback to connecting)
        flights = _usable_flights(
        data_provider.fetch_flights(origin, destination))
        if not flights:
            flights = _usable_flights(
            data_provider.find_connecting_flights(origin, destination))

         # 2. Fetch Return Flights 
        return_flights = _usable_flights(
        data_provider.fetch_flights(destination, origin))
        if not return_flights:
            return_flights = _usable_flights(
            data_provider.find_connecting_flights(destination, origin))
        # 2. Fetch all other itinerary data (No early return!)
        hotels = data_provider.get_hotels_by_city(destination) or []
        activities = data_provider.get_activities_by_city(destination) or []
        weather = data_provider.get_average_weather(destination) or []
        best_time = data_provider.get_best_time_to_visit(destination) or {}
        
        # 3. Build the Data Bundle
        data_bundle = {
            "flights": flights,
            "return_flights": return_flights,
            "hotels": hotels,
            "activities": activities,
            "weather": weather,
            "best_time": best_time,
            "budget": state.get("total_budget", 0),
            "trip_days": state.get("trip_days", 3),
            "preferences": state.get("user_preferences", {}),
        }
        
        # הדפסת הלוג עכשיו תעבוד בטוח!
        print(
            f"📦 DATA BUNDLE READY | flights={len(flights)} hotels={len(hotels)} activities={len(activities)} keys={list(data_bundle.keys())}"
        )
        
        return {
            "flight_options": flights,
            "has_flights": bool(flights),
            "itinerary_data_bundle": data_bundle,
        }