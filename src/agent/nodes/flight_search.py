from agent.state import AgentState
from tools.dependencies import data_provider


class FlightSearchNode:
    """
    Dumb Data Loader:
    רק שולף נתונים מה-DB ומרכיב bundle בסיסי.
    בלי פילטרים, בלי preferences, בלי החלטות.
    """

    def __call__(self, state: AgentState) -> dict:
        origin = state.get("current_city")
        destination = state.get("destination_city")

        if not origin or not destination:
            return {
                "flight_options": [],
                "has_flights": False,
                "itinerary_data_bundle": {}
            }

        # -------------------------
        # FLIGHTS (RAW ONLY)
        # -------------------------
        flights = data_provider.fetch_flights(origin, destination) or []
        flights += data_provider.find_connecting_flights(origin, destination) or []
        flights = flights[:5]  # רק הגבלה טכנית, לא לוגיקה

        # -------------------------
        # HOTELS (RAW ONLY)
        # -------------------------
        hotels = data_provider.get_hotels_by_city(destination) or []

        # -------------------------
        # ACTIVITIES (RAW ONLY)
        # -------------------------
        activities = data_provider.get_activities_by_city(destination) or []

        # -------------------------
        # WEATHER / META
        # -------------------------
        weather = data_provider.get_average_weather(destination) or []
        best_time = data_provider.get_best_time_to_visit(destination) or {}

        # -------------------------
        # BUILD BUNDLE (NO FILTERING)
        # -------------------------
        data_bundle = {
            "flights": flights,
            "hotels": hotels,
            "activities": activities,
            "weather": weather,
            "best_time": best_time,
            "budget": state.get("total_budget", 0),
            "trip_days": state.get("trip_days", 3),
            "preferences": state.get("user_preferences", {}),
        }

        return {
            "flight_options": flights,
            "has_flights": bool(flights),
            "itinerary_data_bundle": data_bundle,
        }