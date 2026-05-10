"""LangChain tools exposed to the agent."""
from tools.activities import fetch_activities
from tools.flights import fetch_flights, get_flight_filter_options
from tools.hotels import fetch_hotels, get_hotel_filter_options
from tools.trip_calculator import calculate_trip_cost
from tools.weather_and_time import get_average_weather, get_best_time_to_visit

core_tools = [fetch_flights, fetch_hotels, fetch_activities, get_best_time_to_visit, get_average_weather, calculate_trip_cost]
enrichment_tools = [get_hotel_filter_options, get_flight_filter_options]

enrichment_tool_map = {t.name: t for t in enrichment_tools}
