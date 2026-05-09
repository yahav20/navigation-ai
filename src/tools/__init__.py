from .flights import fetch_flights, get_flight_filter_options
from .hotels import fetch_hotels, get_hotel_filter_options
from .weather_and_time import get_best_time_to_visit, get_average_weather
from .activities import fetch_activities
from .trip_calculator import calculate_trip_cost

core_tools = [fetch_flights, fetch_hotels, fetch_activities, get_best_time_to_visit, get_average_weather, calculate_trip_cost]
enrichment_tools = [get_hotel_filter_options, get_flight_filter_options]

enrichment_tool_map = {t.name: t for t in enrichment_tools}
