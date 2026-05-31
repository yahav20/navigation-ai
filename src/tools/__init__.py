"""LangChain tools exposed to the agent."""
from tools.activities import fetch_activities
from tools.flights import fetch_flights, get_flight_filter_options
from tools.hotels import fetch_hotels, get_hotel_filter_options
from tools.trip_calculator import calculate_trip_cost
from tools.weather_and_time import get_average_weather, get_best_time_to_visit
from tools.google_maps_hotels import fetch_hotels_gm, get_hotel_filter_options_gm
from tools.google_maps_attractions import fetch_attractions, fetch_restaurants, fetch_landmarks
from tools.xotelo_hotels import fetch_hotels_xotelo, fetch_hotels_with_ratings_xotelo
from tools.geocoding import geocode_location
from tools.attraction_enrichment import fetch_attraction_details, lookup_wikidata

core_tools = [fetch_flights, fetch_hotels, fetch_activities, get_best_time_to_visit, get_average_weather, calculate_trip_cost]
enrichment_tools = [get_hotel_filter_options, get_flight_filter_options]

api_tools = [fetch_hotels_gm, get_hotel_filter_options_gm, fetch_attractions, fetch_restaurants, fetch_landmarks, 
             fetch_hotels_xotelo, fetch_hotels_with_ratings_xotelo, geocode_location, 
             fetch_attraction_details, lookup_wikidata]

enrichment_tool_map = {t.name: t for t in enrichment_tools}
