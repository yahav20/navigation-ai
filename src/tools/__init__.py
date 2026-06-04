"""LangChain tools exposed to the agent."""
from tools.activities import fetch_activities, fetch_attractions, fetch_restaurants, fetch_landmarks, fetch_attraction_details, lookup_wikidata
from tools.flights import fetch_flights, find_connecting_flights, get_flight_filter_options
from tools.hotels import fetch_hotels, get_hotel_filter_options, fetch_hotels_gm, get_hotel_filter_options_gm, fetch_hotels_xotelo, fetch_hotels_with_ratings_xotelo
from tools.weather import get_average_weather, get_best_time_to_visit
from tools.maps import geocode_location
from tools.calculator import calculate_trip_cost
from tools.destinations import get_currency_exchange, get_travel_safety_info, get_visa_requirements, get_local_customs, get_wikipedia_summary
from tools.packing import get_packing_list

core_tools = [fetch_flights, fetch_hotels, fetch_activities, get_best_time_to_visit, get_average_weather, calculate_trip_cost]

enrichment_tools = [get_hotel_filter_options, get_flight_filter_options]
enrichment_tool_map = {t.name: t for t in enrichment_tools}

api_tools = [
    fetch_hotels_gm, get_hotel_filter_options_gm,
    fetch_attractions, fetch_restaurants, fetch_landmarks,
    fetch_hotels_xotelo, fetch_hotels_with_ratings_xotelo,
    geocode_location,
    fetch_attraction_details, lookup_wikidata,
]

general_chat_tools = [
    get_currency_exchange,
    get_travel_safety_info,
    get_visa_requirements,
    get_packing_list,
    get_local_customs,
    get_wikipedia_summary,
]
