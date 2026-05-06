import os
import json
from .base import BaseDataProvider
from .sqlite_modules.json_result_counter import (
    get_hotel_dimensions as _hotel_dims,
    get_flight_dimensions as _flight_dims,
    get_cities_in_country as _country_cities,
)

TRAVEL_DB = './data/travel_db.json'


class JSONDataProvider(BaseDataProvider):
    """
    JSON-based data provider for travel data (flights and hotels).
    Loads travel data from a JSON file.
    """
    
    def __init__(self, db_path: str = TRAVEL_DB):
        """
        Initialize the provider with a JSON database file.
        
        Args:
            db_path: Path to the JSON database file
        """
        self.db_path = db_path
        self.data = self._load_data()
    
    def _load_data(self) -> dict:
        """Load the travel database from a JSON file."""
        if not os.path.exists(self.db_path):
            return {}
        with open(self.db_path, 'r') as file:
            return json.load(file)
    
    def fetch_flights(self, origin: str, destination: str) -> list:
        """
        Fetch available flights between two cities.
        
        Args:
            origin: Origin city name
            destination: Destination city name
            
        Returns:
            List of available flights with flight_number, price, availability, and airline
        """
        if not self.data or "flights" not in self.data:
            return [{"message": "No available flights."}]
        
        flights = []
        for flight in self.data.get('flights', []):
            if flight['origin'].lower() == origin.lower() and flight['destination'].lower() == destination.lower():
                flights.append({
                    'flight_number': flight['flight_number'],
                    'price': flight['price'],
                    'availability': flight['availability'],
                    'airline': flight['airline']
                })
        
        if not flights:
            return [{"message": f"No available flights from {origin} to {destination}."}]
        
        return flights
    
    def fetch_hotels(self, city: str, max_price: int = None) -> list:
        """
        Fetch available hotels in a city.
        
        Args:
            city: City name
            
        Returns:
            List of available hotels with name, price_per_night, stars, and amenities
        """
        if not self.data or "hotels" not in self.data:
            return [{"message": "No available hotels."}]
        
        hotels = []
        for hotel in self.data.get('hotels', []):
            if hotel['city'].lower() == city.lower():
                hotels.append({
                    'name': hotel['name'],
                    'price_per_night': hotel['price_per_night'],
                    'stars': hotel['stars'],
                    'amenities': hotel['amenities'],
                })
        
        if not hotels:
            return [{"message": f"No available hotels in {city}."}]
        
        return hotels

    def fetch_activities(self, city: str) -> list:
        """
        Fetch available activities in a city.
        """
        if not self.data or "activities" not in self.data:
            return [{"message": "No available activities in the database."}]
        
        activities = []
        for activity in self.data.get('activities', []):
            if activity['city'].lower() == city.lower():
                activities.append(activity)
                
        if not activities:
            return [{"message": f"No available activities found in {city}."}]
            
        return activities

    def get_best_time_to_visit(self, city: str) -> dict:
        """
        Find the recommended months to visit a specific city.
        """
        best_time_data = self.data.get("best_time_to_visit", {})
        
        for key, value in best_time_data.items():
            if key.lower() == city.lower():
                return value
                
        return {"message": f"No recommendations found for {city}."}


    def get_hotel_dimensions(self, city: str) -> dict:
        return _hotel_dims(self, city)

    def get_flight_dimensions(self, origin: str, destination: str) -> dict:
        return _flight_dims(self, origin, destination)

    def get_cities_in_country(self, country_name: str, origin: str = None) -> list:
        return _country_cities(self, country_name, origin)

    def get_average_weather(self, city: str, season: str) -> dict:
        """
        Get the average temperature for a specific city during a specific season.
        """
        weather_data = self.data.get("average_weather", {})
        
        for key, city_weather in weather_data.items():
            if key.lower() == city.lower():
                for s_key, temp in city_weather.items():
                    if s_key.lower() == season.lower():
                        return {"season": s_key, "temperature": temp}
                return {"message": f"No weather data found for season '{season}' in {city}."}
                
        return {"message": f"No average weather data found for {city}."}
