import os
import json
from .base import BaseDataProvider

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
    
    def fetch_hotels(self, city: str) -> list:
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
