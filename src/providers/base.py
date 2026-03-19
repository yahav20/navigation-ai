from abc import ABC, abstractmethod


class BaseDataProvider(ABC):
    """
    Abstract base class for travel data providers.
    Defines the interface that all data providers must implement.
    """
    
    @abstractmethod
    def fetch_flights(self, origin: str, destination: str) -> list:
        """
        Fetch available flights between two cities.
        
        Args:
            origin: Origin city name
            destination: Destination city name
            
        Returns:
            List of available flights with flight_number, price, availability, and airline
        """
        pass
    
    @abstractmethod
    def fetch_hotels(self, city: str) -> list:
        """
        Fetch available hotels in a city.
        
        Args:
            city: City name
            
        Returns:
            List of available hotels with name, price_per_night, stars, and amenities
        """
        pass
