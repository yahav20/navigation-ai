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
    def fetch_hotels(self, city: str, max_price: int = None) -> list:
        """
        Fetch available hotels in a city.
        
        Args:
            city: City name
            
        Returns:
            List of available hotels with name, price_per_night, stars, and amenities
        """
        pass


    @abstractmethod
    def fetch_activities(self, city: str) -> list:
        """
        Fetch available activities, museums, tours, and attractions for a specific city.
        Returns a list of activities including their category, price, duration, operational days, and closed dates.
        Crucial for building itineraries or checking if a specific venue is open on a given day.
        Args:
            city: City name
        Returns:
            List of activities with name, category, price, duration, operational_days, and closed_dates
        """
        pass
    
    @abstractmethod
    def get_best_time_to_visit(self, city: str) -> list:
        """
        Fetch the best time to visit a city based on weather patterns, local events, and tourist seasons.
        Returns a list of recommended months or seasons for visiting the city.
        Args:
            city: City name
        Returns:
            List of recommended months or seasons
        """
        pass
    
    @abstractmethod
    def get_average_weather(self, city: str, season: str) -> float:
        """
        Get the average temperature for a specific city during a specific season.
        Valid seasons are: 'Spring', 'Summer', 'Autumn', 'Winter'.
        Args:
            city: City name
            season: Season name ('Spring', 'Summer', 'Autumn', 'Winter')
        Returns:
            Average temperature in Celsius for the specified city and season
        """
        pass

    @abstractmethod
    def get_reachable_destinations_by_distance(
        self, origin: str, destination: str, limit: int = 10
    ) -> list:
        """
        Return destinations that have at least one flight from `origin`,
        ordered by great-circle distance to `destination` (closest first).
        Used to suggest reachable alternatives when no flights to the
        requested destination are available.

        Args:
            origin: Origin city the user is flying from
            destination: City the user originally wanted to fly to
            limit: Maximum number of candidates to return
        Returns:
            List of dicts with city, country, lat, lng, distance_km — or a
            single message dict if the origin or destination cannot be located.
        """
        pass
