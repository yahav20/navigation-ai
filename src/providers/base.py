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
    def get_hotel_dimensions(self, city: str) -> dict:
        """
        Return only the distinct filtering dimensions for hotels in a city —
        star ratings available and price range. Used to formulate targeted
        preference questions without fetching full hotel records.
        """
        pass

    @abstractmethod
    def get_flight_dimensions(self, origin: str, destination: str) -> dict:
        """
        Return only the distinct filtering dimensions for flights on a route —
        available airlines and price range. Used to formulate targeted
        preference questions without fetching full flight records.
        """
        pass

    @abstractmethod
    def get_origin_cities_in_country(self, country_name: str, destination: str = None) -> list:
        """
        Check if the given name is a country and return origin cities that have
        outgoing flights from that country. Returns an empty list if the name is
        not a recognized country in the database.
        """
        pass

    @abstractmethod
    def get_cities_in_country(self, country_name: str, origin: str = None) -> list:
        """
        Check if the given name is a country and return destination cities that have
        available flights to that country. Returns an empty list if the name is not
        a recognized country in the database.

        Args:
            country_name: Potential country name to look up
            origin: Optional origin city to restrict results to routable destinations
        Returns:
            List of city name strings, empty if country not found
        """
        pass

    @abstractmethod
    def get_reachable_destinations_by_distance(
        self, origin: str, destination: str, limit: int = 10
    ) -> list:
        """
        Return cities reachable by flight from `origin`, ordered by distance to
        `destination`. Used by the alternative_destination node when no flights
        to the requested destination exist.
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
