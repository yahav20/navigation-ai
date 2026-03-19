from langchain_core.tools import tool
import json
import os

TRAVEL_DB = 'travel_db.json'

def load_travel_db():
    """
    Load the travel database from a JSON file.
    """
    if not os.path.exists(TRAVEL_DB):
        return None
    with open(TRAVEL_DB, 'r') as file:
        return json.load(file)
    
@tool
def fetch_flights(origin: str, destination: str):
    """
    Fetch available flights between two cities from the local database.
    Returns a list of flights with prices and availability status.
    """
    # Load your travel_db.json here and implement search logic
    travel_db = load_travel_db()
    
    if not travel_db or "flights" not in travel_db:
        return [{ "message": f"No available flights." }]
    
    flights = []
    for flight in travel_db.get('flights', []):
        if flight['origin'].lower() == origin.lower() and flight['destination'].lower() == destination.lower():
            flights.append({
                'flight_number': flight['flight_number'],
                'price': flight['price'],
                'availability': flight['availability'],
                "airline": flight['airline']
            })
    # If no flights are found, return a message indicating that
    if not flights:
        return [{ "message": f"No available flights from {origin} to {destination}." }]
    
    return flights  

@tool
def fetch_hotels(city: str):
    """
    Fetch available hotels in a city from the local database.
    Returns a list of hotels with prices and availability status.
    """
    # Load your travel_db.json here and implement search logic
    travel_db = load_travel_db()
    
    if not travel_db or "hotels" not in travel_db:
        return [{ "message": f"No available hotels." }]
    
    hotels = []
    for hotel in travel_db.get('hotels', []):
        if hotel['city'].lower() == city.lower():
            hotels.append({
                'name': hotel['name'],
                'price_per_night': hotel['price_per_night'],
                "stars": hotel['stars'],
                'amenities': hotel['amenities'],
            })
            
    # If no hotels are found, return a message indicating that
    if not hotels:
        return [{ "message": f"No available hotels in {city}." }]
    return hotels

@tool
def calculate_trip_cost(flight_price: float, hotel_price_per_night: float, nights: int):
    """
    Calculate the total cost of a trip based on flight price and hotel price per night.
    """
    total_cost = flight_price + (hotel_price_per_night * nights)
    # Add 10% service fee
    total_cost = total_cost * 1.10
    
    return f"The total trip cost, including a 10% service fee, is ${total_cost:.2f}."

if __name__ == "__main__":
    # Example usage
    
    # origin_city = "New York"
    # destination_city = "Tel Aviv"
    # available_flights = fetch_flights.invoke({"origin": origin_city, "destination": destination_city})
    # # print(f"Available flights from {origin_city} to {destination_city}:")
    # for flight in available_flights:
    #     print(flight)
    
    # city = "Tel Aviv"
    # available_hotels = fetch_hotels.invoke({"city": city})
    # print(f"\nAvailable hotels in {city}:")
    # for hotel in available_hotels:
    #     print(hotel)
    
    
    # flight_price = 500.0
    # hotel_price_per_night = 150.0
    # nights = 5
    # total_cost = calculate_trip_cost.invoke({
    #     "flight_price": flight_price,
    #     "hotel_price_per_night": hotel_price_per_night,
    #     "nights": nights
    # })
    # print(f"\n{total_cost}")