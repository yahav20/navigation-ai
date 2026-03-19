from tools.tools import fetch_flights, fetch_hotels, calculate_trip_cost


# --- Tests for fetch_flights ---
def test_fetch_flights_valid_route():
    # Assuming travel_db.json has a flight from New York to Los Angeles
    result = fetch_flights.invoke({"origin": "New York", "destination": "London"})
    assert isinstance(result, list)
    assert len(result) > 0
    assert "flight_number" in result[0]
    assert "price" in result[0]
    assert "availability" in result[0]
    assert "airline" in result[0]
    
def test_fetch_flights_no_flights():
    # Assuming travel_db.json has no flights from New York to Tokyo
    result = fetch_flights.invoke({"origin": "New York", "destination": "Tokyo"})
    assert isinstance(result, list)
    assert len(result) == 1
    assert "message" in result[0]
    assert "No available flights" in result[0]["message"]
    
# --- Tests for fetch_hotels ---
def test_fetch_hotels_valid_city():
    # Assuming travel_db.json has hotels in Paris
    result = fetch_hotels.invoke({"city" : "Paris"})
    assert isinstance(result, list)
    assert len(result) > 0
    assert "name" in result[0]
    assert "price_per_night" in result[0]
    assert "stars" in result[0]
    assert "amenities" in result[0]
    
def test_fetch_hotels_no_hotels():
    # Assuming travel_db.json has no hotels in Tokyo
    result = fetch_hotels.invoke({"city" : "Tokyo"})
    assert isinstance(result, list) 
    assert len(result) == 1
    assert "message" in result[0]
    assert "No available hotels" in result[0]["message"]
    
# --- Tests for calculate_trip_cost ---
def test_calculate_trip_cost():
    flight_price = 500.0
    hotel_price_per_night = 150.0
    nights = 5
    result = calculate_trip_cost.invoke({
        "flight_price": flight_price,
        "hotel_price_per_night": hotel_price_per_night,
        "nights": nights
    })
    expected_total_cost = (flight_price + (hotel_price_per_night * nights)) * 1.10
    assert isinstance(result, str)
    assert f"${expected_total_cost:.2f}" in result  