from tools.tools import (
    fetch_flights, 
    fetch_hotels, 
    calculate_trip_cost,
    fetch_activities,
    get_best_time_to_visit,
    get_average_weather
)

# --- Tests for fetch_flights (Travelpayouts-backed) ---
def test_fetch_flights_valid_route():
    # Major real-world route — the Travelpayouts feed should always have offers.
    result = fetch_flights.invoke({"origin": "New York", "destination": "London"})
    assert isinstance(result, list)
    assert len(result) > 0
    sample = result[0]
    # Either a direct flight (flight_number + price) or a connecting route (total_price + route).
    assert ("flight_number" in sample and "price" in sample and "airline" in sample) \
        or ("route" in sample and "total_price" in sample)


def test_fetch_flights_unknown_city():
    # An unresolvable city name has no IATA code, so the tool returns an empty list.
    result = fetch_flights.invoke({"origin": "Atlantis", "destination": "Wakanda"})
    assert isinstance(result, list)
    assert result == []
    
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
    # Assuming travel_db.json has no hotels in Hawaii
    result = fetch_hotels.invoke({"city" : "Hawaii"})
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
    
# --- Tests for fetch_activities ---
def test_fetch_activities_valid_city():
    # Assuming travel_db.json has activities in Paris
    result = fetch_activities.invoke({"city": "Paris"})
    assert isinstance(result, list)
    assert len(result) > 0
    assert "name" in result[0]
    assert "category" in result[0]
    assert "price" in result[0]

def test_fetch_activities_no_activities():
    # Assuming travel_db.json has no activities for Atlantis
    result = fetch_activities.invoke({"city": "Atlantis"})
    assert isinstance(result, list)
    assert len(result) == 1
    assert "message" in result[0]
    assert "No available activities" in result[0]["message"]

# --- Tests for get_best_time_to_visit ---
def test_get_best_time_to_visit_valid_city():
    # Assuming travel_db.json has data for London
    result = get_best_time_to_visit.invoke({"city": "London"})
    assert isinstance(result, dict)
    assert "months" in result
    assert "reason" in result
    assert isinstance(result["months"], list)

def test_get_best_time_to_visit_no_data():
    # Assuming travel_db.json has no recommendations for Atlantis
    result = get_best_time_to_visit.invoke({"city": "Atlantis"})
    assert isinstance(result, dict)
    assert "message" in result
    assert "No advisor found" in result["message"]

# --- Tests for get_average_weather ---
def test_get_average_weather_valid_data():
    # Assuming travel_db.json has Winter weather for Amsterdam
    result = get_average_weather.invoke({"city": "Amsterdam", "season": "Winter"})
    assert isinstance(result, dict)
    assert "season" in result
    assert "temperature" in result
    assert result["season"].lower() == "winter"

def test_get_average_weather_invalid_season():
    # Assuming travel_db.json doesn't have a "Monsoon" season for Paris
    result = get_average_weather.invoke({"city": "Paris", "season": "Monsoon"})
    assert isinstance(result, dict)
    assert "message" in result
    assert "No weather data found for season" in result["message"]

def test_get_average_weather_invalid_city():
    # Assuming travel_db.json has no weather data for Atlantis
    result = get_average_weather.invoke({"city": "Atlantis", "season": "Summer"})
    assert isinstance(result, dict)
    assert "message" in result
    assert "No average weather data found" in result["message"]