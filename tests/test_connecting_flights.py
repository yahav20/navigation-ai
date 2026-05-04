import pytest
from unittest.mock import patch

# 1. Fix import path - matches your project structure
from tools.tools import search_best_flight_route

# --- Tests for Smart Tool (search_best_flight_route) ---

# 2. Fix patch path for all tests!
@patch("tools.tools.data_provider")
def test_direct_flights_within_budget(mock_data_provider):
    """
   senario 1: straightforward case where direct flights are available and within the user's budget.
    """
    mock_data_provider.fetch_flights.return_value = [
        {"flight_number": "LY1", "price": 300, "destination_city": "Paris"},
        {"flight_number": "AF1", "price": 600, "destination_city": "Paris"}
    ]
    
    result = search_best_flight_route.invoke({"origin": "TLV", "destination": "Paris", "max_budget": 400.0})
    
    mock_data_provider.fetch_connecting_flights.assert_not_called()
    assert result["route_type"] == "Direct Flights (Exact or Alternatives)"
    assert len(result["options"]) == 1
    assert result["options"][0]["flight_number"] == "LY1"


@patch("tools.tools.data_provider")
def test_direct_flights_exceed_budget_fallback_to_connecting(mock_data_provider):
    """
   senario 2: direct flights are available but exceed the user's budget, so the tool should automatically search for connecting flights and return those instead.
   
    """
    mock_data_provider.fetch_flights.return_value = [
        {"flight_number": "LY1", "price": 800, "destination_city": "Paris"}
    ]
    mock_data_provider.fetch_connecting_flights.return_value = [
        {"leg1_flight": "LY2", "layover1": "Athens", "total_price": 350, "stops": 1}
    ]
    
    result = search_best_flight_route.invoke({"origin": "TLV", "destination": "Paris", "max_budget": 500.0})
    
    mock_data_provider.fetch_connecting_flights.assert_called_once_with("TLV", "Paris", 500.0)
    assert result["route_type"] == "Connecting Flights (1 or 2 stops)"
    assert "Direct flights exceeded budget" in result["note"]
    assert result["options"][0]["total_price"] == 350


@patch("tools.tools.data_provider")
def test_fallback_to_alternative_cities_direct(mock_data_provider):
    """
    scenario 3: fallback to an alternative city in the same country (direct flights).
    """
    mock_data_provider.fetch_flights.return_value = [{
        "message": "No direct flights to Lyon...",
        "alternatives": [
            {"flight_number": "LY1", "price": 300, "destination_city": "Paris"},
            {"flight_number": "EZY", "price": 200, "destination_city": "Nice"}
        ]
    }]
    
    result = search_best_flight_route.invoke({"origin": "TLV", "destination": "Lyon", "max_budget": 250.0})
    
    assert result["route_type"] == "Direct Flights (Exact or Alternatives)"
    assert len(result["options"]) == 1
    assert result["options"][0]["destination_city"] == "Nice"


@patch("tools.tools.data_provider")
def test_no_flights_found_at_all(mock_data_provider):
    """
    scenario 4: no flights found at all.
    """
    mock_data_provider.fetch_flights.return_value = [{"message": "No flights found"}]
    mock_data_provider.fetch_connecting_flights.return_value = [{"message": "No connecting flights found"}]
    
    result = search_best_flight_route.invoke({"origin": "TLV", "destination": "Atlantis", "max_budget": 1000.0})
    
    assert "message" in result
    assert "No flights (direct or connecting) found from TLV to Atlantis" in result["message"]
@patch("tools.tools.data_provider")
def test_no_budget_provided_the_blank_check(mock_data_provider):
    """
   _scenario 1: the user did not provide a budget (max_budget=None).
    the tool should return all flights available in the database, even if they are very expensive.
    """
    mock_data_provider.fetch_flights.return_value = [
        {"flight_number": "VIP1", "price": 10000, "destination_city": "Dubai"},
        {"flight_number": "REG1", "price": 500, "destination_city": "Dubai"}
    ]
    
    # Run without max_budget key (or with None)
    result = search_best_flight_route.invoke({"origin": "TLV", "destination": "Dubai"})
    
    assert result["route_type"] == "Direct Flights (Exact or Alternatives)"
    assert len(result["options"]) == 2 # both passed because there is no budget limit
    # Ensure sorting still works (cheapest first)
    assert result["options"][0]["price"] == 500
    assert result["options"][1]["price"] == 10000


@patch("tools.tools.data_provider")
def test_negative_budget_insane_input(mock_data_provider):
    """
    scenario 2: the LLM got confused and entered a negative budget or 0.
    the system should not crash, but simply return that no flights match the provided budget.
    """
    mock_data_provider.fetch_flights.return_value = [
        {"flight_number": "LY1", "price": 100, "destination_city": "Eilat"}
    ]
    mock_data_provider.fetch_connecting_flights.return_value = [
        {"message": "No connecting flights found"} # the DB should find nothing for a negative budget
    ]
    
    # Run with a negative budget
    result = search_best_flight_route.invoke({"origin": "TLV", "destination": "Eilat", "max_budget": -50.0})
    
    assert "message" in result
    assert "exceed" in result["message"] or "No flights" in result["message"]


@patch("tools.tools.data_provider")
def test_missing_origin_or_destination_empty_strings(mock_data_provider):
    """
    scenario 3: missing origin or destination city (sent as empty strings).
    in this case, the SQL query will return an empty result, and we want to ensure the tool handles this gracefully
    and returns a well-formatted error message instead of crashing (Exception).
    """
    mock_data_provider.fetch_flights.return_value = [{"message": "No flights found"}]
    mock_data_provider.fetch_connecting_flights.return_value = [{"message": "No connecting flights found"}]
    
    # Send an empty origin city
    result = search_best_flight_route.invoke({"origin": "", "destination": "Paris", "max_budget": 500.0})
    
    assert "message" in result
    assert "No flights" in result["message"]


@patch("tools.tools.data_provider")
def test_exact_budget_match_penny_pincher(mock_data_provider):
    """
    scenario 4: the flight costs *exactly* the same as the budget.
    due to our use of `<=` in the code, the flight must be approved and included in the results.
    """
    mock_data_provider.fetch_flights.return_value = [
        {"flight_number": "LY1", "price": 400.0, "destination_city": "Rome"}, # price exactly equals the budget
        {"flight_number": "LY2", "price": 400.01, "destination_city": "Rome"} # exceeds by one cent!
    ]
    
    result = search_best_flight_route.invoke({"origin": "TLV", "destination": "Rome", "max_budget": 400.0})
    
    assert result["route_type"] == "Direct Flights (Exact or Alternatives)"
    assert len(result["options"]) == 1 # only the flight costing exactly 400 should pass
    assert result["options"][0]["price"] == 400.0