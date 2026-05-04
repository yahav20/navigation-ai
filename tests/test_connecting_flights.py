import pytest
from unittest.mock import patch

# 1. תיקון נתיב הייבוא - מתאים למבנה הפרויקט שלך
from tools.tools import search_best_flight_route

# --- טסטים ל-Smart Tool (search_best_flight_route) ---

# 2. תיקון נתיב ה-Patch לכל הטסטים!
@patch("tools.tools.data_provider")
def test_direct_flights_within_budget(mock_data_provider):
    """
    תרחיש 1: יש טיסות ישירות שעומדות בתקציב.
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
    תרחיש 2: טיסות ישירות יקרות מדי, המערכת מפעילה קונקשיינים אוטומטית.
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
    תרחיש 3: פולבק לעיר אחרת באותה מדינה (טיסות ישירות).
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
    תרחיש 4: כלום לא נמצא.
    """
    mock_data_provider.fetch_flights.return_value = [{"message": "No flights found"}]
    mock_data_provider.fetch_connecting_flights.return_value = [{"message": "No connecting flights found"}]
    
    result = search_best_flight_route.invoke({"origin": "TLV", "destination": "Atlantis", "max_budget": 1000.0})
    
    assert "message" in result
    assert "No flights (direct or connecting) found from TLV to Atlantis" in result["message"]
@patch("tools.tools.data_provider")
def test_no_budget_provided_the_blank_check(mock_data_provider):
    """
    מקרה קצה 1: המשתמש לא הכניס תקציב (max_budget=None).
    הכלי אמור להביא לו את כל הטיסות שיש למסד הנתונים להציע, גם אם הן עולות מיליון דולר.
    """
    mock_data_provider.fetch_flights.return_value = [
        {"flight_number": "VIP1", "price": 10000, "destination_city": "Dubai"},
        {"flight_number": "REG1", "price": 500, "destination_city": "Dubai"}
    ]
    
    # הפעלה ללא מפתח max_budget (או עם None)
    result = search_best_flight_route.invoke({"origin": "TLV", "destination": "Dubai"})
    
    assert result["route_type"] == "Direct Flights (Exact or Alternatives)"
    assert len(result["options"]) == 2 # שתיהן עברו כי אין מגבלת תקציב
    # נוודא שהמיון עדיין עובד (הזול קודם)
    assert result["options"][0]["price"] == 500
    assert result["options"][1]["price"] == 10000


@patch("tools.tools.data_provider")
def test_negative_budget_insane_input(mock_data_provider):
    """
    מקרה קצה 2: ה-LLM השתגע והכניס תקציב שלילי או 0.
    המערכת לא אמורה לקרוס, אלא פשוט להחזיר שאין טיסות שעומדות בתקציב ההזוי הזה.
    """
    mock_data_provider.fetch_flights.return_value = [
        {"flight_number": "LY1", "price": 100, "destination_city": "Eilat"}
    ]
    mock_data_provider.fetch_connecting_flights.return_value = [
        {"message": "No connecting flights found"} # ה-DB לא ימצא כלום לתקציב שלילי
    ]
    
    # הפעלה עם תקציב שלילי
    result = search_best_flight_route.invoke({"origin": "TLV", "destination": "Eilat", "max_budget": -50.0})
    
    assert "message" in result
    assert "exceed" in result["message"] or "No flights" in result["message"]


@patch("tools.tools.data_provider")
def test_missing_origin_or_destination_empty_strings(mock_data_provider):
    """
    מקרה קצה 3: חסרה עיר מקור או עיר יעד (נשלחו כמחרוזות ריקות).
    במצב כזה שאילתת ה-SQL תחזיר ריק, ואנחנו רוצים לוודא שהכלי מטפל בזה באלגנטיות 
    ומחזיר הודעת שגיאה מסודרת ולא קורס (Exception).
    """
    mock_data_provider.fetch_flights.return_value = [{"message": "No flights found"}]
    mock_data_provider.fetch_connecting_flights.return_value = [{"message": "No connecting flights found"}]
    
    # נשלח עיר מקור ריקה
    result = search_best_flight_route.invoke({"origin": "", "destination": "Paris", "max_budget": 500.0})
    
    assert "message" in result
    assert "No flights" in result["message"]


@patch("tools.tools.data_provider")
def test_exact_budget_match_penny_pincher(mock_data_provider):
    """
    מקרה קצה 4: הטיסה עולה *בדיוק* כמו התקציב.
    בגלל שהשתמשנו ב- `<=` בקוד שלנו, הטיסה חייבת להיות מאושרת ולהיכלל בתוצאות.
    """
    mock_data_provider.fetch_flights.return_value = [
        {"flight_number": "LY1", "price": 400.0, "destination_city": "Rome"}, # מחיר זהה לתקציב
        {"flight_number": "LY2", "price": 400.01, "destination_city": "Rome"} # חורג בסנט אחד!
    ]
    
    result = search_best_flight_route.invoke({"origin": "TLV", "destination": "Rome", "max_budget": 400.0})
    
    assert result["route_type"] == "Direct Flights (Exact or Alternatives)"
    assert len(result["options"]) == 1 # רק הטיסה שעולה בדיוק 400 אמורה לעבור
    assert result["options"][0]["price"] == 400.0