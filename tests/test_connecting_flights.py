import sqlite3
import pytest
import time

# =====================================================================
# 1. פונקציית המטרה (The Function to Test)
# =====================================================================
def find_flights(conn: sqlite3.Connection, src_id: str, dest_id: str, max_budget: float = None) -> list:
    """
    חיפוש מסלולי טיסה (ישירים וקונקשנים עד 2 עצירות).
    """
    budget_limit = max_budget if max_budget is not None else 9999999.0
    
    sql = """
    WITH RECURSIVE
    route_builder(current_dest_id, path_cities, path_flight_ids, first_departure, last_arrival, total_price, stops) AS (
        -- Base Case: טיסות ישירות מיעד המקור
        SELECT 
            destination_city_id,
            ',' || origin_city_id || ',' || destination_city_id || ',',
            CAST(id AS TEXT),
            departure_time,
            arrival_time,
            price,
            0
        FROM flights
        WHERE origin_city_id = ? AND price <= ?

        UNION ALL

        -- Recursive Step: טיסות קונקשן
        SELECT 
            f.destination_city_id,
            rb.path_cities || f.destination_city_id || ',',
            rb.path_flight_ids || ' -> ' || CAST(f.id AS TEXT),
            rb.first_departure,
            f.arrival_time,
            rb.total_price + f.price,
            rb.stops + 1
        FROM route_builder rb
        JOIN flights f ON rb.current_dest_id = f.origin_city_id
        WHERE 
            rb.stops < 2
            AND f.departure_time >= datetime(rb.last_arrival, '+1 hour')
            AND rb.path_cities NOT LIKE '%,' || f.destination_city_id || ',%'
            AND (rb.total_price + f.price) <= ?
    )
    SELECT 
        path_flight_ids AS flight_sequence,
        stops,
        total_price,
        first_departure,
        last_arrival,
        ROUND((strftime('%s', last_arrival) - strftime('%s', first_departure)) / 3600.0, 2) AS total_duration_hours
    FROM route_builder
    WHERE current_dest_id = ?
    ORDER BY total_price ASC, stops ASC, total_duration_hours ASC;
    """
    
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    # שימוש ב-Parameterized queries בלבד למניעת SQL Injection
    cur.execute(sql, (src_id, budget_limit, budget_limit, dest_id))
    return [dict(row) for row in cur.fetchall()]


# =====================================================================
# 2. Fixtures (הקמת סביבת הבדיקות והזרקת נתונים)
# =====================================================================

@pytest.fixture
def empty_db():
    """מקים מסד נתונים בזיכרון בלבד (ריק, ללא נתונים)"""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE flights (
            id INTEGER PRIMARY KEY,
            origin_city_id TEXT,
            destination_city_id TEXT,
            price REAL,
            departure_time TEXT,
            arrival_time TEXT
        )
    """)
    yield conn
    conn.close()

@pytest.fixture
def db(empty_db):
    """מזריק נתונים (Seed Data) מגוונים למסד הנתונים"""
    conn = empty_db
    flights_data = [
        # --- טיסות תקינות ---
        (1, 'TLV', 'ATH', 100, '2026-06-01 10:00:00', '2026-06-01 12:00:00'), # ישירה וזולה
        (2, 'TLV', 'JFK', 800, '2026-06-01 05:00:00', '2026-06-01 16:00:00'), # ישירה יקרה
        (3, 'ATH', 'JFK', 500, '2026-06-01 14:00:00', '2026-06-01 20:00:00'), # קונקשן תקין (TLV->ATH->JFK)
        
        # --- זמנים לא תקינים (לוגיקה) ---
        (4, 'ATH', 'CDG', 150, '2026-06-01 11:30:00', '2026-06-01 14:30:00'), # חפיפת זמנים (יוצא לפני שנחת מ-TLV)
        (5, 'TLV', 'LHR', 200, '2026-06-01 08:00:00', '2026-06-01 13:00:00'),
        (6, 'LHR', 'JFK', 400, '2026-06-01 13:30:00', '2026-06-01 18:00:00'), # זמן המתנה קצר משעה (חצי שעה בלבד)
        
        # --- 2 עצירות (תקין) ---
        (7, 'TLV', 'FRA', 150, '2026-06-01 06:00:00', '2026-06-01 10:00:00'),
        (8, 'FRA', 'MAD', 100, '2026-06-01 12:00:00', '2026-06-01 15:00:00'),
        (9, 'MAD', 'JFK', 300, '2026-06-01 17:00:00', '2026-06-01 22:00:00'), # TLV->FRA->MAD->JFK ($550)
        
        # --- לולאות מרושעות ---
        (10, 'JFK', 'TLV', 300, '2026-06-02 10:00:00', '2026-06-02 22:00:00'), # יוצר לולאה פוטנציאלית TLV->JFK->TLV
        
        # --- יותר מדי עצירות (3 עצירות) ---
        (11, 'MAD', 'CDG', 100, '2026-06-01 16:30:00', '2026-06-01 18:30:00'),
        (12, 'CDG', 'JFK', 200, '2026-06-01 20:00:00', '2026-06-02 02:00:00'), # מסלול של 3 עצירות דרך FRA, MAD, CDG
    ]
    
    conn.executemany("""
        INSERT INTO flights (id, origin_city_id, destination_city_id, price, departure_time, arrival_time)
        VALUES (?, ?, ?, ?, ?, ?)
    """, flights_data)
    conn.commit()
    return conn


# =====================================================================
# 3. Tests (מקרי הבדיקה)
# =====================================================================

# 🟢 בדיקות בסיסיות וסינון זמנים
def test_direct_flight_exists(db):
    res = find_flights(db, 'TLV', 'ATH')
    assert len(res) == 1, f"Expected exactly 1 route. Got: {res}"
    assert res[0]['flight_sequence'] == '1', "Expected direct flight ID 1"

def test_no_routes_returns_empty(db):
    res = find_flights(db, 'ATH', 'FRA')
    assert len(res) == 0, f"Expected no routes. Got: {res}"

def test_time_overlap_rejected(db):
    # טיסה ATH->CDG (ID 4) ממריאה ב-11:30.
    # טיסה TLV->ATH (ID 1) נוחתת ב-12:00.
    # המערכת חייבת לפסול את המסלול TLV -> ATH -> CDG.
    res = find_flights(db, 'TLV', 'CDG')
    
    # שולפים את כל רצפי הטיסות שחזרו
    sequences = [r['flight_sequence'] for r in res]
    
    # מוודאים שהמסלול החופף (1 -> 4) לא קיים בתוצאות
    assert '1 -> 4' not in sequences, f"Overlap flight 1->4 was NOT rejected! Got: {res}"
    
    # אפשר גם לוודא שהמסלול התקין (7 -> 8 -> 11) כן חזר!
    assert '7 -> 8 -> 11' in sequences, "Expected the valid alternative route to CDG to be found."

def test_short_layover_rejected(db):
    # טיסה LHR->JFK ממריאה חצי שעה אחרי הנחיתה מ-TLV. חייב להיפסל.
    # המסלולים שכן צריכים לחזור ל-JFK הם:
    # 1. הישיר (800$)
    # 2. ה-1 עצירה דרך ATH (600$)
    # 3. ה-2 עצירות דרך FRA ו-MAD (550$)
    res = find_flights(db, 'TLV', 'JFK')
    sequences = [r['flight_sequence'] for r in res]
    assert '5 -> 6' not in sequences, f"Short layover (< 1 hr) was not rejected! Data: {res}"

def test_loop_rejected(db):
    # מנסים למצוא טיסה ל-TLV. המערכת צריכה למנוע מסלול של TLV->JFK->TLV.
    res = find_flights(db, 'TLV', 'TLV')
    assert len(res) == 0, f"Infinite loop detected! Got: {res}"

def test_max_two_stops_enforced(db):
    # יש לנו במסד אפשרות ל-3 עצירות: TLV -> FRA -> MAD -> CDG -> JFK
    # אבל המערכת מוגבלת לעד 2 עצירות, אז זה לא יחזור.
    res = find_flights(db, 'TLV', 'JFK')
    for route in res:
        assert route['stops'] <= 2, f"Route found with more than 2 stops! Data: {route}"

# 🔵 בדיקות תקציב מתקדמות (Parameterized)
@pytest.mark.parametrize("budget, expected_routes, expected_cheapest", [
    (None, 3, 550),  # ללא הגבלה: כל ה-3 חוזרים. הזול הוא 550$
    (700, 2, 550),   # הטיסה הישירה (800$) נפסלת
    (580, 1, 550),   # גם טיסת ה-1 עצירה (600$) נפסלת
    (400, 0, None),  # שום מסלול לא עומד בתקציב
])
def test_budget_filtering(db, budget, expected_routes, expected_cheapest):
    res = find_flights(db, 'TLV', 'JFK', max_budget=budget)
    assert len(res) == expected_routes, f"Expected {expected_routes} routes for budget {budget}. Got: {res}"
    if expected_routes > 0:
        assert res[0]['total_price'] == expected_cheapest, f"Cheapest route price mismatch. Got: {res[0]['total_price']}"

# 🔵 בדיקות מיון וסדר תוצאות (Sorting)
def test_sorting_by_price_and_stops(db):
    res = find_flights(db, 'TLV', 'JFK')
    assert len(res) >= 2, "Need at least 2 routes to test sorting"
    
    # נוודא שהמחיר של כל תוצאה גדול או שווה לזה שלפניה (מוין כראוי)
    for i in range(len(res) - 1):
        assert res[i]['total_price'] <= res[i+1]['total_price'], \
            f"Sorting error! Item at idx {i} is more expensive than {i+1}. Data: {res}"

# 🔴 קצה ואבטחה
def test_sql_injection_safe(db):
    # במידה ונעשה שימוש מסוכן בשרשור מחרוזות, פקודה כזו עלולה להחזיר את כל ה-DB.
    # מכיוון שהשתמשנו ב-?, השאילתה תחפש עיר בשם "1 OR 1=1" ותחזיר רשימה ריקה.
    res = find_flights(db, "1 OR 1=1", "JFK")
    assert len(res) == 0, f"SQL Injection vulnerability detected! Got: {res}"

def test_empty_db_does_not_crash(empty_db):
    # מוודא שהפונקציה מתמודדת באלגנטיות עם מסד נתונים ריק
    res = find_flights(empty_db, 'TLV', 'JFK')
    assert res == [], "Expected empty list for empty DB"

def test_non_existent_cities(db):
    res = find_flights(db, 'MARS', 'MOON')
    assert res == [], "Expected empty list for fake cities"

def test_no_duplicates_in_paths(db):
    res = find_flights(db, 'TLV', 'JFK')
    sequences = [r['flight_sequence'] for r in res]
    # המרת הרשימה ל-Set מסירה כפילויות. אם האורך השתנה - יש כפילות.
    assert len(sequences) == len(set(sequences)), f"Duplicate routes found! Data: {res}"

# ⭐️ בונוס: בדיקת ביצועים
def test_performance(db):
    start_time = time.time()
    res = find_flights(db, 'TLV', 'JFK')
    end_time = time.time()
    
    runtime = end_time - start_time
    assert runtime < 0.1, f"Query took too long! Runtime: {runtime}s"