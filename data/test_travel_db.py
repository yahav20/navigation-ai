"""
Standalone integration tests for travel_agency.db (normalized FK schema).
Run from the project root:  python data/test_travel_db.py
No external dependencies required.
"""

import os
import sqlite3
import sys

DB_PATH = os.path.join(os.path.dirname(__file__), "travel_agency.db")

_passed = 0
_failed = 0


def check(description: str, condition: bool):
    global _passed, _failed
    if condition:
        print(f"  PASS  {description}")
        _passed += 1
    else:
        print(f"  FAIL  {description}")
        _failed += 1


def get_conn():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}. Run data/init_db.py first."
        )
    return sqlite3.connect(DB_PATH)


# ── Schema ─────────────────────────────────────────────────────────────────────

def test_schema():
    print("\n[Schema — tables]")
    conn = get_conn()
    cur = conn.cursor()

    for table in ("countries", "cities", "flights", "hotels", "activities",
                  "best_time_to_visit", "average_weather"):
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
        check(f"table '{table}' exists", cur.fetchone() is not None)

    print("\n[Schema — flights columns]")
    cur.execute("PRAGMA table_info(flights)")
    flight_cols = {row[1] for row in cur.fetchall()}
    for col in ("id", "origin_city_id", "destination_city_id",
                "airline", "price", "flight_number", "availability"):
        check(f"flights.{col} exists", col in flight_cols)

    print("\n[Schema — FK tables]")
    cur.execute("PRAGMA table_info(cities)")
    city_cols = {row[1] for row in cur.fetchall()}
    check("cities.country_id FK column exists", "country_id" in city_cols)

    cur.execute("PRAGMA table_info(best_time_to_visit)")
    btv_cols = {row[1] for row in cur.fetchall()}
    check("best_time_to_visit.city_id FK column exists", "city_id" in btv_cols)

    cur.execute("PRAGMA table_info(average_weather)")
    aw_cols = {row[1] for row in cur.fetchall()}
    check("average_weather.city_id FK column exists", "city_id" in aw_cols)

    conn.close()


# ── Data integrity ─────────────────────────────────────────────────────────────

def test_data_integrity():
    print("\n[Data integrity]")
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM countries")
    check("countries table has rows", cur.fetchone()[0] > 0)

    cur.execute("SELECT COUNT(*) FROM cities")
    check("cities table has rows", cur.fetchone()[0] > 0)

    cur.execute("SELECT COUNT(*) FROM flights")
    check("flights table has rows", cur.fetchone()[0] > 0)

    cur.execute("SELECT COUNT(*) FROM hotels")
    check("hotels table has rows", cur.fetchone()[0] > 0)

    cur.execute("SELECT COUNT(*) FROM activities")
    check("activities table has rows", cur.fetchone()[0] > 0)

    cur.execute("SELECT COUNT(*) FROM best_time_to_visit")
    check("best_time_to_visit table has rows", cur.fetchone()[0] > 0)

    cur.execute("SELECT COUNT(*) FROM average_weather")
    check("average_weather table has rows", cur.fetchone()[0] > 0)

    # All flights link to valid cities
    cur.execute("""
        SELECT COUNT(*) FROM flights
        WHERE origin_city_id NOT IN (SELECT id FROM cities)
           OR destination_city_id NOT IN (SELECT id FROM cities)
    """)
    check("all flights reference valid city IDs", cur.fetchone()[0] == 0)

    # All cities link to valid countries
    cur.execute("""
        SELECT COUNT(*) FROM cities
        WHERE country_id NOT IN (SELECT id FROM countries)
    """)
    check("all cities reference valid country IDs", cur.fetchone()[0] == 0)

    conn.close()


# ── City search via JOIN ───────────────────────────────────────────────────────

def test_city_search():
    print("\n[City search via JOIN]")
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT f.airline, f.price, f.flight_number,
               oc.name AS origin, dc.name AS destination
          FROM flights f
          JOIN cities oc ON f.origin_city_id      = oc.id
          JOIN cities dc ON f.destination_city_id = dc.id
         WHERE LOWER(oc.name) = 'tel aviv' AND LOWER(dc.name) = 'paris'
    """)
    rows = cur.fetchall()
    check("flights Tel Aviv -> Paris found", len(rows) > 0)

    cur.execute("""
        SELECT f.id FROM flights f
          JOIN cities dc ON f.destination_city_id = dc.id
         WHERE LOWER(dc.name) = 'atlantis'
    """)
    check("no flights to non-existent city", len(cur.fetchall()) == 0)

    conn.close()


# ── Country search (fallback) via JOIN ────────────────────────────────────────

def test_country_search():
    print("\n[Country search (fallback) via JOIN]")
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT dc.name AS destination_city, co.name AS destination_country
          FROM flights f
          JOIN cities dc   ON f.destination_city_id = dc.id
          JOIN countries co ON dc.country_id         = co.id
         WHERE LOWER(co.name) = 'france'
    """)
    rows = cur.fetchall()
    check("flights to France found by country JOIN", len(rows) > 0)
    cities = {r[0].lower() for r in rows}
    check("France results include Paris", "paris" in cities)

    cur.execute("""
        SELECT co.name AS country
          FROM flights f
          JOIN cities dc   ON f.destination_city_id = dc.id
          JOIN countries co ON dc.country_id         = co.id
         WHERE co.alpha2 = 'GB'
    """)
    check("flights to United Kingdom (GB) found by alpha2", len(cur.fetchall()) > 0)

    conn.close()


# ── Hotels via JOIN ────────────────────────────────────────────────────────────

def test_hotels():
    print("\n[Hotels via JOIN]")
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT h.name, h.price_per_night, h.stars
          FROM hotels h
          JOIN cities c ON h.city_id = c.id
         WHERE LOWER(c.name) = 'paris'
    """)
    rows = cur.fetchall()
    check("hotels in Paris found", len(rows) > 0)

    cur.execute("""
        SELECT h.name, h.price_per_night, h.stars
          FROM hotels h
          JOIN cities c ON h.city_id = c.id
         WHERE LOWER(c.name) = 'paris' AND h.price_per_night <= 200
    """)
    rows = cur.fetchall()
    check("hotels in Paris with max_price=200 filter works", all(r[1] <= 200 for r in rows))

    cur.execute("""
        SELECT h.id FROM hotels h
          JOIN cities c ON h.city_id = c.id
         WHERE LOWER(c.name) = 'atlantis'
    """)
    check("no hotels in non-existent city", len(cur.fetchall()) == 0)

    conn.close()


# ── Activities via JOIN ────────────────────────────────────────────────────────

def test_activities():
    print("\n[Activities via JOIN]")
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT a.name, a.category, a.price
          FROM activities a
          JOIN cities c ON a.city_id = c.id
         WHERE LOWER(c.name) = 'tokyo'
    """)
    rows = cur.fetchall()
    check("activities in Tokyo found", len(rows) > 0)

    conn.close()


# ── Best time to visit via JOIN ────────────────────────────────────────────────

def test_best_time():
    print("\n[Best time to visit via JOIN]")
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT b.months, b.reason
          FROM best_time_to_visit b
          JOIN cities c ON b.city_id = c.id
         WHERE LOWER(c.name) = 'amsterdam'
    """)
    rows = cur.fetchall()
    check("best_time_to_visit for Amsterdam found", len(rows) > 0)
    check("Amsterdam months field is non-empty", len(rows[0][0]) > 0 if rows else False)

    cur.execute("""
        SELECT b.id FROM best_time_to_visit b
          JOIN cities c ON b.city_id = c.id
         WHERE LOWER(c.name) = 'atlantis'
    """)
    check("no best_time for non-existent city", len(cur.fetchall()) == 0)

    conn.close()


# ── Average weather via JOIN ───────────────────────────────────────────────────

def test_weather():
    print("\n[Average weather via JOIN]")
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT w.season, w.temperature
          FROM average_weather w
          JOIN cities c ON w.city_id = c.id
         WHERE LOWER(c.name) = 'london' AND LOWER(w.season) = 'summer'
    """)
    rows = cur.fetchall()
    check("weather for London Summer found", len(rows) > 0)
    check("London Summer has temperature value", len(rows[0][1]) > 0 if rows else False)

    cur.execute("""
        SELECT w.id FROM average_weather w
          JOIN cities c ON w.city_id = c.id
         WHERE LOWER(c.name) = 'london' AND LOWER(w.season) = 'monsoon'
    """)
    check("no weather for invalid season", len(cur.fetchall()) == 0)

    conn.close()


# ── SQLiteDataProvider end-to-end ──────────────────────────────────────────────

def test_provider():
    print("\n[SQLiteDataProvider end-to-end]")
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from providers.sqlite_provider import SQLiteDataProvider
    p = SQLiteDataProvider()

    r = p.fetch_flights("Tel Aviv", "Paris")
    check("provider fetch_flights returns results", isinstance(r, list) and len(r) > 0 and "airline" in r[0])

    r = p.fetch_flights("Tel Aviv", "Lyon")
    check("provider fetch_flights fallback to France", "message" in r[0] and "France" in r[0]["message"])

    r = p.fetch_flights("Tel Aviv", "Atlantis")
    check("provider fetch_flights no results message", "message" in r[0])

    r = p.fetch_hotels("Tokyo")
    check("provider fetch_hotels returns results", len(r) > 0 and "stars" in r[0])

    r = p.fetch_hotels("Paris", max_price=200)
    check("provider fetch_hotels max_price filter", all(h["price_per_night"] <= 200 for h in r))

    r = p.fetch_hotels("Atlantis")
    check("provider fetch_hotels no results message", "message" in r[0])

    r = p.fetch_activities("Berlin")
    check("provider fetch_activities returns results", len(r) > 0 and "category" in r[0])

    r = p.get_best_time_to_visit("Amsterdam")
    check("provider get_best_time_to_visit returns months list", isinstance(r.get("months"), list))

    r = p.get_best_time_to_visit("Atlantis")
    check("provider get_best_time_to_visit no results message", "message" in r)

    r = p.get_average_weather("London", "Summer")
    check("provider get_average_weather returns temperature", "temperature" in r)

    r = p.get_average_weather("London", "Monsoon")
    check("provider get_average_weather invalid season message", "message" in r)


# ── Runner ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Testing: {DB_PATH}")
    test_schema()
    test_data_integrity()
    test_city_search()
    test_country_search()
    test_hotels()
    test_activities()
    test_best_time()
    test_weather()
    test_provider()
    print(f"\n{'='*40}")
    print(f"Results: {_passed} passed, {_failed} failed")
    if _failed:
        sys.exit(1)
