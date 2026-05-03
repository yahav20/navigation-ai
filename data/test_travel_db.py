"""
Standalone tests for travel_agency.db.
Run from the project root:  python data/test_travel_db.py
No external dependencies required.
"""

import os
import sqlite3

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


# ── Schema ────────────────────────────────────────────────────────────────────

def test_schema():
    print("\n[Schema]")
    conn = get_conn()
    cur = conn.cursor()

    for table in ("flights", "hotels", "activities"):
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
        check(f"table '{table}' exists", cur.fetchone() is not None)

    cur.execute("PRAGMA table_info(flights)")
    flight_cols = {row[1] for row in cur.fetchall()}
    for col in ("origin_city", "origin_country", "destination_city", "destination_country",
                "airline", "price", "flight_number", "availability"):
        check(f"flights.{col} exists", col in flight_cols)

    conn.close()


# ── Data integrity ─────────────────────────────────────────────────────────────

def test_data_integrity():
    print("\n[Data integrity]")
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM flights")
    check("flights table has rows", cur.fetchone()[0] > 0)

    cur.execute("SELECT COUNT(*) FROM hotels")
    check("hotels table has rows", cur.fetchone()[0] > 0)

    cur.execute("SELECT COUNT(*) FROM activities")
    check("activities table has rows", cur.fetchone()[0] > 0)

    cur.execute("SELECT COUNT(*) FROM flights WHERE origin_country = '' OR destination_country = ''")
    check("no flights with empty country", cur.fetchone()[0] == 0)

    conn.close()


# ── City search ────────────────────────────────────────────────────────────────

def test_city_search():
    print("\n[City search]")
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        "SELECT * FROM flights WHERE LOWER(destination_city) = ?", ("paris",)
    )
    rows = cur.fetchall()
    check("flights to Paris found by city", len(rows) > 0)

    cur.execute(
        "SELECT * FROM flights WHERE LOWER(destination_city) = ?", ("london",)
    )
    rows = cur.fetchall()
    check("flights to London found by city", len(rows) > 0)

    cur.execute(
        "SELECT * FROM flights WHERE LOWER(destination_city) = ?", ("atlantis",)
    )
    rows = cur.fetchall()
    check("no flights to non-existent city", len(rows) == 0)

    conn.close()


# ── Country search (fallback) ──────────────────────────────────────────────────

def test_country_search():
    print("\n[Country search (fallback)]")
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        "SELECT destination_city, destination_country FROM flights WHERE LOWER(destination_country) = ?",
        ("france",),
    )
    rows = cur.fetchall()
    check("flights to France found by country", len(rows) > 0)
    cities = {r[0].lower() for r in rows}
    check("France results include Paris", "paris" in cities)

    cur.execute(
        "SELECT destination_city FROM flights WHERE LOWER(destination_country) = ?",
        ("united kingdom",),
    )
    rows = cur.fetchall()
    check("flights to United Kingdom found by country", len(rows) > 0)

    conn.close()


# ── Hotels ─────────────────────────────────────────────────────────────────────

def test_hotels():
    print("\n[Hotels]")
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT * FROM hotels WHERE LOWER(city) = ?", ("paris",))
    rows = cur.fetchall()
    check("hotels in Paris found", len(rows) > 0)

    cur.execute("SELECT * FROM hotels WHERE LOWER(city) = ?", ("atlantis",))
    rows = cur.fetchall()
    check("no hotels in non-existent city", len(rows) == 0)

    conn.close()


# ── Activities ─────────────────────────────────────────────────────────────────

def test_activities():
    print("\n[Activities]")
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT * FROM activities WHERE LOWER(city) = ?", ("tokyo",))
    rows = cur.fetchall()
    check("activities in Tokyo found", len(rows) > 0)

    conn.close()


# ── Runner ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Testing: {DB_PATH}")
    test_schema()
    test_data_integrity()
    test_city_search()
    test_country_search()
    test_hotels()
    test_activities()
    print(f"\n{'='*40}")
    print(f"Results: {_passed} passed, {_failed} failed")
    if _failed:
        raise SystemExit(1)
