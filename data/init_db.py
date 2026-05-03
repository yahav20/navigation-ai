"""
One-time script to create and seed data/travel_agency.db.
Run from the project root:  python data/init_db.py
"""

import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), "travel_agency.db")


def create_travel_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # ── Tables ────────────────────────────────────────────────────────────────
    cursor.execute("DROP TABLE IF EXISTS hotels")
    cursor.execute("DROP TABLE IF EXISTS flights")
    cursor.execute("DROP TABLE IF EXISTS activities")

    cursor.execute("""
        CREATE TABLE hotels (
            id              INTEGER PRIMARY KEY,
            city            TEXT    NOT NULL,
            name            TEXT    NOT NULL,
            price_per_night INTEGER NOT NULL,
            stars           INTEGER NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE flights (
            id                  INTEGER PRIMARY KEY,
            origin_city         TEXT    NOT NULL,
            origin_country      TEXT    NOT NULL,
            destination_city    TEXT    NOT NULL,
            destination_country TEXT    NOT NULL,
            airline             TEXT    NOT NULL,
            price               INTEGER NOT NULL,
            flight_number       TEXT    NOT NULL,
            availability        TEXT    NOT NULL DEFAULT 'Available'
        )
    """)

    cursor.execute("""
        CREATE TABLE activities (
            id       INTEGER PRIMARY KEY,
            city     TEXT    NOT NULL,
            name     TEXT    NOT NULL,
            category TEXT    NOT NULL,
            price    INTEGER NOT NULL
        )
    """)

    # Indexes for fast country-level and city-level lookups
    cursor.execute("CREATE INDEX idx_flights_dest_city    ON flights(LOWER(destination_city))")
    cursor.execute("CREATE INDEX idx_flights_dest_country ON flights(LOWER(destination_country))")
    cursor.execute("CREATE INDEX idx_hotels_city          ON hotels(LOWER(city))")
    cursor.execute("CREATE INDEX idx_activities_city      ON activities(LOWER(city))")

    # ── Seed data ─────────────────────────────────────────────────────────────

    # Hotels
    hotels = [
        ("paris",    "Hotel de Ville",       150, 3),
        ("paris",    "Luxury Ritz",          600, 5),
        ("paris",    "Ibis Budget Paris",     85, 2),
        ("london",   "The Savoy",            450, 5),
        ("london",   "Premier Inn London",   120, 3),
        ("tokyo",    "Shibuya Capsule",       50, 2),
        ("tokyo",    "Park Hyatt Tokyo",     700, 5),
        ("new york", "The Plaza",            850, 5),
        ("new york", "Broadway Hotel",       190, 3),
        ("berlin",   "Berlin Central Hostel", 40, 1),
        ("berlin",   "Hilton Berlin",        220, 4),
        ("amsterdam","Canal Boutique Hotel", 180, 4),
    ]
    cursor.executemany(
        "INSERT INTO hotels (city, name, price_per_night, stars) VALUES (?, ?, ?, ?)",
        hotels,
    )

    # Flights  (origin_city, origin_country, dest_city, dest_country, airline, price, flight_number, availability)
    flights = [
        # From TLV (Israel)
        ("TLV", "Israel", "Paris",     "France",         "El Al",           350, "LY321", "Available"),
        ("TLV", "Israel", "Paris",     "France",         "Air France",      420, "AF123", "Available"),
        ("TLV", "Israel", "London",    "United Kingdom", "British Airways", 450, "BA164", "Limited"),
        ("TLV", "Israel", "London",    "United Kingdom", "Virgin Atlantic", 390, "VS100", "Available"),
        ("TLV", "Israel", "Tokyo",     "Japan",          "El Al",           950, "LY091", "Available"),
        ("TLV", "Israel", "Tokyo",     "Japan",          "Emirates",        820, "EK312", "Available"),
        ("TLV", "Israel", "New York",  "United States",  "United",          750, "UA445", "Limited"),
        ("TLV", "Israel", "Berlin",    "Germany",        "Lufthansa",       280, "LH909", "Available"),
        ("TLV", "Israel", "Berlin",    "Germany",        "Ryanair",         110, "FR101", "Available"),
        ("TLV", "Israel", "Amsterdam", "Netherlands",    "KLM",             410, "KL456", "Available"),
        # From London (United Kingdom)
        ("London", "United Kingdom", "Paris",    "France", "Air France", 120, "AF124", "Available"),
        ("London", "United Kingdom", "Tokyo",    "Japan",  "JAL",        890, "JL402", "Available"),
        ("London", "United Kingdom", "New York", "United States", "Virgin Atlantic", 550, "VS001", "Limited"),
        # From New York (United States)
        ("New York", "United States", "London", "United Kingdom", "Virgin Atlantic", 550, "VS002", "Limited"),
        ("New York", "United States", "Paris",  "France",         "Air France",      480, "AF200", "Available"),
    ]
    cursor.executemany(
        """INSERT INTO flights
               (origin_city, origin_country, destination_city, destination_country,
                airline, price, flight_number, availability)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        flights,
    )

    # Activities
    activities = [
        ("paris",     "Louvre Museum",          "Culture",       20),
        ("paris",     "Eiffel Tower",           "Sightseeing",   35),
        ("paris",     "Disneyland Paris",       "Family",        95),
        ("london",    "London Eye",             "Sightseeing",   30),
        ("london",    "British Museum",         "Culture",        0),
        ("tokyo",     "Robot Cafe",             "Entertainment", 60),
        ("tokyo",     "Mount Fuji Day Trip",    "Nature",       120),
        ("new york",  "Statue of Liberty",      "Sightseeing",   25),
        ("berlin",    "Berlin Wall Tour",       "History",       15),
        ("berlin",    "Techno Club Entry",      "Nightlife",     25),
        ("amsterdam", "Rijksmuseum",            "Culture",       22),
        ("amsterdam", "Canal Boat Tour",        "Sightseeing",   18),
    ]
    cursor.executemany(
        "INSERT INTO activities (city, name, category, price) VALUES (?, ?, ?, ?)",
        activities,
    )

    conn.commit()
    conn.close()
    print(f"Database created at: {DB_PATH}")
    print(f"  hotels:     {len(hotels)}")
    print(f"  flights:    {len(flights)}")
    print(f"  activities: {len(activities)}")


if __name__ == "__main__":
    if os.path.exists(DB_PATH):
        answer = input(
            f"WARNING: '{DB_PATH}' already exists and will be wiped.\n"
            "Are you sure you want to continue? (yes/no): "
        ).strip().lower()
        if answer != "yes":
            print("Aborted. Database was not modified.")
            raise SystemExit(0)
    create_travel_db()
