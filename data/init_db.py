"""
One-time script to create and seed data/travel_agency.db.

Schema:
    countries          — full ISO 3166-1 list, fetched from public CSV
    cities             — world cities with lat/lng, FK to countries
    flights            — origin/destination FK to cities
    hotels             — FK to cities
    activities         — FK to cities
    best_time_to_visit — recommended travel months per city, FK to cities
    average_weather    — seasonal temperatures per city, FK to cities

Run from the project root:  python data/init_db.py
"""

import csv
import io
import os
import sqlite3
import urllib.request

DB_PATH = os.path.join(os.path.dirname(__file__), "travel_agency.db")

COUNTRY_CODES_URL = "https://raw.githubusercontent.com/datasets/country-codes/main/data/country-codes.csv"
WORLD_CITIES_URL = "https://raw.githubusercontent.com/joelacus/world-cities/main/world_cities.csv"

SCHEMA = """
CREATE TABLE countries (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    alpha2    TEXT NOT NULL UNIQUE,
    alpha3    TEXT NOT NULL UNIQUE,
    numeric   TEXT NOT NULL UNIQUE,
    name      TEXT NOT NULL,
    region    TEXT,
    subregion TEXT
);

CREATE TABLE cities (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    country_id INTEGER NOT NULL REFERENCES countries(id),
    name       TEXT NOT NULL,
    lat        REAL NOT NULL,
    lng        REAL NOT NULL
);
CREATE INDEX idx_cities_country_id ON cities(country_id);

CREATE TABLE flights (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    origin_city_id      INTEGER NOT NULL REFERENCES cities(id),
    destination_city_id INTEGER NOT NULL REFERENCES cities(id),
    airline             TEXT    NOT NULL,
    price               INTEGER NOT NULL,
    flight_number       TEXT    NOT NULL,
    availability        TEXT    NOT NULL DEFAULT 'Available',
    departure_time      DATETIME NOT NULL,
    arrival_time        DATETIME NOT NULL
);
CREATE INDEX idx_flights_origin      ON flights(origin_city_id);
CREATE INDEX idx_flights_destination ON flights(destination_city_id);

CREATE TABLE hotels (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    city_id         INTEGER NOT NULL REFERENCES cities(id),
    name            TEXT    NOT NULL,
    price_per_night INTEGER NOT NULL,
    stars           INTEGER NOT NULL
);
CREATE INDEX idx_hotels_city ON hotels(city_id);

CREATE TABLE activities (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    city_id  INTEGER NOT NULL REFERENCES cities(id),
    name     TEXT    NOT NULL,
    category TEXT    NOT NULL,
    price    INTEGER NOT NULL
);
CREATE INDEX idx_activities_city ON activities(city_id);

CREATE TABLE best_time_to_visit (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    city_id INTEGER NOT NULL UNIQUE REFERENCES cities(id),
    months  TEXT    NOT NULL,
    reason  TEXT    NOT NULL
);

CREATE TABLE average_weather (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    city_id     INTEGER NOT NULL REFERENCES cities(id),
    season      TEXT    NOT NULL,
    temperature TEXT    NOT NULL,
    UNIQUE(city_id, season)
);
CREATE INDEX idx_weather_city ON average_weather(city_id);
"""


def fetch_csv(url: str) -> list[dict]:
    with urllib.request.urlopen(url) as resp:
        text = resp.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def seed_reference(conn: sqlite3.Connection) -> None:
    print("fetching country codes...")
    countries = fetch_csv(COUNTRY_CODES_URL)
    print("fetching world cities...")
    cities = fetch_csv(WORLD_CITIES_URL)

    country_rows = [
        (
            row["ISO3166-1-Alpha-2"],
            row["ISO3166-1-Alpha-3"],
            row["ISO3166-1-numeric"],
            row.get("official_name_en") or row.get("UNTERM English Short") or row.get("CLDR display name") or "",
            row.get("Region Name") or None,
            row.get("Sub-region Name") or None,
        )
        for row in countries
        if row.get("ISO3166-1-Alpha-2") and row.get("ISO3166-1-Alpha-3") and row.get("ISO3166-1-numeric")
    ]
    conn.executemany(
        "INSERT INTO countries (alpha2, alpha3, numeric, name, region, subregion) VALUES (?, ?, ?, ?, ?, ?)",
        country_rows,
    )

    alpha2_to_id = {alpha2: cid for cid, alpha2 in conn.execute("SELECT id, alpha2 FROM countries")}
    city_rows = []
    for city in cities:
        cid = alpha2_to_id.get(city["country"])
        if cid is None:
            continue
        city_rows.append((cid, city["name"], float(city["lat"]), float(city["lng"])))
    conn.executemany(
        "INSERT INTO cities (country_id, name, lat, lng) VALUES (?, ?, ?, ?)",
        city_rows,
    )
    print(f"  countries: {len(country_rows)}")
    print(f"  cities:    {len(city_rows)}")


def resolve_city(conn: sqlite3.Connection, name: str, alpha2: str) -> int:
    row = conn.execute(
        """SELECT cities.id
             FROM cities
             JOIN countries ON cities.country_id = countries.id
            WHERE LOWER(cities.name) = LOWER(?)
              AND countries.alpha2 = ?
            LIMIT 1""",
        (name, alpha2),
    ).fetchone()
    if row is None:
        raise ValueError(f"city not found in seed reference data: {name!r} ({alpha2})")
    return row[0]


def seed_travel(conn: sqlite3.Connection) -> None:
    # key → (city name as it appears in world_cities, ISO alpha-2)
    city_keys = {
        "tel aviv":  ("Tel Aviv",      "IL"),
        "paris":     ("Paris",         "FR"),
        "london":    ("London",        "GB"),
        "tokyo":     ("Tokyo",         "JP"),
        "new york":  ("New York City", "US"),
        "berlin":    ("Berlin",        "DE"),
        "amsterdam": ("Amsterdam",     "NL"),
        "athens": ("Athens", "GR"), 
    }
    ids = {key: resolve_city(conn, name, alpha2) for key, (name, alpha2) in city_keys.items()}

    flights = [
    
    ("tel aviv", "paris", "El Al", 350, "LY321", "Available", "2026-06-01 08:00:00", "2026-06-01 12:00:00"),
    ("tel aviv", "paris", "Air France", 420, "AF123", "Available", "2026-06-01 14:30:00", "2026-06-01 18:30:00"),
    ("tel aviv", "london", "British Airways", 450, "BA164", "Limited", "2026-06-01 09:00:00", "2026-06-01 13:30:00"),
    ("london", "paris", "Air France", 120, "AF124", "Available", "2026-06-01 16:00:00", "2026-06-01 17:30:00"),
    ("london", "new york", "Virgin Atlantic", 550, "VS001", "Available", "2026-06-01 18:30:00", "2026-06-01 21:30:00"),
    ("new york", "paris", "Air France", 480, "AF200", "Available", "2026-06-02 00:30:00", "2026-06-02 10:00:00"),
    ("tel aviv", "new york", "El Al", 1200, "LY001", "Available", "2026-06-01 05:00:00", "2026-06-01 16:00:00"),
    ("tel aviv", "athens", "Aegean", 100, "A301", "Available", "2026-06-01 10:00:00", "2026-06-01 12:00:00"),
    ("athens", "new york", "Delta", 450, "DL101", "Available", "2026-06-01 15:00:00", "2026-06-01 21:00:00"),
    ("london", "tokyo", "JAL", 700, "JL001", "Available", "2026-06-01 16:00:00", "2026-06-02 12:00:00"),
]
    conn.executemany(
        """INSERT INTO flights
               (origin_city_id, destination_city_id, airline, price, flight_number, availability, departure_time, arrival_time)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [(ids[o], ids[d], a, p, fn, av, dep, arr) for o, d, a, p, fn, av, dep, arr in flights],
    )

    hotels = [
        ("paris",     "Hotel de Ville",        150, 3),
        ("paris",     "Luxury Ritz",           600, 5),
        ("paris",     "Ibis Budget Paris",      85, 2),
        ("london",    "The Savoy",             450, 5),
        ("london",    "Premier Inn London",    120, 3),
        ("tokyo",     "Shibuya Capsule",        50, 2),
        ("tokyo",     "Park Hyatt Tokyo",      700, 5),
        ("new york",  "The Plaza",             850, 5),
        ("new york",  "Broadway Hotel",        190, 3),
        ("berlin",    "Berlin Central Hostel",  40, 1),
        ("berlin",    "Hilton Berlin",         220, 4),
        ("amsterdam", "Canal Boutique Hotel",  180, 4),
    ]
    conn.executemany(
        "INSERT INTO hotels (city_id, name, price_per_night, stars) VALUES (?, ?, ?, ?)",
        [(ids[c], n, p, s) for c, n, p, s in hotels],
    )

    activities = [
        ("paris",     "Louvre Museum",       "Culture",       20),
        ("paris",     "Eiffel Tower",        "Sightseeing",   35),
        ("paris",     "Disneyland Paris",    "Family",        95),
        ("london",    "London Eye",          "Sightseeing",   30),
        ("london",    "British Museum",      "Culture",        0),
        ("tokyo",     "Robot Cafe",          "Entertainment", 60),
        ("tokyo",     "Mount Fuji Day Trip", "Nature",       120),
        ("new york",  "Statue of Liberty",   "Sightseeing",   25),
        ("berlin",    "Berlin Wall Tour",    "History",       15),
        ("berlin",    "Techno Club Entry",   "Nightlife",     25),
        ("amsterdam", "Rijksmuseum",         "Culture",       22),
        ("amsterdam", "Canal Boat Tour",     "Sightseeing",   18),
    ]
    conn.executemany(
        "INSERT INTO activities (city_id, name, category, price) VALUES (?, ?, ?, ?)",
        [(ids[c], n, cat, p) for c, n, cat, p in activities],
    )

    # best_time_to_visit  (city_key, months CSV, reason)
    best_times = [
        ("paris",     "April,May,September",  "Pleasant weather and fewer crowds."),
        ("london",    "May,June,July",         "Best chance for sunshine and outdoor events."),
        ("tokyo",     "March,April,November",  "Cherry blossoms in spring, beautiful autumn foliage."),
        ("amsterdam", "April,May,September",   "Tulip season in spring and mild cycling weather."),
        ("new york",  "April,May,September",   "Mild temperatures and fewer tourists than summer."),
        ("berlin",    "May,June,July",         "Warm summers with long daylight hours and festivals."),
    ]
    conn.executemany(
        "INSERT INTO best_time_to_visit (city_id, months, reason) VALUES (?, ?, ?)",
        [(ids[c], m, r) for c, m, r in best_times],
    )

    # average_weather  (city_key, season, temperature)
    weather = [
        ("paris",     "Spring", "15C"), ("paris",     "Summer", "25C"),
        ("paris",     "Autumn", "16C"), ("paris",     "Winter",  "5C"),
        ("london",    "Spring", "12C"), ("london",    "Summer", "22C"),
        ("london",    "Autumn", "14C"), ("london",    "Winter",  "6C"),
        ("tokyo",     "Spring", "18C"), ("tokyo",     "Summer", "30C"),
        ("tokyo",     "Autumn", "21C"), ("tokyo",     "Winter",  "8C"),
        ("amsterdam", "Spring", "13C"), ("amsterdam", "Summer", "22C"),
        ("amsterdam", "Autumn", "14C"), ("amsterdam", "Winter",  "4C"),
        ("new york",  "Spring", "13C"), ("new york",  "Summer", "28C"),
        ("new york",  "Autumn", "15C"), ("new york",  "Winter",  "2C"),
        ("berlin",    "Spring", "12C"), ("berlin",    "Summer", "25C"),
        ("berlin",    "Autumn", "13C"), ("berlin",    "Winter",  "1C"),
    ]
    conn.executemany(
        "INSERT INTO average_weather (city_id, season, temperature) VALUES (?, ?, ?)",
        [(ids[c], s, t) for c, s, t in weather],
    )

    print(f"  flights:           {len(flights)}")
    print(f"  hotels:            {len(hotels)}")
    print(f"  activities:        {len(activities)}")
    print(f"  best_time_to_visit:{len(best_times)}")
    print(f"  average_weather:   {len(weather)}")


def create_travel_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executescript(SCHEMA)
        seed_reference(conn)
        seed_travel(conn)
        conn.commit()
    finally:
        conn.close()
    print(f"Database created at: {DB_PATH}")


if __name__ == "__main__":
    if os.path.exists(DB_PATH):
        answer = input(
            f"WARNING: '{DB_PATH}' already exists and will be wiped.\n"
            "Are you sure you want to continue? (yes/no): "
        ).strip().lower()
        if answer != "yes":
            print("Aborted. Database was not modified.")
            raise SystemExit(0)
        os.remove(DB_PATH)
    create_travel_db()
