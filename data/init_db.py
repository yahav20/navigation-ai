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
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alpha2 TEXT NOT NULL UNIQUE,
    alpha3 TEXT NOT NULL UNIQUE,
    numeric TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    region TEXT,
    subregion TEXT
);

CREATE TABLE cities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    country_id INTEGER NOT NULL REFERENCES countries (id),
    name TEXT NOT NULL,
    lat REAL NOT NULL,
    lng REAL NOT NULL
);
CREATE INDEX idx_cities_country_id ON cities (country_id);

CREATE TABLE activities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    city_id INTEGER NOT NULL REFERENCES cities (id),
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    price INTEGER NOT NULL,
    min_age INTEGER DEFAULT 0
);

CREATE TABLE average_weather (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    city_id INTEGER NOT NULL REFERENCES cities (id),
    season TEXT NOT NULL,
    temperature TEXT NOT NULL,
    UNIQUE (city_id, season)
);

CREATE TABLE best_time_to_visit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    city_id INTEGER NOT NULL UNIQUE REFERENCES cities (id),
    months TEXT NOT NULL,
    reason TEXT NOT NULL
);

CREATE TABLE flights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    origin_city_id INTEGER NOT NULL REFERENCES cities (id),
    destination_city_id INTEGER NOT NULL REFERENCES cities (id),
    airline TEXT NOT NULL,
    price INTEGER NOT NULL,
    flight_number TEXT NOT NULL,
    departure_time DATETIME NOT NULL,
    arrival_time DATETIME NOT NULL,
    duration_minutes INTEGER NOT NULL,
    availability TEXT NOT NULL DEFAULT 'Available'
);
CREATE INDEX idx_flights_destination ON flights (destination_city_id);

CREATE TABLE hotels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    city_id INTEGER NOT NULL REFERENCES cities (id),
    name TEXT NOT NULL,
    price_per_night INTEGER NOT NULL,
    stars INTEGER NOT NULL,
    min_age INTEGER DEFAULT 0,
    hotel_type TEXT CHECK (hotel_type IN ('Luxury', 'Family', 'Romantic', 'Backpacker', 'Business')),
    distance_from_center_km REAL
);
CREATE INDEX idx_hotels_city ON hotels (city_id);

CREATE TABLE transportation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_city_id INTEGER NOT NULL REFERENCES cities (id),
    to_city_id INTEGER NOT NULL REFERENCES cities (id),
    transport_type TEXT CHECK (transport_type IN ('Train', 'Bus', 'Rental Car', 'Shuttle')),
    price_estimate REAL,
    duration_hours REAL,
    FOREIGN KEY (from_city_id) REFERENCES cities (id),
    FOREIGN KEY (to_city_id) REFERENCES cities (id)
);
"""

def fetch_csv(url: str) -> list[dict]:
    with urllib.request.urlopen(url) as resp:
        text = resp.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))

def seed_reference(conn: sqlite3.Connection) -> None:
    print("Fetching reference data...")
    countries = fetch_csv(COUNTRY_CODES_URL)
    cities = fetch_csv(WORLD_CITIES_URL)

    country_rows = [
        (r["ISO3166-1-Alpha-2"], r["ISO3166-1-Alpha-3"], r["ISO3166-1-numeric"], 
         r.get("official_name_en") or r.get("UNTERM English Short") or "", 
         r.get("Region Name"), r.get("Sub-region Name"))
        for r in countries if r.get("ISO3166-1-Alpha-2")
    ]
    conn.executemany("INSERT INTO countries (alpha2, alpha3, numeric, name, region, subregion) VALUES (?,?,?,?,?,?)", country_rows)

    alpha2_to_id = {a2: cid for cid, a2 in conn.execute("SELECT id, alpha2 FROM countries")}
    city_rows = [(alpha2_to_id[c["country"]], c["name"], float(c["lat"]), float(c["lng"])) 
                 for c in cities if c["country"] in alpha2_to_id]
    conn.executemany("INSERT INTO cities (country_id, name, lat, lng) VALUES (?,?,?,?)", city_rows)

def resolve_city(conn: sqlite3.Connection, name: str, alpha2: str) -> int:
    row = conn.execute("SELECT cities.id FROM cities JOIN countries ON cities.country_id = countries.id WHERE LOWER(cities.name) = LOWER(?) AND countries.alpha2 = ? LIMIT 1", (name, alpha2)).fetchone()
    return row[0] if row else None

def seed_travel(conn: sqlite3.Connection) -> None:
    city_keys = {
        "tel aviv": ("Tel Aviv", "IL"), "paris": ("Paris", "FR"), "london": ("London", "GB"),
        "tokyo": ("Tokyo", "JP"), "new york": ("New York City", "US"), "berlin": ("Berlin", "DE"),
        "amsterdam": ("Amsterdam", "NL")
    }
    ids = {k: resolve_city(conn, name, a2) for k, (name, a2) in city_keys.items()}

    # Flights: (origin, dest, airline, price, flight_no, availability, dep, arr, duration_minutes)
    flights = [
        ("tel aviv", "paris", "El Al", 350, "LY321", "Available", "2026-06-01 08:00:00", "2026-06-01 11:50:00", 290),
        ("tel aviv", "london", "British Airways", 450, "BA164", "Limited", "2026-06-01 07:30:00", "2026-06-01 11:00:00", 330),
        
        ("new york", "london", "Virgin Atlantic", 550, "VS002", "Available", "2026-06-01 19:00:00", "2026-06-02 07:00:00", 420),
        ("new york", "paris", "Air France", 480, "AF200", "Available", "2026-06-01 18:30:00", "2026-06-02 07:45:00", 435),
        ("new york", "amsterdam", "Delta", 500, "DL300", "Available", "2026-06-01 20:00:00", "2026-06-02 09:30:00", 450),
        
        ("london", "berlin", "Ryanair", 60, "FR555", "Available", "2026-06-02 10:00:00", "2026-06-02 12:00:00", 120),
        ("paris", "berlin", "Lufthansa", 160, "LH111", "Available", "2026-06-02 11:30:00", "2026-06-02 13:15:00", 105),
        ("amsterdam", "berlin", "EasyJet", 80, "U2444", "Available", "2026-06-02 13:00:00", "2026-06-02 14:20:00", 80),
        
        ("paris", "london", "Air France", 120, "AF125", "Available", "2026-06-01 13:00:00", "2026-06-01 14:15:00", 75),
        ("london", "tokyo", "JAL", 890, "JL402", "Available", "2026-06-01 19:00:00", "2026-06-02 15:00:00", 780),
    ]
    conn.executemany(
        "INSERT INTO flights (origin_city_id, destination_city_id, airline, price, flight_number, availability, departure_time, arrival_time, duration_minutes) VALUES (?,?,?,?,?,?,?,?,?)",
        [(ids[o], ids[d], a, p, fn, av, dep, arr, dur) for o,d,a,p,fn,av,dep,arr,dur in flights]
    )

    # Hotels: (city, name, price, stars, min_age, type, distance)
    hotels = [
        ("paris", "Luxury Ritz", 600, 5, 18, "Luxury", 0.2),
        ("london", "Premier Inn London", 120, 3, 0, "Business", 2.5),
        ("tokyo", "Shibuya Capsule", 50, 2, 18, "Backpacker", 1.0),
        ("tel aviv", "The Norman", 500, 5, 21, "Luxury", 0.8),
       
        ("berlin", "Hilton Berlin", 220, 4, 18, "Business", 0.5),
        ("berlin", "Berlin Central Hostel", 40, 1, 18, "Backpacker", 2.0),
  
        ("new york", "The Plaza", 850, 5, 21, "Luxury", 0.1),
        ("new york", "Broadway Hotel", 190, 3, 18, "Family", 1.2),
    ]
    conn.executemany(
        "INSERT INTO hotels (city_id, name, price_per_night, stars, min_age, hotel_type, distance_from_center_km) VALUES (?,?,?,?,?,?,?)",
        [(ids[c], n, p, s, ma, t, dist) for c,n,p,s,ma,t,dist in hotels]
    )

    activities = [
        ("paris", "Louvre Museum", "Culture", 20, 0),
        ("tokyo", "Robot Cafe", "Entertainment", 60, 12),
        ("london", "Pub Crawl", "Nightlife", 30, 18),
        ("berlin", "Berlin Wall Tour", "History", 15, 0),
        ("new york", "Statue of Liberty", "Sightseeing", 25, 0),
    ]
    conn.executemany(
        "INSERT INTO activities (city_id, name, category, price, min_age) VALUES (?,?,?,?,?)",
        [(ids[c], n, cat, p, ma) for c,n,cat,p,ma in activities]
    )

    transport = [
        ("paris", "london", "Train", 95, 2.25),
        ("london", "paris", "Bus", 35, 7.0)
    ]
    conn.executemany(
        "INSERT INTO transportation (from_city_id, to_city_id, transport_type, price_estimate, duration_hours) VALUES (?,?,?,?,?)",
        [(ids[f], ids[t], ty, p, d) for f,t,ty,p,d in transport]
    )

def create_travel_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
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
        os.remove(DB_PATH)
    create_travel_db()