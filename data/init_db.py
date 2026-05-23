# Recommendation db file

import csv
import io
import json
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
    departure_time DATETIME NOT NULL,
    arrival_time DATETIME NOT NULL,
    duration_minutes INTEGER NOT NULL,
    availability        TEXT    NOT NULL DEFAULT 'Available',
    duration_hours      REAL
);
CREATE INDEX idx_flights_origin      ON flights(origin_city_id);
CREATE INDEX idx_flights_destination ON flights(destination_city_id);

CREATE TABLE hotels (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    city_id         INTEGER NOT NULL REFERENCES cities(id),
    name            TEXT    NOT NULL,
    price_per_night INTEGER NOT NULL,
    stars           INTEGER NOT NULL,
    min_age         INTEGER DEFAULT 0,
    hotel_type      TEXT CHECK (hotel_type IN ('Luxury', 'Family', 'Romantic', 'Backpacker', 'Business')),
    distance_from_center_km REAL,
    breakfast_available BOOLEAN DEFAULT FALSE,
    breakfast_price REAL,
    amenities TEXT, 
    is_kosher BOOLEAN DEFAULT FALSE,
    latitude REAL,
    longitude REAL
);
CREATE INDEX idx_hotels_city ON hotels(city_id);

CREATE TABLE activities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    city_id INTEGER NOT NULL REFERENCES cities(id),
    name TEXT NOT NULL,
    categories TEXT NOT NULL, 
    price INTEGER NOT NULL,
    min_age INTEGER DEFAULT 0,
    latitude REAL,
    longitude REAL,
    avg_duration_minutes INTEGER,
    opening_time TEXT,
    closing_time TEXT,
    operating_days TEXT, 
    best_time_of_day TEXT, 
    food_available BOOLEAN DEFAULT FALSE,
    requires_booking BOOLEAN DEFAULT FALSE,
    rating REAL
);
CREATE INDEX idx_activities_city ON activities(city_id);

CREATE TABLE activity_features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL
);

CREATE TABLE activity_feature_mapping (
    activity_id INTEGER NOT NULL,
    feature_id INTEGER NOT NULL,
    PRIMARY KEY (activity_id, feature_id),
    FOREIGN KEY (activity_id) REFERENCES activities(id) ON DELETE CASCADE,
    FOREIGN KEY (feature_id) REFERENCES activity_features(id) ON DELETE CASCADE
);

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

CREATE TABLE city_tags (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    city_id INTEGER NOT NULL REFERENCES cities(id),
    tag     TEXT    NOT NULL,
    UNIQUE(city_id, tag)
);
CREATE INDEX idx_city_tags_city ON city_tags(city_id);
CREATE INDEX idx_city_tags_tag  ON city_tags(tag);

CREATE TABLE recommended_duration (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    city_id  INTEGER NOT NULL UNIQUE REFERENCES cities(id),
    min_days INTEGER NOT NULL,
    max_days INTEGER NOT NULL,
    notes    TEXT    NOT NULL
);

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

    city_keys = {
        "tel aviv":  ("Tel Aviv",      "IL"),
        "paris":     ("Paris",         "FR"),
        "london":    ("London",        "GB"),
        "tokyo":     ("Tokyo",         "JP"),
        "new york":  ("New York City", "US"),
        "berlin":    ("Berlin",        "DE"),
        "amsterdam": ("Amsterdam",     "NL"),
    }
    ids = {key: resolve_city(conn, name, alpha2) for key, (name, alpha2) in city_keys.items()}

    flights = [
    ("tel aviv", "paris", "El Al", 350, "LY321", "2026-06-01 08:00:00", "2026-06-01 11:50:00", 290, "Available", 4.5),
    ("tel aviv", "paris", "Air France", 420, "AF123", "2026-06-01 09:30:00", "2026-06-01 13:00:00", 270, "Available", 4.5),
    ("tel aviv", "paris", "EasyJet", 180, "U2111", "2026-06-01 14:00:00", "2026-06-01 18:30:00", 270, "Available", 4.5), # אופציית לואו-קוסט
    ("paris", "tel aviv", "El Al", 355, "LY322", "2026-06-04 15:00:00", "2026-06-04 20:50:00", 295, "Available", 4.8),
    ("paris", "tel aviv", "Air France", 400, "AF126", "2026-06-04 10:00:00", "2026-06-04 15:30:00", 270, "Available", 4.5),
    ("paris", "tel aviv", "EasyJet", 150, "U2112", "2026-06-04 21:00:00", "2026-06-05 02:30:00", 270, "Available", 4.5),
    ("tel aviv", "london", "British Airways", 450, "BA164","2026-06-01 07:30:00", "2026-06-01 11:00:00",330, "Limited", 5.0),
    ("tel aviv", "london", "Virgin Atlantic", 390, "VS100","2026-06-01 10:00:00", "2026-06-01 13:30:00", 300, "Available", 5.0),
    ("tel aviv", "tokyo", "El Al", 950, "LY091","2026-06-01 22:00:00", "2026-06-02 14:30:00",690, "Available", 11.5),
    ("tel aviv", "tokyo", "Emirates", 820, "EK312", "2026-06-01 23:30:00", "2026-06-02 18:30:00",780, "Available", 13.0),
    ("tel aviv", "new york", "United", 750, "UA445","2026-06-01 16:00:00", "2026-06-01 21:00:00",660, "Limited", 11.0),
    ("tel aviv", "berlin", "Lufthansa", 280, "LH909","2026-06-01 06:30:00", "2026-06-01 09:00:00",210, "Available", 3.5),
    ("tel aviv", "berlin", "Ryanair", 110, "FR101","2026-06-01 12:00:00", "2026-06-01 14:30:00",210, "Available", 3.5),
    ("tel aviv", "amsterdam", "KLM", 410, "KL456","2026-06-01 11:00:00", "2026-06-01 14:30:00",270, "Available", 4.5),
    ("london", "paris", "Air France", 120, "AF124","2026-06-01 12:00:00", "2026-06-01 14:15:00",75, "Available", 1.25),
    ("paris", "tel aviv", "El Al", 355, "LY322","2026-06-02 08:00:00", "2026-06-02 11:50:00",295, "Available", 5.0),
    ("paris", "london", "Air France", 120, "AF125","2026-06-01 13:00:00", "2026-06-01 14:15:00",75, "Available", 1.25),
    ("london", "tokyo", "JAL", 890, "JL402","2026-06-01 19:00:00", "2026-06-02 15:00:00",780, "Available", 12.0),
    ("london", "new york", "Virgin Atlantic", 550, "VS001","2026-06-01 15:00:00", "2026-06-01 18:30:00",450, "Limited", 7.5),
    ("new york", "london", "Virgin Atlantic", 550, "VS002","2026-06-01 19:00:00", "2026-06-02 07:00:00",420, "Available", 7.0),
    ("new york", "paris", "Air France", 480, "AF200","2026-06-01 18:30:00", "2026-06-02 07:45:00",435, "Available", 7.5),
    ("new york", "amsterdam", "Delta", 500, "DL300","2026-06-01 20:00:00", "2026-06-02 09:30:00",450, "Available", 7.5),
    ("london", "berlin", "Ryanair", 60, "FR555","2026-06-02 10:00:00", "2026-06-02 12:00:00",120, "Available", 2.0),
    ("paris", "berlin", "Lufthansa", 160, "LH111","2026-06-02 11:30:00", "2026-06-02 13:15:00",105, "Available", 1.75),
    ("amsterdam", "berlin", "EasyJet", 80, "U2444","2026-06-02 13:00:00", "2026-06-02 14:20:00",80, "Available", 1.33),
    
    ]
    conn.executemany(
        """
        INSERT INTO flights (
            origin_city_id,
            destination_city_id,
            airline,
            price,
            flight_number,
            departure_time,
            arrival_time,
            duration_minutes,
            availability,
            duration_hours
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (ids[o], ids[d], a, p, fn, dep, arr, dur, av, dh)
            for o, d, a, p, fn, dep, arr, dur, av, dh in flights
        ],
    )

    hotels = [
        ("paris", "Hotel de Ville", 150, 3, 0, "Family", 1.0, True, 15.0, json.dumps(["WiFi", "Family Rooms"]), False, 48.8566, 2.3522),
        ("paris", "Luxury Ritz", 600, 5, 18, "Luxury", 0.2, True, 45.0, json.dumps(["Pool", "Spa", "Gym", "Bar"]), False, 48.8680, 2.3280),
        ("paris", "Ibis Budget Paris", 80, 2, 0, "Backpacker", 3.5, False, 8.0, json.dumps(["WiFi", "Vending Machine"]), False, 48.8825, 2.3222),
        ("paris", "Le Marais Boutique (Kosher)", 220, 4, 0, "Romantic", 0.5, True, 20.0, json.dumps(["WiFi", "Terrace"]), True, 48.8575, 2.3588),
        
        ("london", "The Savoy", 450, 5, 18, "Luxury", 0.5, True, 35.0, json.dumps(["Pool", "Gym", "River View"]), False, 51.5100, -0.1200),
        ("london", "Premier Inn London", 120, 3, 0, "Business", 2.5, True, 10.0, json.dumps(["WiFi", "Restaurant"]), False, 51.5300, -0.1250),
        
        ("tokyo", "Park Hyatt Tokyo", 700, 5, 18, "Luxury", 1.5, True, 50.0, json.dumps(["Pool", "Spa", "City View"]), False, 35.6853, 139.6912),
        
        ("new york", "The Plaza", 850, 5, 21, "Luxury", 0.1, True, 40.0, json.dumps(["Spa", "Gym", "Room Service"]), False, 40.7644, -73.9745),
        
        ("berlin", "Hilton Berlin", 220, 4, 18, "Business", 0.5, True, 25.0, json.dumps(["Pool", "Gym", "Executive Lounge"]), False, 52.5126, 13.3916),
        
        ("amsterdam", "Canal Boutique Hotel", 180, 4, 18, "Romantic", 0.7, True, 20.0, json.dumps(["WiFi", "Bicycle Rental"]), False, 52.3670, 4.8870),

        ("tel aviv", "The Norman", 500, 5, 21, "Luxury", 0.8, True, 30.0, json.dumps(["Rooftop Pool", "Spa", "Fine Dining"]), False, 32.0645, 34.7744),
        ("tel aviv", "Dan Tel Aviv", 350, 5, 0, "Family", 0.1, True, 25.0, json.dumps(["Pool", "Beachfront", "Kids Club"]), True, 32.0792, 34.7672),
    ]
    
    conn.executemany(
        """INSERT INTO hotels (city_id, name, price_per_night, stars, min_age, hotel_type, distance_from_center_km, breakfast_available, breakfast_price, amenities, is_kosher, latitude, longitude) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [(ids[c], n, p, s, ma, t, dist, ba, bp, am, kos, lat, lng) for c, n, p, s, ma, t, dist, ba, bp, am, kos, lat, lng in hotels],
    )

    activities = [
        ("paris", "Louvre Museum", json.dumps(["Culture", "Museum", "Indoor"]), 20, 0, 48.8606, 2.3376, 180, "09:00", "18:00", "1,3,4,5,6,7", "MORNING", True, True, 4.8),
        ("paris", "Eiffel Tower", json.dumps(["Sightseeing", "Landmark", "Outdoor"]), 35, 0, 48.8584, 2.2945, 120, "09:30", "23:45", "1,2,3,4,5,6,7", "SUNSET", True, True, 4.7),
        ("paris", "Disneyland Paris", json.dumps(["Family", "Theme Park", "Outdoor"]), 95, 0, 48.8722, 2.7758, 480, "09:30", "21:00", "1,2,3,4,5,6,7", "MORNING", True, True, 4.5),
        ("paris", "Seine River Cruise", json.dumps(["Sightseeing", "Relaxing", "Outdoor"]), 15, 0, 48.8616, 2.3266, 60, "10:00", "22:00", "1,2,3,4,5,6,7", "SUNSET", False, False, 4.6),
        ("paris", "Montmartre Walking Tour", json.dumps(["Culture", "Walking", "Outdoor"]), 0, 0, 48.8867, 2.3431, 120, "10:00", "16:00", "1,2,3,4,5,6,7", "MORNING", False, False, 4.7), # פעילות חינמית! מעולה לתקציב
        ("paris", "Palace of Versailles", json.dumps(["Historic", "Day Trip", "Indoor", "Outdoor"]), 45, 0, 48.8049, 2.1204, 300, "09:00", "18:30", "2,3,4,5,6,7", "MORNING", True, True, 4.8),
        ("paris", "Musee d'Orsay", json.dumps(["Culture", "Museum", "Indoor"]), 18, 0, 48.8599, 2.3266, 150, "09:30", "18:00", "2,3,4,5,6,7", "AFTERNOON", True, True, 4.9),
        ("paris", "Latin Quarter Food Tour", json.dumps(["Food", "Walking", "Outdoor"]), 65, 0, 48.8500, 2.3499, 180, "11:00", "15:00", "1,2,3,4,5,6,7", "AFTERNOON", True, True, 4.8),
        
        ("london", "London Eye", json.dumps(["Sightseeing", "Viewpoint"]), 30, 0, 51.5033, -0.1195, 60, "10:00", "20:30", "1,2,3,4,5,6,7", "SUNSET", False, True, 4.5),
        ("london", "British Museum", json.dumps(["Culture", "Museum", "Indoor"]), 0, 0, 51.5194, -0.1270, 150, "10:00", "17:00", "1,2,3,4,5,6,7", "MORNING", True, False, 4.8),
        ("london", "Pub Crawl", json.dumps(["Nightlife", "Bar", "Social"]), 30, 18, 51.5126, -0.1325, 240, "19:00", "23:59", "4,5,6", "NIGHT", True, True, 4.4),

        ("tokyo", "Robot Cafe", json.dumps(["Entertainment", "Unique", "Indoor"]), 60, 12, 35.6943, 139.7028, 90, "15:00", "23:00", "1,2,3,4,5,6,7", "EVENING", True, True, 4.1),
        ("tokyo", "Mount Fuji Day Trip", json.dumps(["Nature", "Day Trip", "Outdoor"]), 120, 0, 35.3606, 138.7274, 600, "08:00", "18:00", "1,2,3,4,5,6,7", "MORNING", True, True, 4.9),

        ("new york", "Statue of Liberty", json.dumps(["Sightseeing", "Historic", "Outdoor"]), 25, 0, 40.6892, -74.0445, 180, "09:00", "17:00", "1,2,3,4,5,6,7", "MORNING", True, True, 4.7),

        ("berlin", "Berlin Wall Tour", json.dumps(["History", "Walking", "Outdoor"]), 15, 0, 52.5351, 13.3902, 120, "10:00", "14:00", "1,2,3,4,5,6,7", "MORNING", False, False, 4.6),
        ("berlin", "Techno Club Entry", json.dumps(["Nightlife", "Club", "Indoor"]), 25, 18, 52.5108, 13.4318, 300, "23:59", "08:00", "5,6", "NIGHT", False, False, 4.4),

        ("amsterdam", "Rijksmuseum", json.dumps(["Culture", "Museum", "Indoor"]), 22, 0, 52.3600, 4.8852, 150, "09:00", "17:00", "1,2,3,4,5,6,7", "AFTERNOON", True, True, 4.8),
        ("amsterdam", "Canal Boat Tour", json.dumps(["Sightseeing", "Relaxing", "Outdoor"]), 18, 0, 52.3780, 4.9000, 75, "10:00", "21:00", "1,2,3,4,5,6,7", "SUNSET", False, True, 4.5),
    ]
    conn.executemany(
        """
        INSERT INTO activities (
            city_id, name, categories, price, min_age,
            latitude, longitude, avg_duration_minutes,
            opening_time, closing_time, operating_days,
            best_time_of_day, food_available, requires_booking, rating
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (ids[c], n, cat, p, ma, lat, lng, dur, op, cl, days, bt, food, book, rat)
            for c, n, cat, p, ma, lat, lng, dur, op, cl, days, bt, food, book, rat in activities
        ],
    )
    
    feature_names = [
        "Kosher", 
        "Vegan Options", 
        "Vegetarian Options", 
        "Wheelchair Accessible", 
        "Pet Friendly", 
        "Air Conditioned",
        "Guided Tour",
        "Family Friendly"
    ]
    
    conn.executemany(
        "INSERT INTO activity_features (name) VALUES (?)",
        [(name,) for name in feature_names]
    )
    
    feature_ids = {name: fid for fid, name in conn.execute("SELECT id, name FROM activity_features")}
    
    activity_ids = {name: aid for aid, name in conn.execute("SELECT id, name FROM activities")}
    
   
    activity_features_map = [
       
        ("Louvre Museum", ["Wheelchair Accessible", "Air Conditioned", "Guided Tour"]),
        ("Eiffel Tower", ["Wheelchair Accessible"]),
        ("Disneyland Paris", ["Wheelchair Accessible", "Vegetarian Options", "Family Friendly"]),
        ("Seine River Cruise", ["Wheelchair Accessible", "Family Friendly"]),
        ("Palace of Versailles", ["Wheelchair Accessible", "Guided Tour"]),
        ("Musee d'Orsay", ["Wheelchair Accessible", "Air Conditioned", "Vegetarian Options"]),
        ("Latin Quarter Food Tour", ["Vegetarian Options", "Guided Tour", "Vegan Options"]),
        
        ("London Eye", ["Wheelchair Accessible", "Air Conditioned"]),
        ("Mount Fuji Day Trip", ["Guided Tour"]),
        ("Rijksmuseum", ["Wheelchair Accessible", "Air Conditioned", "Vegetarian Options"]),        
    ]
    mapping_insert_data = []
    for activity_name, feats in activity_features_map:
        act_id = activity_ids.get(activity_name)
        if act_id:
            for feat in feats:
                feat_id = feature_ids.get(feat)
                if feat_id:
                    mapping_insert_data.append((act_id, feat_id))
                    
    conn.executemany(
        "INSERT INTO activity_feature_mapping (activity_id, feature_id) VALUES (?, ?)",
        mapping_insert_data
    )

    transport = [
        ("paris", "london", "Train", 95, 2.25),
        ("london", "paris", "Bus", 35, 7.0)
    ]
    conn.executemany(
        "INSERT INTO transportation (from_city_id, to_city_id, transport_type, price_estimate, duration_hours) VALUES (?,?,?,?,?)",
        [(ids[f], ids[t], ty, p, d) for f,t,ty,p,d in transport]
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

    # city_tags
    tags = [
        ("tel aviv",  "beach"),       ("tel aviv",  "mediterranean"), ("tel aviv",  "nightlife"),
        ("tel aviv",  "foodie"),      ("tel aviv",  "sunny"),         ("tel aviv",  "modern"),
        ("tel aviv",  "city-break"),
        ("paris",     "romantic"),    ("paris",     "city-break"),    ("paris",     "cultural"),
        ("paris",     "luxury"),      ("paris",     "foodie"),        ("paris",     "historic"),
        ("paris",     "fashion"),     ("paris",     "walkable"),
        ("london",    "city-break"),  ("london",    "cultural"),      ("london",    "historic"),
        ("london",    "multicultural"),("london",   "theatre"),       ("london",    "foodie"),
        ("london",    "rainy"),
        ("tokyo",     "modern"),      ("tokyo",     "foodie"),        ("tokyo",     "unique"),
        ("tokyo",     "technology"),  ("tokyo",     "historic"),      ("tokyo",     "safe"),
        ("tokyo",     "city-break"),  ("tokyo",     "nature-nearby"),
        ("new york",  "city-break"),  ("new york",  "shopping"),      ("new york",  "entertainment"),
        ("new york",  "iconic"),      ("new york",  "multicultural"), ("new york",  "foodie"),
        ("new york",  "expensive"),
        ("berlin",    "nightlife"),   ("berlin",    "budget-friendly"),("berlin",   "cultural"),
        ("berlin",    "historic"),    ("berlin",    "alternative"),   ("berlin",    "art"),
        ("berlin",    "city-break"),  ("berlin",    "student-friendly"),
        ("amsterdam", "cycling"),     ("amsterdam", "canals"),        ("amsterdam", "romantic"),
        ("amsterdam", "city-break"),  ("amsterdam", "cultural"),      ("amsterdam", "historic"),
        ("amsterdam", "liberal"),     ("amsterdam", "walkable"),
    ]
    conn.executemany(
        "INSERT OR IGNORE INTO city_tags (city_id, tag) VALUES (?, ?)",
        [(ids[c], t) for c, t in tags],
    )

    # recommended_duration
    durations = [
        ("tel aviv",  3, 5,
         "3 days covers the beach, Jaffa Old City, and the main food markets. "
         "Add 2 more days for day trips to Jerusalem or the Dead Sea."),
        ("paris",     3, 7,
         "3 days for the iconic sights (Eiffel Tower, Louvre, Notre-Dame). "
         "5-7 days lets you explore neighborhoods like Montmartre and take a day trip to Versailles."),
        ("london",    4, 7,
         "4 days covers the top landmarks and a few museums. "
         "7 days gives you time for day trips to Oxford, Stonehenge, or the Cotswolds."),
        ("tokyo",     5, 10,
         "5 days for the main districts (Shibuya, Shinjuku, Asakusa) and a day trip to Mt. Fuji. "
         "8-10 days to also explore Kyoto, Osaka, or Hakone."),
        ("new york",  5, 10,
         "5 days to cover Manhattan highlights and Brooklyn. "
         "7-10 days to explore all five boroughs and do a day trip to the Hamptons or Philadelphia."),
        ("berlin",    3, 5,
         "3 days is plenty for the main historical and cultural sites. "
         "5 days lets you enjoy the nightlife scene and explore surrounding neighborhoods at a relaxed pace."),
        ("amsterdam", 2, 4,
         "2-3 days is ideal for this compact city - you can walk or cycle almost everywhere. "
         "4 days lets you do a day trip to Keukenhof (tulip season) or Delft."),
    ]
    conn.executemany(
        "INSERT INTO recommended_duration (city_id, min_days, max_days, notes) VALUES (?, ?, ?, ?)",
        [(ids[c], mn, mx, n) for c, mn, mx, n in durations],
    )

    print(f"  flights:             {len(flights)}")
    print(f"  hotels:              {len(hotels)}")
    print(f"  activities:          {len(activities)}")
    print(f"  best_time_to_visit:  {len(best_times)}")
    print(f"  average_weather:     {len(weather)}")
    print(f"  city_tags:           {len(tags)}")
    print(f"  recommended_duration:{len(durations)}")


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
        answer = input(
            f"WARNING: '{DB_PATH}' already exists and will be wiped.\n"
            "Are you sure you want to continue? (yes/no): "
        ).strip().lower()
        if answer != "yes":
            print("Aborted. Database was not modified.")
            raise SystemExit(0)
        os.remove(DB_PATH)
    create_travel_db()
