import sqlite3
import json
import random
from datetime import datetime, timedelta

def init_atlas_db(db_path="atlas_travel.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("--- Creating Tables ---")

    # 1. Countries
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS countries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        alpha2 CHAR(2) UNIQUE,
        alpha3 CHAR(3) UNIQUE,
        numeric TEXT,
        name TEXT,
        region TEXT,
        subregion TEXT,
        currency CHAR(3),
        currency_symbol TEXT,
        exchange_rate_to_usd REAL,
        languages TEXT,
        visa_requirements TEXT
    )""")

    # 2. Cities
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS cities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        country_id INTEGER REFERENCES countries(id),
        name TEXT,
        lat REAL,
        lng REAL,
        timezone TEXT,
        airport_code CHAR(3),
        secondary_airports TEXT,
        cost_level TEXT,
        safety_score INTEGER,
        description TEXT,
        image_url TEXT
    )""")

    # 3. Flights
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS flights (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        origin_city_id INTEGER REFERENCES cities(id),
        destination_city_id INTEGER REFERENCES cities(id),
        airline TEXT,
        flight_number TEXT,
        aircraft_type TEXT,
        price REAL,
        price_currency CHAR(3),
        cabin_class TEXT,
        fare_basis TEXT,
        price_breakdown TEXT,
        baggage_allowance TEXT,
        seat_selection_included BOOLEAN,
        meal_included BOOLEAN,
        wifi_available BOOLEAN,
        departure_timezone TEXT,
        arrival_timezone TEXT,
        scheduled_departure_time TEXT,
        scheduled_arrival_time TEXT,
        actual_departure_time TEXT,
        actual_arrival_time TEXT,
        delay_minutes INTEGER DEFAULT 0,
        departure_terminal TEXT,
        arrival_terminal TEXT,
        gate TEXT,
        boarding_time TEXT,
        checkin_open_time TEXT,
        codeshare TEXT,
        seats_available INTEGER,
        booking_url TEXT,
        status TEXT,
        departure_time TEXT, -- Helper for recursive queries
        arrival_time TEXT    -- Helper for recursive queries
    )""")

    # 4. Hotels
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS hotels (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        city_id INTEGER REFERENCES cities(id),
        name TEXT,
        price_per_night REAL,
        price_currency CHAR(3),
        stars INTEGER,
        location_vector BLOB,
        taxes_included BOOLEAN,
        cancellation_policy TEXT,
        room_types TEXT,
        max_occupancy INTEGER,
        amenities TEXT,
        rating REAL,
        reviews_count INTEGER,
        check_in_time TEXT,
        check_out_time TEXT,
        late_check_in_available BOOLEAN,
        address TEXT,
        lat REAL,
        lng REAL,
        images TEXT,
        booking_url TEXT,
        availability_status TEXT
    )""")

    # 5. Users
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        email TEXT UNIQUE,
        full_name TEXT,
        phone TEXT,
        travel_profile TEXT,
        ai_memory TEXT,
        preferred_currency CHAR(3),
        preferred_language TEXT,
        home_airport CHAR(3),
        seat_preference TEXT,
        meal_preference TEXT,
        hotel_preferences TEXT,
        loyalty_programs TEXT,
        risk_tolerance TEXT,
        notification_preferences TEXT,
        created_at TEXT,
        last_active_at TEXT
    )""")

    # 6. Activities
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS activities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        city_id INTEGER REFERENCES cities(id),
        name TEXT,
        category TEXT,
        price REAL,
        price_currency CHAR(3),
        duration_min INTEGER,
        operating_hours TEXT,
        rating REAL,
        reviews_count INTEGER,
        location_lat REAL,
        location_lng REAL,
        booking_required BOOLEAN,
        tickets_available INTEGER,
        age_restrictions TEXT,
        indoor_outdoor TEXT,
        image_url TEXT,
        tags TEXT
    )""")

    # 7. Ground Transport
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ground_transport (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        origin_id INTEGER,
        destination_id INTEGER,
        transport_type TEXT,
        provider_name TEXT,
        price REAL,
        price_currency CHAR(3),
        departure_time TEXT,
        arrival_time TEXT,
        duration_min INTEGER,
        pickup_location_lat REAL,
        pickup_location_lng REAL,
        dropoff_location_lat REAL,
        dropoff_location_lng REAL,
        luggage_capacity INTEGER,
        booking_url TEXT
    )""")

    print("--- Seeding Data ---")

    # Seed Countries
    countries = [
        (1, 'IL', 'ISR', '376', 'Israel', 'Asia', 'Western Asia', 'ILS', '₪', 3.7, 'Hebrew, Arabic', '{"visa_free": ["EU", "USA"]}') ,
        (2, 'FR', 'FRA', '250', 'France', 'Europe', 'Western Europe', 'EUR', '€', 0.92, 'French', '{"schengen": true}') ,
        (3, 'US', 'USA', '840', 'United States', 'Americas', 'Northern America', 'USD', '$', 1.0, 'English', '{"esta_required": true}')
    ]
    cursor.executemany("INSERT INTO countries VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", countries)

    # Seed Cities
    cities = [
        (1, 1, 'Tel Aviv', 32.0853, 34.7818, 'Asia/Jerusalem', 'TLV', '[]', 'expensive', 85, 'The city that never sleeps', 'http://image.tlv'),
        (2, 2, 'Paris', 48.8566, 2.3522, 'Europe/Paris', 'CDG', '["ORY", "BVA"]', 'expensive', 80, 'City of lights', 'http://image.paris'),
        (3, 2, 'Nice', 43.7102, 7.2620, 'Europe/Paris', 'NCE', '[]', 'moderate', 88, 'Cote dAzur gem', 'http://image.nice')
    ]
    cursor.executemany("INSERT INTO cities VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", cities)

    # Seed Flights (Logical connections for testing)
    now = datetime(2026, 6, 1, 10, 0)
    flights = [
        # TLV -> CDG (Direct)
        (1, 1, 2, 'El Al', 'LY323', 'Boeing 787', 450.0, 'USD', 'economy', 'flex', 
         json.dumps({"base": 380, "taxes": 70}), json.dumps({"checked": "23kg"}), 1, 1, 1,
         'Asia/Jerusalem', 'Europe/Paris', 
         (now).strftime('%Y-%m-%d %H:%M:%S'), (now + timedelta(hours=4.5)).strftime('%Y-%m-%d %H:%M:%S'),
         None, None, 0, 'T3', 'T2E', 'B12', (now - timedelta(minutes=40)).strftime('%H:%M'), 
         (now - timedelta(hours=3)).strftime('%H:%M'), '[]', 45, 'http://book.elal', 'scheduled',
         (now).strftime('%Y-%m-%d %H:%M:%S'), (now + timedelta(hours=4.5)).strftime('%Y-%m-%d %H:%M:%S')),
        
        # TLV -> NCE (Direct)
        (2, 1, 3, 'Air France', 'AF120', 'Airbus A321', 320.0, 'USD', 'economy', 'non-refundable', 
         json.dumps({"base": 280, "taxes": 40}), json.dumps({"checked": "0kg"}), 0, 0, 1,
         'Asia/Jerusalem', 'Europe/Paris', 
         (now + timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S'), (now + timedelta(hours=6)).strftime('%Y-%m-%d %H:%M:%S'),
         None, None, 0, 'T3', 'T1', 'A05', (now + timedelta(hours=1.5)).strftime('%H:%M'), 
         (now - timedelta(hours=1)).strftime('%H:%M'), '[]', 12, 'http://book.af', 'scheduled',
         (now + timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S'), (now + timedelta(hours=6)).strftime('%Y-%m-%d %H:%M:%S')),

        # NCE -> CDG (Connection leg)
        (3, 3, 2, 'Air France', 'AF200', 'Airbus A320', 120.0, 'USD', 'economy', 'flex', 
         json.dumps({"base": 100, "taxes": 20}), json.dumps({"checked": "23kg"}), 1, 0, 0,
         'Europe/Paris', 'Europe/Paris', 
         (now + timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S'), (now + timedelta(hours=9.5)).strftime('%Y-%m-%d %H:%M:%S'),
         None, None, 0, 'T1', 'T2F', 'F10', (now + timedelta(hours=7.5)).strftime('%H:%M'), 
         (now + timedelta(hours=6)).strftime('%H:%M'), '[]', 80, 'http://book.af', 'scheduled',
         (now + timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S'), (now + timedelta(hours=9.5)).strftime('%Y-%m-%d %H:%M:%S'))
    ]
    cursor.executemany("INSERT INTO flights VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", flights)

    # Seed Hotels (with mock vector)
    mock_vector = json.dumps([random.uniform(-1, 1) for _ in range(768)])
    hotels = [
        (1, 2, 'Hotel Le Meurice', 850.0, 'EUR', 5, mock_vector, 0, 
         json.dumps({"free_until": "24h_before"}), json.dumps(["Deluxe", "Suite"]), 2, 
         json.dumps(["Spa", "Michelin Star Restaurant", "WiFi"]), 9.5, 1240, 
         "15:00", "12:00", 1, "228 Rue de Rivoli, Paris", 48.8650, 2.3280, 
         json.dumps(["img1.jpg"]), "http://booking.meurice", "available")
    ]
    cursor.executemany("INSERT INTO hotels VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", hotels)

    # Seed User
    user = (
        "user_123", "noa@example.com", "Noa Amram", "+972500000000",
        json.dumps({"style": "luxury", "budget": "high"}), 
        json.dumps({"recent_searches": ["Paris", "NYC"]}),
        "USD", "Hebrew", "TLV", "window", "kosher", 
        json.dumps({"min_stars": 4}), json.dumps({"elal_matmid": "Gold"}), 
        "medium", json.dumps({"email": True}), "2025-10-01", "2026-05-06"
    )
    cursor.execute("INSERT INTO users VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", user)

    conn.commit()
    print(f"--- Success! Database initialized at {db_path} ---")
    conn.close()

if __name__ == "__main__":
    init_atlas_db()