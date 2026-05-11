import os
import sqlite3
import unittest

DB_PATH = os.path.join(os.path.dirname(__file__), "../data/travel_agency.db")

class TestTravelAgencyDB(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
       
        assert os.path.exists(DB_PATH), "DB not found. Run create_db() first."
        cls.conn = sqlite3.connect(DB_PATH)
        cls.conn.row_factory = sqlite3.Row

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()

    # -------------------------
    # SCHEMA TESTS
    # -------------------------
    def test_tables_exist(self):
        tables = {row[0] for row in self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}

        expected = {
            "countries",
            "cities",
            "flights",
            "hotels",
            "activities",
            "best_time_to_visit",
            "average_weather",
            "transportation",
        }

        self.assertTrue(expected.issubset(tables), f"Missing tables: {expected - tables}")

    # -------------------------
    # DATA EXISTENCE
    # -------------------------
    def test_cities_exist(self):
        count = self.conn.execute("SELECT COUNT(*) FROM cities").fetchone()[0]
        self.assertGreater(count, 0, "No cities found in the database.")

    def test_flights_exist(self):
        count = self.conn.execute("SELECT COUNT(*) FROM flights").fetchone()[0]
        self.assertGreater(count, 0, "No flights found in the database.")

    def test_hotels_exist(self):
        count = self.conn.execute("SELECT COUNT(*) FROM hotels").fetchone()[0]
        self.assertGreater(count, 0, "No hotels found in the database.")

    def test_transportation_exist(self):
        count = self.conn.execute("SELECT COUNT(*) FROM transportation").fetchone()[0]
        self.assertGreater(count, 0, "No transportation data found in the database.")

    # -------------------------
    # FOREIGN KEY / JOIN TESTS
    # -------------------------
    def test_flight_join_validity(self):
        row = self.conn.execute("""
            SELECT
                f.flight_number,
                c1.name AS origin,
                c2.name AS destination,
                f.airline,
                f.price
            FROM flights f
            JOIN cities c1 ON f.origin_city_id = c1.id
            JOIN cities c2 ON f.destination_city_id = c2.id
            LIMIT 1
        """).fetchone()

        self.assertIsNotNone(row)
        self.assertIsNotNone(row["flight_number"])
        self.assertIsNotNone(row["origin"])
        self.assertIsNotNone(row["destination"])

    def test_tel_aviv_has_flights(self):
        row = self.conn.execute("""
            SELECT COUNT(*) as cnt
            FROM flights f
            JOIN cities c ON f.origin_city_id = c.id
            WHERE LOWER(c.name) = 'tel aviv'
        """).fetchone()

        self.assertGreater(row["cnt"], 0, "No outgoing flights found for Tel Aviv.")

    # -------------------------
    # HOTELS TESTS
    # -------------------------
    def test_hotels_schema_fields(self):
        row = self.conn.execute("""
            SELECT hotel_type, min_age, distance_from_center_km
            FROM hotels
            LIMIT 1
        """).fetchone()

        self.assertIsNotNone(row["hotel_type"])
        self.assertIsNotNone(row["min_age"])
        self.assertIsNotNone(row["distance_from_center_km"])

    # -------------------------
    # ACTIVITIES TESTS
    # -------------------------
    def test_activities_min_age(self):
        row = self.conn.execute("""
            SELECT min_age
            FROM activities
            LIMIT 1
        """).fetchone()

        self.assertIsNotNone(row)
        self.assertIsNotNone(row["min_age"])

    # -------------------------
    # TRANSPORTATION TESTS
    # -------------------------
    def test_transportation_data_valid(self):
        row = self.conn.execute("""
            SELECT transport_type, price_estimate, duration_hours
            FROM transportation
            LIMIT 1
        """).fetchone()

        self.assertIsNotNone(row["transport_type"])
        self.assertIsNotNone(row["price_estimate"])
        self.assertIsNotNone(row["duration_hours"])

    # -------------------------
    # FLIGHT FIELD VALIDATION
    # -------------------------
    def test_flights_new_fields(self):
        row = self.conn.execute("""
            SELECT departure_time, arrival_time, duration_minutes
            FROM flights
            LIMIT 1
        """).fetchone()

        self.assertIsNotNone(row["departure_time"])
        self.assertIsNotNone(row["arrival_time"])
        self.assertGreater(row["duration_minutes"], 0)

    # -------------------------
    # INTEGRATION TEST
    # -------------------------
    def test_full_flight_query(self):
        rows = self.conn.execute("""
            SELECT
                c1.name AS origin,
                c2.name AS destination,
                f.airline,
                f.price,
                f.availability
            FROM flights f
            JOIN cities c1 ON f.origin_city_id = c1.id
            JOIN cities c2 ON f.destination_city_id = c2.id
            WHERE f.price > 0
            LIMIT 5
        """).fetchall()

        self.assertGreater(len(rows), 0)

        for r in rows:
            self.assertIsNotNone(r["origin"])
            self.assertIsNotNone(r["destination"])
            self.assertGreater(r["price"], 0)

if __name__ == "__main__":
    unittest.main()