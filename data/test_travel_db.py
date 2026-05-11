import unittest
import sqlite3
from io import StringIO

# ייבוא הפונקציות והסכימה מהקובץ המקורי שלך
# נניח שהקובץ שלך נקרא init_db.py
from init_db import SCHEMA, seed_reference, seed_travel

class TestTravelAgencyDB(unittest.TestCase):

    def setUp(self):
        """הכנת בסיס נתונים נקי בזיכרון לפני כל טסט"""
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row  # מאפשר גישה לפי שמות עמודות
        
        # ---- התוספת הקריטית: הדלקת אכיפת Foreign Keys ----
        self.conn.execute("PRAGMA foreign_keys = ON")
        
        self.conn.executescript(SCHEMA)
        self.mock_seed_data()

    def tearDown(self):
        """סגירת החיבור לאחר כל טסט"""
        self.conn.close()

    def mock_seed_data(self):
        """הזנת נתונים מינימליים כדי לבדוק קשרים (Foreign Keys)"""
        # הכנסת מדינה לדוגמה
        self.conn.execute(
            "INSERT INTO countries (alpha2, alpha3, numeric, name) VALUES (?, ?, ?, ?)",
            ("IL", "ISR", "376", "Israel")
        )
        # הכנסת עיר לדוגמה שמקושרת למדינה
        self.conn.execute(
            "INSERT INTO cities (country_id, name, lat, lng) VALUES (?, ?, ?, ?)",
            (1, "Tel Aviv", 32.0853, 34.7818)
        )
        self.conn.commit()

    def test_country_insertion(self):
        """בדיקה שהמדינות הוכנסו כראוי"""
        cursor = self.conn.execute("SELECT name FROM countries WHERE alpha2 = 'IL'")
        row = cursor.fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["name"], "Israel")

    def test_city_foreign_key(self):
        """בדיקה שהעיר מקושרת למדינה הנכונה"""
        query = """
            SELECT countries.name as country_name 
            FROM cities 
            JOIN countries ON cities.country_id = countries.id 
            WHERE cities.name = 'Tel Aviv'
        """
        row = self.conn.execute(query).fetchone()
        self.assertEqual(row["country_name"], "Israel")

    def test_flight_constraints(self):
        """בדיקה שאי אפשר להכניס טיסה לעיר שלא קיימת (Constraint Check)"""
        # ננסה להכניס טיסה עם ID של עיר שלא קיים (למשל 999)
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute(
                "INSERT INTO flights (origin_city_id, destination_city_id, airline, price, flight_number) VALUES (?, ?, ?, ?, ?)",
                (1, 999, "Test Air", 100, "TA123")
            )

    def test_unique_weather_constraint(self):
        """בדיקה שהגבלת ה-UNIQUE על עיר ועונה עובדת"""
        self.conn.execute(
            "INSERT INTO average_weather (city_id, season, temperature) VALUES (?, ?, ?)",
            (1, "Summer", "30C")
        )
        # ניסיון להכניס את אותה עיר ואותה עונה שוב
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute(
                "INSERT INTO average_weather (city_id, season, temperature) VALUES (?, ?, ?)",
                (1, "Summer", "35C")
            )

    def test_cascade_or_deletion(self):
        """בדיקה מה קורה כשמוחקים מדינה (לוודא שלמות נתונים)"""
        # הערה: ב-SQLite צריך להפעיל אכיפת FK ידנית בחיבור
        self.conn.execute("PRAGMA foreign_keys = ON")
        
        # מחיקת המדינה אמורה להיכשל או להתנהג לפי ה-REFERENCES שהגדרת
        # כיוון שלא הגדרת ON DELETE CASCADE, ה-Foreign Key ימנע מחיקה
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute("DELETE FROM countries WHERE id = 1")

if __name__ == "__main__":
    unittest.main()