import os
import sqlite3
from .flight_queries import SQLiteFlightQueriesMixin
from .hotel_queries import SQLiteHotelQueriesMixin
from .activity_queries import SQLiteActivityQueriesMixin
from .best_time_queries import SQLiteBestTimeQueriesMixin
from .weather_queries import SQLiteWeatherQueriesMixin

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "travel_agency.db")


class SQLiteDataProvider(
    SQLiteFlightQueriesMixin,
    SQLiteHotelQueriesMixin,
    SQLiteActivityQueriesMixin,
    SQLiteBestTimeQueriesMixin,
    SQLiteWeatherQueriesMixin,
):
    """SQLite-backed data provider. Query logic lives in topic mixins."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = os.path.abspath(db_path)

    def _query(self, sql: str, params: tuple = ()) -> list[dict]:
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(
                f"Database not found at {self.db_path}. Run data/init_db.py first."
            )
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]
        finally:
            conn.close()
