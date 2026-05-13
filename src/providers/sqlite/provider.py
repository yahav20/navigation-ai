"""SQLite-backed data provider."""
import sqlite3
from pathlib import Path

from providers.sqlite.activity_queries import SQLiteActivityQueriesMixin
from providers.sqlite.best_time_queries import SQLiteBestTimeQueriesMixin
from providers.sqlite.flight_queries import SQLiteFlightQueriesMixin
from providers.sqlite.hotel_queries import SQLiteHotelQueriesMixin
from providers.sqlite.rec_queries import RecommendationQueriesMixin
from providers.sqlite.weather_queries import SQLiteWeatherQueriesMixin

DB_PATH = str(Path(__file__).parent / ".." / ".." / ".." / "data" / "travel_agency.db")


class SQLiteDataProvider(
    SQLiteFlightQueriesMixin,
    SQLiteHotelQueriesMixin,
    SQLiteActivityQueriesMixin,
    SQLiteBestTimeQueriesMixin,
    SQLiteWeatherQueriesMixin,
    RecommendationQueriesMixin,
):
    """SQLite-backed data provider. Query logic lives in topic mixins."""

    def __init__(self, db_path: str = DB_PATH) -> None:
        """Initialize the provider with the database path."""
        self.db_path = str(Path(db_path).resolve())

    def _query(self, sql: str, params: tuple = ()) -> list[dict]:
        if not Path(self.db_path).exists():
            msg = f"Database not found at {self.db_path}. Run data/init_db.py first."
            raise FileNotFoundError(
                msg,
            )
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]
        finally:
            conn.close()
