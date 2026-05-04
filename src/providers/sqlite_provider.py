import os
import sqlite3
from .base import BaseDataProvider

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "travel_agency.db")


class SQLiteDataProvider(BaseDataProvider):
    """SQLite-backed data provider for travel data."""

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

    def fetch_flights(self, origin: str, destination: str) -> list:
        """
        Search by destination city first.
        If no results, fall back to destination country and suggest
        other cities in that same country.
        """
        rows = self._query(
            """SELECT origin_city, destination_city, destination_country,
                      airline, price, flight_number, availability
               FROM flights
               WHERE LOWER(origin_city) = ? AND LOWER(destination_city) = ?""",
            (origin.strip().lower(), destination.strip().lower()),
        )

        if rows:
            return rows

        # Fallback: find the country for the requested destination
        country_rows = self._query(
            "SELECT destination_country FROM flights WHERE LOWER(destination_city) = ? LIMIT 1",
            (destination.strip().lower(),),
        )

        if not country_rows:
            return [{"message": f"No flights found from {origin} to {destination}."}]

        country = country_rows[0]["destination_country"]

        alt_rows = self._query(
            """SELECT origin_city, destination_city, destination_country,
                      airline, price, flight_number, availability
               FROM flights
               WHERE LOWER(origin_city) = ? AND LOWER(destination_country) = ?
                 AND LOWER(destination_city) != ?""",
            (origin.strip().lower(), country.lower(), destination.strip().lower()),
        )

        if not alt_rows:
            return [{"message": f"No flights found from {origin} to {destination} or elsewhere in {country}."}]

        return [{
            "message": f"No direct flights to {destination}, but here are flights to other cities in {country}:",
            "alternatives": alt_rows,
        }]

    def fetch_hotels(self, city: str, max_price: int = None) -> list:
        sql = "SELECT name, price_per_night, stars FROM hotels WHERE LOWER(city) = ?"
        params = [city.strip().lower()]
        if max_price is not None:
            sql += " AND price_per_night <= ?"
            params.append(max_price)
        rows = self._query(sql, tuple(params))
        if not rows:
            return [{"message": f"No available hotels in {city}."}]
        return rows

    def fetch_activities(self, city: str) -> list:
        rows = self._query(
            "SELECT name, category, price FROM activities WHERE LOWER(city) = ?",
            (city.strip().lower(),),
        )
        if not rows:
            return [{"message": f"No available activities found in {city}."}]
        return rows

    def get_best_time_to_visit(self, city: str) -> dict:
        rows = self._query(
            "SELECT months, reason FROM best_time_to_visit WHERE LOWER(city) = ?",
            (city.strip().lower(),),
        )
        if not rows:
            return {"message": f"No recommendations found for {city}."}
        row = rows[0]
        return {
            "months": [m.strip() for m in row["months"].split(",")],
            "reason": row["reason"],
        }

    def get_average_weather(self, city: str, season: str) -> dict:
        rows = self._query(
            "SELECT season, temperature FROM average_weather WHERE LOWER(city) = ? AND LOWER(season) = ?",
            (city.strip().lower(), season.strip().lower()),
        )
        if not rows:
            return {"message": f"No weather data found for {season} in {city}."}
        return {"season": rows[0]["season"], "temperature": rows[0]["temperature"]}
