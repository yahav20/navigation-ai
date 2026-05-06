import os
import sqlite3
from .base import BaseDataProvider

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "travel_agency.db")


class SQLiteDataProvider(BaseDataProvider):
    """SQLite-backed data provider using the normalized FK schema."""

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
        Search by destination city name first.
        If no results, fall back to destination country and suggest
        other cities in that same country.
        """
        rows = self._query(
            """SELECT oc.name  AS origin_city,
                      dc.name  AS destination_city,
                      co.name  AS destination_country,
                      f.airline, f.price, f.flight_number, f.availability
               FROM flights f
               JOIN cities oc  ON f.origin_city_id      = oc.id
               JOIN cities dc  ON f.destination_city_id = dc.id
               JOIN countries co ON dc.country_id        = co.id
              WHERE LOWER(oc.name) = ? AND LOWER(dc.name) = ?""",
            (origin.strip().lower(), destination.strip().lower()),
        )

        if rows:
            return rows

        # Fallback: find the country of the requested destination city
        country_rows = self._query(
            """SELECT co.name AS country_name, co.id AS country_id
               FROM cities c
               JOIN countries co ON c.country_id = co.id
              WHERE LOWER(c.name) = ?
              LIMIT 1""",
            (destination.strip().lower(),),
        )

        if not country_rows:
            return [{"message": f"No flights found from {origin} to {destination}."}]

        country_name = country_rows[0]["country_name"]
        country_id   = country_rows[0]["country_id"]

        alt_rows = self._query(
            """SELECT oc.name  AS origin_city,
                      dc.name  AS destination_city,
                      co.name  AS destination_country,
                      f.airline, f.price, f.flight_number, f.availability
               FROM flights f
               JOIN cities oc  ON f.origin_city_id      = oc.id
               JOIN cities dc  ON f.destination_city_id = dc.id
               JOIN countries co ON dc.country_id        = co.id
              WHERE LOWER(oc.name) = ?
                AND dc.country_id  = ?
                AND LOWER(dc.name) != ?""",
            (origin.strip().lower(), country_id, destination.strip().lower()),
        )

        if not alt_rows:
            return [{"message": f"No flights found from {origin} to {destination} or elsewhere in {country_name}."}]

        return [{
            "message": f"No direct flights to {destination}, but here are flights to other cities in {country_name}:",
            "alternatives": alt_rows,
        }]

    def fetch_hotels(self, city: str, max_price: int = None) -> list:
        sql = """SELECT h.name, h.price_per_night, h.stars
                   FROM hotels h
                   JOIN cities c ON h.city_id = c.id
                  WHERE LOWER(c.name) = ?"""
        params = [city.strip().lower()]
        if max_price is not None:
            sql += " AND h.price_per_night <= ?"
            params.append(max_price)
        rows = self._query(sql, tuple(params))
        if not rows:
            return [{"message": f"No available hotels in {city}."}]
        return rows

    def fetch_activities(self, city: str) -> list:
        rows = self._query(
            """SELECT a.name, a.category, a.price
               FROM activities a
               JOIN cities c ON a.city_id = c.id
              WHERE LOWER(c.name) = ?""",
            (city.strip().lower(),),
        )
        if not rows:
            return [{"message": f"No available activities found in {city}."}]
        return rows

    def get_best_time_to_visit(self, city: str) -> dict:
        rows = self._query(
            """SELECT b.months, b.reason
               FROM best_time_to_visit b
               JOIN cities c ON b.city_id = c.id
              WHERE LOWER(c.name) = ?""",
            (city.strip().lower(),),
        )
        if not rows:
            return {"message": f"No recommendations found for {city}."}
        return {
            "months": [m.strip() for m in rows[0]["months"].split(",")],
            "reason": rows[0]["reason"],
        }

    def get_hotel_dimensions(self, city: str) -> dict:
        rows = self._query(
            """SELECT DISTINCT h.stars,
                      MIN(h.price_per_night) AS price_min,
                      MAX(h.price_per_night) AS price_max
                 FROM hotels h
                 JOIN cities c ON h.city_id = c.id
                WHERE LOWER(c.name) = ?
                GROUP BY h.stars""",
            (city.strip().lower(),),
        )
        if not rows:
            return {"available": False}
        stars = sorted({r["stars"] for r in rows if r["stars"] is not None})
        all_prices = [r["price_min"] for r in rows] + [r["price_max"] for r in rows]
        all_prices = [p for p in all_prices if p is not None]
        return {
            "stars_available": stars,
            "price_min": min(all_prices) if all_prices else None,
            "price_max": max(all_prices) if all_prices else None,
        }

    def get_flight_dimensions(self, origin: str, destination: str) -> dict:
        rows = self._query(
            """SELECT DISTINCT f.airline,
                      MIN(f.price) AS price_min,
                      MAX(f.price) AS price_max
                 FROM flights f
                 JOIN cities oc ON f.origin_city_id      = oc.id
                 JOIN cities dc ON f.destination_city_id = dc.id
                WHERE LOWER(oc.name) = ? AND LOWER(dc.name) = ?
                GROUP BY f.airline""",
            (origin.strip().lower(), destination.strip().lower()),
        )
        if not rows:
            return {"available": False}
        airlines = sorted({r["airline"] for r in rows if r["airline"]})
        all_prices = [r["price_min"] for r in rows] + [r["price_max"] for r in rows]
        all_prices = [p for p in all_prices if p is not None]
        return {
            "airlines_available": airlines,
            "price_min": min(all_prices) if all_prices else None,
            "price_max": max(all_prices) if all_prices else None,
        }

    def get_cities_in_country(self, country_name: str, origin: str = None) -> list:
        if origin:
            rows = self._query(
                """SELECT DISTINCT dc.name
                     FROM flights f
                     JOIN cities oc ON f.origin_city_id      = oc.id
                     JOIN cities dc ON f.destination_city_id = dc.id
                     JOIN countries co ON dc.country_id      = co.id
                    WHERE LOWER(co.name) = ? AND LOWER(oc.name) = ?""",
                (country_name.strip().lower(), origin.strip().lower()),
            )
        else:
            rows = self._query(
                """SELECT DISTINCT dc.name
                     FROM flights f
                     JOIN cities dc ON f.destination_city_id = dc.id
                     JOIN countries co ON dc.country_id      = co.id
                    WHERE LOWER(co.name) = ?""",
                (country_name.strip().lower(),),
            )
        return [r["name"] for r in rows]

    def get_average_weather(self, city: str, season: str) -> dict:
        rows = self._query(
            """SELECT w.season, w.temperature
               FROM average_weather w
               JOIN cities c ON w.city_id = c.id
              WHERE LOWER(c.name) = ? AND LOWER(w.season) = ?""",
            (city.strip().lower(), season.strip().lower()),
        )
        if not rows:
            return {"message": f"No weather data found for {season} in {city}."}
        return {"season": rows[0]["season"], "temperature": rows[0]["temperature"]}
