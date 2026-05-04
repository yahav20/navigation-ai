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
    
    def fetch_connecting_flights(self, origin: str, destination: str, max_budget: float = None) -> list:
        """
        Find connecting flights (1 or 2 stops).
        If no results to the exact destination city, fallback to searching for connecting flights
        to other cities in the same country (just like fetch_flights).
        """
        origin_clean = origin.strip().lower()
        dest_clean = destination.strip().lower()

        # Dynamic budget conditions
        budget_cond_1 = " AND (f1.price + f2.price) <= ?" if max_budget else ""
        budget_cond_2 = " AND (f1.price + f2.price + f3.price) <= ?" if max_budget else ""

        # Parameters for exact search
        params_exact_1 = (origin_clean, dest_clean, max_budget) if max_budget else (origin_clean, dest_clean)
        params_exact_2 = (origin_clean, dest_clean, max_budget) if max_budget else (origin_clean, dest_clean)

        # ---------------------------------------------------------
        # Step 1: Search for connecting flights to the exact destination city
        # ---------------------------------------------------------
        query_1_stop_exact = f"""
            SELECT oc.name AS origin_city, l1.name AS layover1, dc.name AS destination_city, co.name AS destination_country,
                   f1.flight_number AS leg1_flight, f2.flight_number AS leg2_flight,
                   (f1.price + f2.price) AS total_price, 1 AS stops
            FROM flights f1
            JOIN flights f2 ON f1.destination_city_id = f2.origin_city_id
            JOIN cities oc  ON f1.origin_city_id      = oc.id
            JOIN cities l1  ON f1.destination_city_id = l1.id
            JOIN cities dc  ON f2.destination_city_id = dc.id
            JOIN countries co ON dc.country_id        = co.id
            WHERE LOWER(oc.name) = ? AND LOWER(dc.name) = ? {budget_cond_1}
        """

        query_2_stops_exact = f"""
            SELECT oc.name AS origin_city, l1.name AS layover1, l2.name AS layover2, dc.name AS destination_city, co.name AS destination_country,
                   f1.flight_number AS leg1_flight, f2.flight_number AS leg2_flight, f3.flight_number AS leg3_flight,
                   (f1.price + f2.price + f3.price) AS total_price, 2 AS stops
            FROM flights f1
            JOIN flights f2 ON f1.destination_city_id = f2.origin_city_id
            JOIN flights f3 ON f2.destination_city_id = f3.origin_city_id
            JOIN cities oc  ON f1.origin_city_id      = oc.id
            JOIN cities l1  ON f1.destination_city_id = l1.id
            JOIN cities l2  ON f2.destination_city_id = l2.id
            JOIN cities dc  ON f3.destination_city_id = dc.id
            JOIN countries co ON dc.country_id        = co.id
            WHERE LOWER(oc.name) = ? AND LOWER(dc.name) = ? {budget_cond_2}
        """

        exact_results = []
        rows_1_stop = self._query(query_1_stop_exact, params_exact_1)
        if rows_1_stop: exact_results.extend(rows_1_stop)

        rows_2_stops = self._query(query_2_stops_exact, params_exact_2)
        if rows_2_stops: exact_results.extend(rows_2_stops)

        # If exact results were found, sort and return
        if exact_results:
            exact_results.sort(key=lambda x: x["total_price"])
            return exact_results[:5]

        # ---------------------------------------------------------
        # Step 2: Fallback (search within the same country)
        # ---------------------------------------------------------
        # Find the country ID of the destination city
        country_rows = self._query(
            "SELECT co.name AS country_name, co.id AS country_id FROM cities c JOIN countries co ON c.country_id = co.id WHERE LOWER(c.name) = ? LIMIT 1",
            (dest_clean,)
        )

        if not country_rows:
            msg = f"No connecting flights found from {origin} to {destination}"
            msg += f" under ${max_budget}." if max_budget else "."
            return [{"message": msg}]

        country_name = country_rows[0]["country_name"]
        country_id   = country_rows[0]["country_id"]

        # Parameters for alternative search
        params_alt_1 = (origin_clean, country_id, dest_clean, max_budget) if max_budget else (origin_clean, country_id, dest_clean)
        params_alt_2 = (origin_clean, country_id, dest_clean, max_budget) if max_budget else (origin_clean, country_id, dest_clean)

        query_1_stop_alt = f"""
            SELECT oc.name AS origin_city, l1.name AS layover1, dc.name AS destination_city, co.name AS destination_country,
                   f1.flight_number AS leg1_flight, f2.flight_number AS leg2_flight,
                   (f1.price + f2.price) AS total_price, 1 AS stops
            FROM flights f1
            JOIN flights f2 ON f1.destination_city_id = f2.origin_city_id
            JOIN cities oc  ON f1.origin_city_id      = oc.id
            JOIN cities l1  ON f1.destination_city_id = l1.id
            JOIN cities dc  ON f2.destination_city_id = dc.id
            JOIN countries co ON dc.country_id        = co.id
            WHERE LOWER(oc.name) = ? AND dc.country_id = ? AND LOWER(dc.name) != ? {budget_cond_1}
        """

        query_2_stops_alt = f"""
            SELECT oc.name AS origin_city, l1.name AS layover1, l2.name AS layover2, dc.name AS destination_city, co.name AS destination_country,
                   f1.flight_number AS leg1_flight, f2.flight_number AS leg2_flight, f3.flight_number AS leg3_flight,
                   (f1.price + f2.price + f3.price) AS total_price, 2 AS stops
            FROM flights f1
            JOIN flights f2 ON f1.destination_city_id = f2.origin_city_id
            JOIN flights f3 ON f2.destination_city_id = f3.origin_city_id
            JOIN cities oc  ON f1.origin_city_id      = oc.id
            JOIN cities l1  ON f1.destination_city_id = l1.id
            JOIN cities l2  ON f2.destination_city_id = l2.id
            JOIN cities dc  ON f3.destination_city_id = dc.id
            JOIN countries co ON dc.country_id        = co.id
            WHERE LOWER(oc.name) = ? AND dc.country_id = ? AND LOWER(dc.name) != ? {budget_cond_2}
        """

        alt_results = []
        rows_1_stop_alt = self._query(query_1_stop_alt, params_alt_1)
        if rows_1_stop_alt: alt_results.extend(rows_1_stop_alt)

        rows_2_stops_alt = self._query(query_2_stops_alt, params_alt_2)
        if rows_2_stops_alt: alt_results.extend(rows_2_stops_alt)

        if not alt_results:
            msg = f"No connecting flights found from {origin} to {destination} or elsewhere in {country_name}"
            msg += f" under ${max_budget}." if max_budget else "."
            return [{"message": msg}]

        alt_results.sort(key=lambda x: x["total_price"])
        
        # Wrap the results in the same format as the direct flights function
        return [{
            "message": f"No connecting flights to {destination}, but here are options to other cities in {country_name}:",
            "alternatives": alt_results[:5]
        }]