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
        Searches for all possible flight routes (direct or up to 2 stops) between origin and destination.
        Handles time validation, budget constraints, and avoids infinite loops.
        """
        origin_clean = origin.strip().lower()
        dest_clean = destination.strip().lower()

        # ---------------------------------------------------------
        # Step 1: Resolve City IDs from City Names
        # ---------------------------------------------------------
        cities_info = self._query(
            "SELECT id, LOWER(name) as name FROM cities WHERE LOWER(name) IN (?, ?)",
            (origin_clean, dest_clean)
        )
        
        origin_id = None
        dest_id = None
        for row in cities_info:
            if row['name'] == origin_clean:
                origin_id = row['id']
            elif row['name'] == dest_clean:
                dest_id = row['id']
                
        if not origin_id or not dest_id:
            return [{"message": f"Could not find geographic data for {origin} or {destination}."}]

        # ---------------------------------------------------------
        # Step 2: Execute the Recursive CTE
        # ---------------------------------------------------------
        # If no budget is provided, use an arbitrarily large number to bypass the constraint
        budget_limit = max_budget if max_budget is not None else 9999999.0

        sql = """
        WITH RECURSIVE
        route_builder(
            current_dest_id, path_cities, path_flight_ids, first_departure, last_arrival, total_price, stops
        ) AS (
            -- Base Case: Flights departing directly from the origin city
            SELECT 
                destination_city_id,
                ',' || CAST(origin_city_id AS TEXT) || ',' || CAST(destination_city_id AS TEXT) || ',',
                CAST(id AS TEXT),
                departure_time,
                arrival_time,
                price,
                0
            FROM flights
            WHERE origin_city_id = ?
              AND price <= ?

            UNION ALL

            -- Recursive Step: Connecting flights
            SELECT 
                f.destination_city_id,
                rb.path_cities || CAST(f.destination_city_id AS TEXT) || ',',
                rb.path_flight_ids || ' -> ' || CAST(f.id AS TEXT),
                rb.first_departure,
                f.arrival_time,
                rb.total_price + f.price,
                rb.stops + 1
            FROM route_builder rb
            JOIN flights f ON rb.current_dest_id = f.origin_city_id
            WHERE 
                rb.stops < 2
                AND f.departure_time >= datetime(rb.last_arrival, '+1 hour')
                AND rb.path_cities NOT LIKE '%,' || CAST(f.destination_city_id AS TEXT) || ',%'
                AND (rb.total_price + f.price) <= ?
        )
        SELECT 
            path_flight_ids AS flight_sequence,
            stops,
            total_price,
            first_departure,
            last_arrival,
            ROUND((strftime('%s', last_arrival) - strftime('%s', first_departure)) / 3600.0, 2) AS total_duration_hours
        FROM route_builder
        WHERE current_dest_id = ?
        ORDER BY 
            total_price ASC,
            stops ASC,
            total_duration_hours ASC
        LIMIT 10;
        """
        
        # Parameters match the ? marks in the SQL: origin_id, budget, budget, dest_id
        params = (origin_id, budget_limit, budget_limit, dest_id)
        rows = self._query(sql, params)
        
        if not rows:
            msg = f"No valid flight routes found from {origin} to {destination}."
            if max_budget:
                msg += f" under budget of ${max_budget}."
            return [{"message": msg}]
            
        return rows