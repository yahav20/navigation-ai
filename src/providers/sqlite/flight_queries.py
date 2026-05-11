"""SQLite queries for flights."""


class SQLiteFlightQueriesMixin:
    """Query flight rows from the SQLite database."""

    def fetch_flights(self, origin: str, destination: str) -> list:
        """Return flights from origin to destination, falling back to other cities in the same country."""
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

        # Fallback: find the country via the flights table so we only match
        # cities that actually exist in flight data (avoids wrong-country matches
        # from the raw cities CSV, e.g. "Barcelona" appearing in Brazil).
        country_rows = self._query(
            """SELECT DISTINCT co.name AS country_name, co.id AS country_id
               FROM flights f
               JOIN cities dc  ON f.destination_city_id = dc.id
               JOIN countries co ON dc.country_id = co.id
              WHERE LOWER(dc.name) = ?
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

    def get_flight_dimensions(self, origin: str, destination: str) -> dict:
        """Return distinct airlines and price range for flights from origin to destination."""
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

    def get_origin_cities_in_country(self, country_name: str, destination: str | None = None) -> list:
        """Return distinct origin cities in country_name that have outgoing flights (optionally to destination)."""
        if destination:
            rows = self._query(
                """SELECT DISTINCT oc.name
                     FROM flights f
                     JOIN cities oc ON f.origin_city_id      = oc.id
                     JOIN cities dc ON f.destination_city_id = dc.id
                     JOIN countries co ON oc.country_id      = co.id
                    WHERE LOWER(co.name) = ? AND LOWER(dc.name) = ?""",
                (country_name.strip().lower(), destination.strip().lower()),
            )
        else:
            rows = self._query(
                """SELECT DISTINCT oc.name
                     FROM flights f
                     JOIN cities oc ON f.origin_city_id = oc.id
                     JOIN countries co ON oc.country_id  = co.id
                    WHERE LOWER(co.name) = ?""",
                (country_name.strip().lower(),),
            )
        return [r["name"] for r in rows]

    def get_cities_in_country(self, country_name: str, origin: str | None = None) -> list:
        """Return distinct destination cities in country_name (optionally only from origin)."""
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

    def get_reachable_destinations_by_distance(self, origin: str, destination: str, limit: int = 10) -> list:
        """Return destinations reachable from origin sorted by distance from destination."""
        target = self._query(
            """SELECT c.lat, c.lng
                FROM cities c
                WHERE LOWER(c.name) = ?
                LIMIT 1""",
            (destination.strip().lower(),),
        )
        if not target:
            return [{"message": f"Could not locate destination {destination}."}]

        lat = target[0]["lat"]
        lng = target[0]["lng"]

        rows = self._query(
            """SELECT dc.name AS city,
                        co.name AS country,
                        dc.lat,
                        dc.lng,
                        6371.0 * acos(
                            max(-1.0, min(1.0,
                                sin(radians(?)) * sin(radians(dc.lat))
                                + cos(radians(?)) * cos(radians(dc.lat))
                                * cos(radians(dc.lng) - radians(?))
                            ))
                        ) AS distance_km
                FROM flights f
                JOIN cities oc ON f.origin_city_id      = oc.id
                JOIN cities dc ON f.destination_city_id = dc.id
                JOIN countries co ON dc.country_id      = co.id
                WHERE LOWER(oc.name) = ?
                AND LOWER(dc.name) != ?
                GROUP BY dc.id
                ORDER BY distance_km ASC
                LIMIT ?""",
            (lat, lat, lng, origin.strip().lower(), destination.strip().lower(), limit),
        )

        if not rows:
            return [{"message": f"No reachable destinations from {origin}."}]
        return rows
    
    def find_connecting_flights(self, origin: str, destination: str) -> list:
        """Find connecting flights (1 or 2 stops) between origin and destination."""
        origin_clean = origin.strip().lower()
        dest_clean = destination.strip().lower()

        # --- שאילתה עבור עצירה אחת (1-Stop) ---
        one_stop_query = """
            SELECT
                c1.name AS origin_city,
                c2.name AS connection_1,
                c3.name AS destination_city,
                f1.airline AS airline_1, f1.flight_number AS flight_1, f1.price AS price_1, 
                f1.departure_time AS dep_1, f1.arrival_time AS arr_1, f1.duration_minutes AS dur_1,
                
                f2.airline AS airline_2, f2.flight_number AS flight_2, f2.price AS price_2,
                f2.departure_time AS dep_2, f2.arrival_time AS arr_2, f2.duration_minutes AS dur_2,
                
                (f1.price + f2.price) AS total_price
            FROM flights f1
            JOIN cities c1 ON f1.origin_city_id = c1.id
            JOIN flights f2 ON f1.destination_city_id = f2.origin_city_id
            JOIN cities c2 ON f1.destination_city_id = c2.id
            JOIN cities c3 ON f2.destination_city_id = c3.id
            WHERE LOWER(c1.name) = ? AND LOWER(c3.name) = ?
              AND c1.id != c2.id AND c2.id != c3.id -- מניעת מסלול מעגלי
            ORDER BY total_price ASC
            LIMIT 10
        """
        one_stop_rows = self._query(one_stop_query, (origin_clean, dest_clean))

        # --- שאילתה עבור שתי עצירות (2-Stops) ---
        two_stop_query = """
            SELECT
                c1.name AS origin_city,
                c2.name AS connection_1,
                c3.name AS connection_2,
                c4.name AS destination_city,
                f1.airline AS airline_1, f1.flight_number AS flight_1, f1.price AS price_1, 
                f1.departure_time AS dep_1, f1.arrival_time AS arr_1, f1.duration_minutes AS dur_1,
                
                f2.airline AS airline_2, f2.flight_number AS flight_2, f2.price AS price_2,
                f2.departure_time AS dep_2, f2.arrival_time AS arr_2, f2.duration_minutes AS dur_2,
                
                f3.airline AS airline_3, f3.flight_number AS flight_3, f3.price AS price_3,
                f3.departure_time AS dep_3, f3.arrival_time AS arr_3, f3.duration_minutes AS dur_3,
                
                (f1.price + f2.price + f3.price) AS total_price
            FROM flights f1
            JOIN cities c1 ON f1.origin_city_id = c1.id
            JOIN flights f2 ON f1.destination_city_id = f2.origin_city_id
            JOIN cities c2 ON f1.destination_city_id = c2.id
            JOIN flights f3 ON f2.destination_city_id = f3.origin_city_id
            JOIN cities c3 ON f2.destination_city_id = c3.id
            JOIN cities c4 ON f3.destination_city_id = c4.id
            WHERE LOWER(c1.name) = ? AND LOWER(c4.name) = ?
              AND c1.id != c2.id AND c2.id != c3.id AND c3.id != c4.id
              AND c1.id != c3.id AND c2.id != c4.id -- מניעת מסלול מעגלי מורכב
            ORDER BY total_price ASC
            LIMIT 10
        """
        two_stop_rows = self._query(two_stop_query, (origin_clean, dest_clean))

        # --- עיבוד התוצאות למבנה נוח עבור ה-Agent ---
        results = []
        
        for row in one_stop_rows:
            results.append({
                "type": "1-stop",
                "total_price": row["total_price"],
                "route": [
                    {
                        "from": row["origin_city"], "to": row["connection_1"], 
                        "airline": row["airline_1"], "flight": row["flight_1"], "price": row["price_1"],
                        "departure_time": row["dep_1"], "arrival_time": row["arr_1"], "duration_minutes": row["dur_1"]
                    },
                    {
                        "from": row["connection_1"], "to": row["destination_city"], 
                        "airline": row["airline_2"], "flight": row["flight_2"], "price": row["price_2"],
                        "departure_time": row["dep_2"], "arrival_time": row["arr_2"], "duration_minutes": row["dur_2"]
                    }
                ]
            })

        for row in two_stop_rows:
            results.append({
                "type": "2-stops",
                "total_price": row["total_price"],
                "route": [
                    {
                        "from": row["origin_city"], "to": row["connection_1"], 
                        "airline": row["airline_1"], "flight": row["flight_1"], "price": row["price_1"],
                        "departure_time": row["dep_1"], "arrival_time": row["arr_1"], "duration_minutes": row["dur_1"]
                    },
                    {
                        "from": row["connection_1"], "to": row["connection_2"], 
                        "airline": row["airline_2"], "flight": row["flight_2"], "price": row["price_2"],
                        "departure_time": row["dep_2"], "arrival_time": row["arr_2"], "duration_minutes": row["dur_2"]
                    },
                    {
                        "from": row["connection_2"], "to": row["destination_city"], 
                        "airline": row["airline_3"], "flight": row["flight_3"], "price": row["price_3"],
                        "departure_time": row["dep_3"], "arrival_time": row["arr_3"], "duration_minutes": row["dur_3"]
                    }
                ]
            })

        # מיון כל התוצאות (גם עצירה אחת וגם שתיים) מהזול ליקר
        results = sorted(results, key=lambda x: x["total_price"])

        if not results:
            return [{"message": f"No connecting flights found from {origin} to {destination}."}]

        # מחזירים את 10 התוצאות הראשונות למודל
        return results[:10]