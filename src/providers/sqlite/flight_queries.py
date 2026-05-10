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
