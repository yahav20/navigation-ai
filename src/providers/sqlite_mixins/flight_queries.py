class SQLiteFlightQueriesMixin:
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

    def get_origin_cities_in_country(self, country_name: str, destination: str = None) -> list:
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

    def get_reachable_destinations_by_distance(self, origin: str, destination: str, limit: int = 10) -> list:
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
