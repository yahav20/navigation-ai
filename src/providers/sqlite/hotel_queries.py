"""SQLite queries for hotels."""


class SQLiteHotelQueriesMixin:
    """Query hotel rows from the SQLite database."""

    def fetch_hotels(self, city: str, max_price: int | None = None) -> list:
        """Return hotels in the given city, optionally filtered by max price."""
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

    def get_hotel_dimensions(self, city: str) -> dict:
        """Return distinct hotel star ratings and price range in the given city."""
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
    def get_hotels_by_city(self, destination: str):
        destination = destination.strip().lower()

        # 1. find city id
        rows = self._query(
            """
            SELECT id
            FROM cities
            WHERE LOWER(name) = ?
            """,
            (destination,),
        )

        if not rows:
            return [{"error": "CITY_NOT_FOUND", "city": destination}]

        city_id = rows[0]["id"]

        # 2. get hotels
        rows = self._query(
            """
            SELECT
                id,
                name,
                price_per_night,
                stars,
                min_age,
                hotel_type,
                distance_from_center_km,
                breakfast_available,
                breakfast_price,
                amenities,
                is_kosher,
                latitude,
                longitude
            FROM hotels
            WHERE city_id = ?
            ORDER BY stars DESC, price_per_night ASC
            """,
            (city_id,),
        )

        return [dict(row) for row in rows]