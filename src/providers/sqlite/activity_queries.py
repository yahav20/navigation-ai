"""SQLite queries for activities."""


class SQLiteActivityQueriesMixin:
    """Query activity rows from the SQLite database."""

    def fetch_activities(self, city: str) -> list:
        """Return activities available in the given city."""
        rows = self._query(
            """SELECT a.name, a.categories, a.price
               FROM activities a
               JOIN cities c ON a.city_id = c.id
              WHERE LOWER(c.name) = ?""",
            (city.strip().lower(),),
        )
        if not rows:
            return [{"message": f"No available activities found in {city}."}]
        return rows
    def get_activities_by_city(self, destination: str):
        """Return activities in a given city."""

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

        # 2. fetch activities
        rows = self._query(
            """
            SELECT
                id,
                name,
                categories,
                price,
                min_age,
                latitude,
                longitude,
                avg_duration_minutes,
                opening_time,
                closing_time,
                operating_days,
                best_time_of_day,
                food_available,
                requires_booking,
                rating
            FROM activities
            WHERE city_id = ?
            ORDER BY rating DESC, price ASC
            """,
            (city_id,),
        )

        if not rows:
            return [{"message": f"No activities found in {destination}."}]

        return [dict(row) for row in rows]
