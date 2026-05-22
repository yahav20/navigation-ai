"""SQLite queries for average weather data."""


class SQLiteWeatherQueriesMixin:
    """Query average-weather rows from the SQLite database."""

    def get_average_weather(self, city: str, season: str) -> dict:
        """Return the average weather for the given city and season."""
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
    def get_average_weather(self, city: str):
        """Return average weather by season for a given city."""

        city = city.strip().lower()

        # 1. find city id
        rows = self._query(
            """
            SELECT id
            FROM cities
            WHERE LOWER(name) = ?
            """,
            (city,),
        )

        if not rows:
            return {"available": False, "city": city}

        city_id = rows[0]["id"]

        # 2. fetch weather
        rows = self._query(
            """
            SELECT
                season,
                temperature
            FROM average_weather
            WHERE city_id = ?
            ORDER BY id
            """,
            (city_id,),
        )

        if not rows:
            return {"available": False, "city": city}

        return {
            "available": True,
            "city": city,
            "weather_by_season": [
                {
                    "season": row["season"],
                    "temperature": row["temperature"],
                }
                for row in rows
            ],
        }
