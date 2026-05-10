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
