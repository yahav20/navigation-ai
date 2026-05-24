"""SQLite queries for average weather data."""


class SQLiteWeatherQueriesMixin:
    """Query average-weather rows from the SQLite database."""

    def get_average_weather(self, city: str, season: str = None) -> dict:
        """Return average weather for a given city. If season is provided, filters by it."""
        city = city.strip().lower()

        if season:
            rows = self._query(
                """SELECT w.season, w.temperature
                   FROM average_weather w
                   JOIN cities c ON w.city_id = c.id
                  WHERE LOWER(c.name) = ? AND LOWER(w.season) = ?""",
                (city, season.strip().lower()),
            )
            if not rows:
                return {"message": f"No weather data found for {season} in {city}."}
            return {"season": rows[0]["season"], "temperature": rows[0]["temperature"]}

        else:
            # 1. find city id
            rows = self._query(
                "SELECT id FROM cities WHERE LOWER(name) = ?",
                (city,),
            )

            if not rows:
                return {"available": False, "city": city}

            city_id = rows[0]["id"]

            # 2. fetch weather for all seasons
            rows = self._query(
                "SELECT season, temperature FROM average_weather WHERE city_id = ? ORDER BY id",
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