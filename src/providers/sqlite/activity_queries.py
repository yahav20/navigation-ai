"""SQLite queries for activities."""


class SQLiteActivityQueriesMixin:
    """Query activity rows from the SQLite database."""

    def fetch_activities(self, city: str) -> list:
        """Return activities available in the given city."""
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
