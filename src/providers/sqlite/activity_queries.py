class SQLiteActivityQueriesMixin:
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
