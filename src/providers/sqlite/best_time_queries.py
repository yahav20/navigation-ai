"""SQLite queries for best-time-to-visit advisor."""


class SQLiteBestTimeQueriesMixin:
    """Query best-time-to-visit rows from the SQLite database."""

    def get_best_time_to_visit(self, city: str) -> dict:
        """Return the recommended visit months and reason for the given city."""
        rows = self._query(
            """SELECT b.months, b.reason
               FROM best_time_to_visit b
               JOIN cities c ON b.city_id = c.id
              WHERE LOWER(c.name) = ?""",
            (city.strip().lower(),),
        )
        if not rows:
            return {"message": f"No advisor found for {city}."}
        return {
            "months": [m.strip() for m in rows[0]["months"].split(",")],
            "reason": rows[0]["reason"],
        }
        
