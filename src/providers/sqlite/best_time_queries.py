class SQLiteBestTimeQueriesMixin:
    def get_best_time_to_visit(self, city: str) -> dict:
        rows = self._query(
            """SELECT b.months, b.reason
               FROM best_time_to_visit b
               JOIN cities c ON b.city_id = c.id
              WHERE LOWER(c.name) = ?""",
            (city.strip().lower(),),
        )
        if not rows:
            return {"message": f"No recommendations found for {city}."}
        return {
            "months": [m.strip() for m in rows[0]["months"].split(",")],
            "reason": rows[0]["reason"],
        }
