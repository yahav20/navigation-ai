"""SQL mixin for recommendation-specific destination queries."""


class RecommendationQueriesMixin:
    """Queries needed by the recommendation agent that don't exist in the core provider."""

    def search_by_activity_category(self, category: str) -> list[dict]:
        """Return cities that have at least one activity matching the given category."""
        rows = self._query(
            """SELECT c.name AS city,
                      co.name AS country,
                      COUNT(a.id) AS activity_count,
                      GROUP_CONCAT(a.name, ', ') AS sample_activities
               FROM activities a
               JOIN cities c ON a.city_id = c.id
               JOIN countries co ON c.country_id = co.id
              WHERE LOWER(a.category) = LOWER(?)
              GROUP BY c.id
              ORDER BY activity_count DESC""",
            (category.strip(),),
        )
        if not rows:
            return [{"message": f"No destinations found for activity category '{category}'."}]
        return rows

    def get_all_destinations_from_origin(
        self,
        origin: str,
        max_flight_hours: float | None = None,
    ) -> list[dict]:
        """Return all cities reachable by flight from origin, with price range and duration.

        When max_flight_hours is set, only routes whose shortest flight fits within
        that limit are returned (uses real duration_hours from the flights table).
        """
        origin_lower = origin.strip().lower()

        if max_flight_hours is not None:
            rows = self._query(
                """SELECT dc.name AS city,
                          co.name AS country,
                          MIN(f.price) AS min_price,
                          MAX(f.price) AS max_price,
                          MIN(f.duration_hours) AS min_duration_hours
                   FROM flights f
                   JOIN cities oc ON f.origin_city_id      = oc.id
                   JOIN cities dc ON f.destination_city_id = dc.id
                   JOIN countries co ON dc.country_id      = co.id
                  WHERE LOWER(oc.name) = ?
                    AND f.duration_hours <= ?
                  GROUP BY dc.id
                  ORDER BY min_duration_hours ASC""",
                (origin_lower, max_flight_hours),
            )
        else:
            rows = self._query(
                """SELECT dc.name AS city,
                          co.name AS country,
                          MIN(f.price) AS min_price,
                          MAX(f.price) AS max_price,
                          MIN(f.duration_hours) AS min_duration_hours
                   FROM flights f
                   JOIN cities oc ON f.origin_city_id      = oc.id
                   JOIN cities dc ON f.destination_city_id = dc.id
                   JOIN countries co ON dc.country_id      = co.id
                  WHERE LOWER(oc.name) = ?
                  GROUP BY dc.id
                  ORDER BY min_duration_hours ASC""",
                (origin_lower,),
            )

        if not rows:
            return [{"message": f"No reachable destinations found from '{origin}'."}]
        return rows

    def search_by_tag(self, tag: str) -> list[dict]:
        """Return cities that carry the given tag."""
        rows = self._query(
            """SELECT c.name AS city, co.name AS country,
                      GROUP_CONCAT(ct.tag, ', ') AS all_tags
               FROM city_tags ct
               JOIN cities c ON ct.city_id = c.id
               JOIN countries co ON c.country_id = co.id
              WHERE LOWER(ct.tag) = LOWER(?)
              GROUP BY c.id
              ORDER BY c.name""",
            (tag.strip(),),
        )
        if not rows:
            return [{"message": f"No destinations found with tag '{tag}'."}]
        return rows

    def get_all_tags(self) -> list[str]:
        """Return every distinct tag in the database."""
        rows = self._query("SELECT DISTINCT tag FROM city_tags ORDER BY tag")
        return [r["tag"] for r in rows]

    def find_within_budget(self, origin: str, budget: float, trip_days: int) -> list[dict]:
        """Return destinations where cheapest_flight + (cheapest_hotel * days) fits the budget.

        Results are ordered by estimated minimum total cost ascending.
        Only destinations that have both flight and hotel data are included.
        """
        rows = self._query(
            """SELECT dc.name AS city,
                      co.name AS country,
                      MIN(f.price)                                    AS cheapest_flight,
                      MIN(h.price_per_night)                          AS cheapest_hotel_per_night,
                      MIN(f.price) + (MIN(h.price_per_night) * ?)     AS estimated_min_total,
                      MIN(f.duration_hours)                           AS min_flight_hours
               FROM flights f
               JOIN cities oc ON f.origin_city_id      = oc.id
               JOIN cities dc ON f.destination_city_id = dc.id
               JOIN countries co ON dc.country_id      = co.id
               JOIN hotels h  ON h.city_id             = dc.id
              WHERE LOWER(oc.name) = ?
              GROUP BY dc.id
             HAVING MIN(f.price) + (MIN(h.price_per_night) * ?) <= ?
              ORDER BY estimated_min_total ASC""",
            (trip_days, origin.strip().lower(), trip_days, budget),
        )
        if not rows:
            return [{"message": f"No destinations found from '{origin}' within a ${budget:.0f} budget for {trip_days} days."}]
        return rows

    def get_recommended_duration(self, city: str) -> dict:
        """Return the recommended trip length and notes for a city."""
        rows = self._query(
            """SELECT rd.min_days, rd.max_days, rd.notes
               FROM recommended_duration rd
               JOIN cities c ON rd.city_id = c.id
              WHERE LOWER(c.name) = ?
              LIMIT 1""",
            (city.strip().lower(),),
        )
        if not rows:
            return {"message": f"No duration recommendation found for '{city}'."}
        return {
            "city": city,
            "min_days": rows[0]["min_days"],
            "max_days": rows[0]["max_days"],
            "notes": rows[0]["notes"],
        }

    def get_city_profile(self, city: str) -> dict:
        """Return a rich profile of a city: activity breakdown, best visit months, and per-season temperatures."""
        city_lower = city.strip().lower()

        activities = self._query(
            """SELECT a.category,
                      GROUP_CONCAT(a.name, ', ') AS activities,
                      COUNT(a.id) AS count
               FROM activities a
               JOIN cities c ON a.city_id = c.id
              WHERE LOWER(c.name) = ?
              GROUP BY a.category
              ORDER BY count DESC""",
            (city_lower,),
        )

        best_time = self._query(
            """SELECT b.months, b.reason
               FROM best_time_to_visit b
               JOIN cities c ON b.city_id = c.id
              WHERE LOWER(c.name) = ?
              LIMIT 1""",
            (city_lower,),
        )

        weather = self._query(
            """SELECT w.season, w.temperature
               FROM average_weather w
               JOIN cities c ON w.city_id = c.id
              WHERE LOWER(c.name) = ?""",
            (city_lower,),
        )

        if not activities and not best_time and not weather:
            return {"message": f"No profile data found for '{city}'."}

        return {
            "city": city,
            "activity_categories": [
                {
                    "category": r["category"],
                    "activities": r["activities"],
                    "count": r["count"],
                }
                for r in activities
            ],
            "best_time_to_visit": {
                "months": (
                    [m.strip() for m in best_time[0]["months"].split(",")]
                    if best_time
                    else []
                ),
                "reason": best_time[0]["reason"] if best_time else None,
            },
            "weather_by_season": {r["season"]: r["temperature"] for r in weather},
        }
