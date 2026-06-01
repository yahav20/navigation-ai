"""SQLite queries for activities."""


class SQLiteActivityQueriesMixin:
    """Query activity rows from the SQLite database."""

    def fetch_activities(self, city: str) -> list:
        """Return activities available in the given city."""
        rows = self._query(
            """SELECT a.id, a.name, a.categories, a.price, a.min_age, 
                      a.latitude, a.longitude, a.avg_duration_minutes, 
                      a.opening_time, a.closing_time, a.operating_days, 
                      a.best_time_of_day, a.food_available, a.requires_booking, a.rating
               FROM activities a
               JOIN cities c ON a.city_id = c.id
               WHERE LOWER(c.name) = ?
               ORDER BY a.rating DESC, a.price ASC""",
            (city.strip().lower(),),
        )
        if not rows:
            return []

        return [dict(row) for row in rows]
    
    
    def fetch_activities_with_features(self, city: str) -> list:
        """Return activities available in the given city, including their features."""
        rows = self._query(
            """SELECT a.id, a.name, a.categories, a.price, a.min_age, 
                      a.latitude, a.longitude, a.avg_duration_minutes, 
                      a.opening_time, a.closing_time, a.operating_days, 
                      a.best_time_of_day, a.food_available, a.requires_booking, a.rating,
                      GROUP_CONCAT(af.name) as features_list
               FROM activities a
               JOIN cities c ON a.city_id = c.id
               LEFT JOIN activity_feature_mapping afm ON a.id = afm.activity_id
               LEFT JOIN activity_features af ON afm.feature_id = af.id
               WHERE LOWER(c.name) = ?
               GROUP BY a.id
               ORDER BY a.rating DESC, a.price ASC""",
            (city.strip().lower(),),
        )
        if not rows:
            return []
        results = []
        for row in rows:
            act = dict(row)
           
            features_str = act.pop('features_list', None)
            if features_str:
                act['features'] = [f.strip() for f in features_str.split(',')]
            else:
                act['features'] = []
            
            results.append(act)

        return results
    
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
