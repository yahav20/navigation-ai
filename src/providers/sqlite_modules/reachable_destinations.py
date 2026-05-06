
def get_reachable_destinations_by_distance(
    self, origin: str, destination: str, limit: int = 10
) -> list:
    target = self._query(
        """SELECT c.lat, c.lng
            FROM cities c
            WHERE LOWER(c.name) = ?
            LIMIT 1""",
        (destination.strip().lower(),),
    )
    if not target:
        return [{"message": f"Could not locate destination {destination}."}]

    lat = target[0]["lat"]
    lng = target[0]["lng"]

    rows = self._query(
        """SELECT dc.name AS city,
                    co.name AS country,
                    dc.lat,
                    dc.lng,
                    6371.0 * acos(
                        max(-1.0, min(1.0,
                            sin(radians(?)) * sin(radians(dc.lat))
                            + cos(radians(?)) * cos(radians(dc.lat))
                            * cos(radians(dc.lng) - radians(?))
                        ))
                    ) AS distance_km
            FROM flights f
            JOIN cities oc ON f.origin_city_id      = oc.id
            JOIN cities dc ON f.destination_city_id = dc.id
            JOIN countries co ON dc.country_id      = co.id
            WHERE LOWER(oc.name) = ?
            AND LOWER(dc.name) != ?
            GROUP BY dc.id
            ORDER BY distance_km ASC
            LIMIT ?""",
        (lat, lat, lng, origin.strip().lower(), destination.strip().lower(), limit),
    )

    if not rows:
        return [{"message": f"No reachable destinations from {origin}."}]
    return rows