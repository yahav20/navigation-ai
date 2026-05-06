def get_hotel_dimensions(provider, city: str) -> dict:
    rows = provider._query(
        """SELECT DISTINCT h.stars,
                  MIN(h.price_per_night) AS price_min,
                  MAX(h.price_per_night) AS price_max
             FROM hotels h
             JOIN cities c ON h.city_id = c.id
            WHERE LOWER(c.name) = ?
            GROUP BY h.stars""",
        (city.strip().lower(),),
    )
    if not rows:
        return {"available": False}
    stars = sorted({r["stars"] for r in rows if r["stars"] is not None})
    all_prices = [r["price_min"] for r in rows] + [r["price_max"] for r in rows]
    all_prices = [p for p in all_prices if p is not None]
    return {
        "stars_available": stars,
        "price_min": min(all_prices) if all_prices else None,
        "price_max": max(all_prices) if all_prices else None,
    }


def get_flight_dimensions(provider, origin: str, destination: str) -> dict:
    rows = provider._query(
        """SELECT DISTINCT f.airline,
                  MIN(f.price) AS price_min,
                  MAX(f.price) AS price_max
             FROM flights f
             JOIN cities oc ON f.origin_city_id      = oc.id
             JOIN cities dc ON f.destination_city_id = dc.id
            WHERE LOWER(oc.name) = ? AND LOWER(dc.name) = ?
            GROUP BY f.airline""",
        (origin.strip().lower(), destination.strip().lower()),
    )
    if not rows:
        return {"available": False}
    airlines = sorted({r["airline"] for r in rows if r["airline"]})
    all_prices = [r["price_min"] for r in rows] + [r["price_max"] for r in rows]
    all_prices = [p for p in all_prices if p is not None]
    return {
        "airlines_available": airlines,
        "price_min": min(all_prices) if all_prices else None,
        "price_max": max(all_prices) if all_prices else None,
    }


def get_cities_in_country(provider, country_name: str, origin: str = None) -> list:
    if origin:
        rows = provider._query(
            """SELECT DISTINCT dc.name
                 FROM flights f
                 JOIN cities oc ON f.origin_city_id      = oc.id
                 JOIN cities dc ON f.destination_city_id = dc.id
                 JOIN countries co ON dc.country_id      = co.id
                WHERE LOWER(co.name) = ? AND LOWER(oc.name) = ?""",
            (country_name.strip().lower(), origin.strip().lower()),
        )
    else:
        rows = provider._query(
            """SELECT DISTINCT dc.name
                 FROM flights f
                 JOIN cities dc ON f.destination_city_id = dc.id
                 JOIN countries co ON dc.country_id      = co.id
                WHERE LOWER(co.name) = ?""",
            (country_name.strip().lower(),),
        )
    return [r["name"] for r in rows]
