def get_hotel_dimensions(provider, city: str) -> dict:
    hotels = [h for h in provider.fetch_hotels(city) if "message" not in h]
    if not hotels:
        return {"available": False}
    stars = sorted({h["stars"] for h in hotels if "stars" in h})
    prices = [h["price_per_night"] for h in hotels if "price_per_night" in h]
    return {
        "stars_available": stars,
        "price_min": min(prices) if prices else None,
        "price_max": max(prices) if prices else None,
    }


def get_flight_dimensions(provider, origin: str, destination: str) -> dict:
    flights = [f for f in provider.fetch_flights(origin, destination) if "message" not in f]
    if not flights:
        return {"available": False}
    airlines = sorted({f["airline"] for f in flights if "airline" in f})
    prices = [f["price"] for f in flights if "price" in f]
    return {
        "airlines_available": airlines,
        "price_min": min(prices) if prices else None,
        "price_max": max(prices) if prices else None,
    }


def get_cities_in_country(provider, country_name: str, origin: str = None) -> list:
    return []  # JSON provider has no country-level data
