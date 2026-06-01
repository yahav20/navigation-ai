"""TripAdvisor location keys for seeded cities."""

CITY_METADATA = {
    "tel aviv":  {"ta_key": "g293984", "full_name": "Tel Aviv", "alpha2": "IL"},
    "paris":     {"ta_key": "g187147", "full_name": "Paris", "alpha2": "FR"},
    "london":    {"ta_key": "g186338", "full_name": "London", "alpha2": "GB"},
    "tokyo":     {"ta_key": "g298184", "full_name": "Tokyo", "alpha2": "JP"},
    "new york":  {"ta_key": "g60763", "full_name": "New York City", "alpha2": "US"},
    "berlin":    {"ta_key": "g187323", "full_name": "Berlin", "alpha2": "DE"},
    "amsterdam": {"ta_key": "g188590", "full_name": "Amsterdam", "alpha2": "NL"},
}

def get_city_info(city_name: str) -> dict | None:
    """Resolve city name to metadata."""
    key = city_name.lower().replace(" city", "").strip()
    return CITY_METADATA.get(key)
