"""LangChain tools for geocoding and location resolution."""
from langchain_core.tools import tool

from providers.google_maps import geocode_city


@tool
def geocode_location(address: str) -> dict:
    """Resolve an address or city name to latitude and longitude coordinates.

    Useful for getting coordinates for a destination before searching for nearby places.
    """
    if not address:
        return {}

    coords = geocode_city(address)
    if not coords:
        return {"error": f"Could not geocode '{address}'"}

    return {
        "address": address,
        "latitude": coords[0],
        "longitude": coords[1],
    }
