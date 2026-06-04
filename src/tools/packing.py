"""LangChain tool for generating smart packing lists."""
from langchain_core.tools import tool

_BASE_ITEMS = [
    "Passport & copies", "Travel insurance documents", "Local currency / travel card",
    "Phone charger & universal adapter", "Earbuds / headphones", "Reusable water bottle",
    "Snacks for the journey", "Lock for luggage", "First aid kit (plasters, pain relief, antiseptic)",
    "Personal medications", "Sunglasses", "Day backpack",
]

_CLOTHING_BY_CLIMATE = {
    "hot": ["Light breathable shirts (x5)", "Shorts (x3)", "Sun hat / cap", "Sandals", "Swimwear (x2)", "Light jacket for AC"],
    "warm": ["T-shirts (x4)", "Light trousers or jeans (x2)", "Comfortable walking shoes", "Light jacket", "Sunhat"],
    "mild": ["T-shirts (x3)", "Long-sleeve shirts (x2)", "Jeans / chinos (x2)", "Light jacket", "Comfortable sneakers"],
    "cold": ["Thermal base layers", "Warm jumper / fleece", "Waterproof jacket", "Warm trousers (x2)", "Boots or waterproof shoes", "Gloves & hat", "Scarf"],
    "very_cold": ["Heavy winter coat", "Thermal underwear (x2)", "Wool socks", "Waterproof boots", "Thick gloves & beanie", "Scarf", "Fleece layer"],
}

_EXTRAS_BY_TYPE = {
    "beach": ["Sunscreen SPF 50+", "After-sun lotion", "Beach towel", "Underwater camera / waterproof phone case", "Flip flops (extra pair)"],
    "city": ["Comfortable walking shoes (broken in!)", "City map or offline maps downloaded", "Anti-theft crossbody bag", "Dress code outfits for restaurants/clubs"],
    "nature": ["Hiking boots", "Trekking poles", "Insect repellent", "Rain poncho", "Headlamp", "Energy bars"],
    "cultural": ["Modest clothing (covered shoulders/knees) for religious sites", "Scarf or shawl", "Respectful footwear"],
    "business": ["Business attire (x2 outfits)", "Laptop & charger", "Business cards", "Ironing bag / wrinkle releaser"],
}

_TOILETRIES = [
    "Toothbrush & toothpaste", "Deodorant", "Shampoo & conditioner (travel size)",
    "Body wash", "Razor", "Lip balm", "Hand sanitiser", "Wet wipes",
]


def _infer_climate(season: str, destination: str) -> str:
    hot_cities = {"Bangkok", "Dubai", "Bali", "Phuket", "Singapore", "Miami", "Cairo", "Tel Aviv", "Marrakech"}
    cold_cities = {"Reykjavik", "Oslo", "Helsinki", "Stockholm", "Copenhagen", "Moscow", "Toronto"}
    dest_cap = destination.strip().title()

    if dest_cap in hot_cities:
        return "hot" if season in ("summer", "spring") else "warm"
    if dest_cap in cold_cities:
        return "very_cold" if season == "winter" else "cold"

    season_map = {"summer": "hot", "spring": "mild", "autumn": "mild", "fall": "mild", "winter": "cold"}
    return season_map.get(season.lower(), "mild")


@tool
def get_packing_list(destination: str, season: str, trip_days: int, trip_type: str = "city") -> dict:
    """Generate a smart packing list for a trip.

    Args:
        destination: city or country to travel to
        season: 'summer', 'winter', 'spring', or 'autumn'
        trip_days: number of days for the trip
        trip_type: one of 'city', 'beach', 'nature', 'cultural', 'business' (default: 'city')

    Use this when the user asks:
    - 'What should I pack for [destination]?'
    - 'Give me a packing list for [trip] days in [city] in [season]'
    - 'What clothes do I need for a beach trip in summer?'
    """
    climate = _infer_climate(season, destination)
    clothing = _CLOTHING_BY_CLIMATE.get(climate, _CLOTHING_BY_CLIMATE["mild"])
    extras = _EXTRAS_BY_TYPE.get(trip_type.lower(), _EXTRAS_BY_TYPE["city"])

    if trip_days > 10:
        clothing = [c.replace("x3", "x5").replace("x2", "x4").replace("x4", "x6").replace("x5", "x7") for c in clothing]
    elif trip_days <= 3:
        clothing = [c.replace("x5", "x3").replace("x4", "x2").replace("x3", "x2") for c in clothing]

    return {
        "destination": destination,
        "season": season,
        "climate": climate,
        "trip_days": trip_days,
        "trip_type": trip_type,
        "essentials": _BASE_ITEMS,
        "clothing": clothing,
        "toiletries": _TOILETRIES,
        "extras_for_trip_type": extras,
        "tip": "Roll clothes instead of folding to save space and reduce wrinkles. Pack a small empty bag for souvenirs on the way back!",
    }
