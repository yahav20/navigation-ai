"""LangChain tools for the advisor agent — destination discovery and city profiling."""
from __future__ import annotations

import json
import urllib.request

import requests
from langchain_core.tools import tool

from providers.sqlite.provider import SQLiteDataProvider
from security import validate_city, validate_positive_number
from tools.activities import fetch_activities
from tools.concerts import search_concerts
from tools.weather import get_average_weather, get_best_time_to_visit

_provider = SQLiteDataProvider()


# ---------------------------------------------------------------------------
# Currency Exchange
# ---------------------------------------------------------------------------

_CURRENCY_ALIASES: dict[str, str] = {
    "dollar": "USD", "dollars": "USD", "usd": "USD",
    "euro": "EUR", "euros": "EUR", "eur": "EUR",
    "pound": "GBP", "pounds": "GBP", "gbp": "GBP",
    "yen": "JPY", "jpy": "JPY",
    "shekel": "ILS", "shekels": "ILS", "ils": "ILS", "nis": "ILS",
    "franc": "CHF", "francs": "CHF", "chf": "CHF",
    "krona": "SEK", "kronor": "SEK", "sek": "SEK",
    "won": "KRW", "krw": "KRW",
    "rupee": "INR", "rupees": "INR", "inr": "INR",
    "real": "BRL", "reais": "BRL", "brl": "BRL",
    "peso": "MXN", "pesos": "MXN", "mxn": "MXN",
    "dirham": "AED", "dirhams": "AED", "aed": "AED",
    "lira": "TRY", "try": "TRY",
    "baht": "THB", "thb": "THB",
    "zloty": "PLN", "zlotys": "PLN", "pln": "PLN",
    "forint": "HUF", "huf": "HUF",
    "crown": "CZK", "crowns": "CZK", "czk": "CZK",
    "kuna": "HRK", "hrk": "HRK",
    "dinar": "JOD", "jod": "JOD",
    "riyal": "SAR", "riyals": "SAR", "sar": "SAR",
    "ringgit": "MYR", "myr": "MYR",
    "dong": "VND", "vnd": "VND",
    "peso arg": "ARS", "ars": "ARS",
    "canadian dollar": "CAD", "cad": "CAD",
    "australian dollar": "AUD", "aud": "AUD",
}


def _normalize_currency(raw: str) -> str:
    key = raw.strip().lower()
    return _CURRENCY_ALIASES.get(key, raw.strip().upper())


@tool
def get_currency_exchange(from_currency: str, to_currency: str, amount: float = 1.0) -> dict:
    """Get the live currency exchange rate between two currencies and convert an amount.

    Accepts currency codes (USD, EUR, ILS) or common names (dollar, euro, shekel, pound).
    Use this when the user asks things like:
    - 'How much is $500 in euros?'
    - 'What is the exchange rate between dollars and shekels?'
    - 'Convert 1000 baht to USD'
    """
    base = _normalize_currency(from_currency)
    target = _normalize_currency(to_currency)

    url = f"https://open.er-api.com/v6/latest/{base}"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read().decode())
    except Exception as exc:
        return {"error": f"Could not fetch exchange rate: {exc}"}

    if data.get("result") != "success":
        return {"error": f"Unsupported currency code: {base}"}

    rates = data.get("rates", {})
    if target not in rates:
        return {"error": f"Unsupported target currency: {target}"}

    rate = rates[target]
    converted = round(amount * rate, 2)
    return {
        "from": base,
        "to": target,
        "rate": round(rate, 4),
        "amount": amount,
        "converted": converted,
        "updated": data.get("time_last_update_utc", "unknown"),
    }


# ---------------------------------------------------------------------------
# Travel Safety Information
# ---------------------------------------------------------------------------

_SAFETY_DB: dict[str, dict] = {
    "Paris": {"level": 2, "label": "Exercise increased caution", "notes": "Pickpocketing in tourist areas; occasional protests."},
    "London": {"level": 1, "label": "Exercise normal precautions", "notes": "Generally very safe; standard big-city vigilance advised."},
    "Rome": {"level": 2, "label": "Exercise increased caution", "notes": "Pickpocketing near the Colosseum and Vatican; scams targeting tourists."},
    "Barcelona": {"level": 2, "label": "Exercise increased caution", "notes": "High pickpocketing rates on Las Ramblas and public transit."},
    "Amsterdam": {"level": 1, "label": "Exercise normal precautions", "notes": "Very safe overall; bicycle theft is common."},
    "Tokyo": {"level": 1, "label": "Exercise normal precautions", "notes": "One of the safest major cities in the world."},
    "Bangkok": {"level": 2, "label": "Exercise increased caution", "notes": "Tuk-tuk and gem scams target tourists; political demonstrations occur."},
    "Istanbul": {"level": 2, "label": "Exercise increased caution", "notes": "Terrorist threat in crowded places; petty crime in tourist districts."},
    "Dubai": {"level": 1, "label": "Exercise normal precautions", "notes": "Very safe; strict local laws - be aware of customs around alcohol and dress."},
    "New York": {"level": 1, "label": "Exercise normal precautions", "notes": "Standard big-city precautions; certain neighborhoods require more care at night."},
    "Los Angeles": {"level": 2, "label": "Exercise increased caution", "notes": "Homeless encampments in some areas; car break-ins are common."},
    "Miami": {"level": 2, "label": "Exercise increased caution", "notes": "Avoid Little Haiti and Overtown at night; hurricane season June-November."},
    "Mexico City": {"level": 3, "label": "Reconsider travel", "notes": "High crime in some colonias; kidnappings and robbery occur. Stick to tourist zones."},
    "Cairo": {"level": 3, "label": "Reconsider travel", "notes": "Terrorism threat; street harassment common. Tourist sites are well-guarded."},
    "Tel Aviv": {"level": 3, "label": "Reconsider travel", "notes": "Active conflict in region; rocket alerts possible. Check latest advisories."},
    "Jerusalem": {"level": 3, "label": "Reconsider travel", "notes": "Active conflict in region; political tensions in the Old City."},
    "Prague": {"level": 1, "label": "Exercise normal precautions", "notes": "Very safe; petty theft on trams and in Old Town Square."},
    "Vienna": {"level": 1, "label": "Exercise normal precautions", "notes": "One of the safest capitals in Europe."},
    "Zurich": {"level": 1, "label": "Exercise normal precautions", "notes": "Extremely safe; very low crime rate."},
    "Singapore": {"level": 1, "label": "Exercise normal precautions", "notes": "One of the safest cities in the world; strict local laws."},
    "Sydney": {"level": 1, "label": "Exercise normal precautions", "notes": "Very safe; standard urban precautions apply."},
    "Rio de Janeiro": {"level": 3, "label": "Reconsider travel", "notes": "High violent crime; avoid favelas and isolated areas. Beach theft common."},
    "Buenos Aires": {"level": 2, "label": "Exercise increased caution", "notes": "Pickpocketing and express kidnappings; avoid displaying valuables."},
    "Nairobi": {"level": 3, "label": "Reconsider travel", "notes": "Terrorism and violent crime; carjackings occur. Westgate area risk."},
    "Cape Town": {"level": 3, "label": "Reconsider travel", "notes": "High violent crime, carjackings, and home invasions; avoid townships at night."},
    "Bali": {"level": 2, "label": "Exercise increased caution", "notes": "Generally safe for tourists; scams and petty theft common in Kuta."},
    "Phuket": {"level": 2, "label": "Exercise increased caution", "notes": "Safe resort areas; road accidents and jet-ski scams are risks."},
    "Athens": {"level": 2, "label": "Exercise increased caution", "notes": "Pickpocketing in Monastiraki; political demonstrations near Syntagma Square."},
    "Budapest": {"level": 1, "label": "Exercise normal precautions", "notes": "Very safe; bar scams and overcharging in tourist areas."},
    "Warsaw": {"level": 1, "label": "Exercise normal precautions", "notes": "Generally safe; standard big-city precautions."},
    "Lisbon": {"level": 1, "label": "Exercise normal precautions", "notes": "Very safe; pickpocketing in trams (especially No. 28) and Alfama."},
    "Madrid": {"level": 2, "label": "Exercise increased caution", "notes": "Pickpocketing in Puerta del Sol and around the Prado."},
    "Berlin": {"level": 1, "label": "Exercise normal precautions", "notes": "Very safe; petty theft on U-Bahn and in Alexanderplatz area."},
    "Munich": {"level": 1, "label": "Exercise normal precautions", "notes": "Very safe; be aware during Oktoberfest (crowding and alcohol-related incidents)."},
    "Copenhagen": {"level": 1, "label": "Exercise normal precautions", "notes": "One of the safest cities in Europe."},
    "Stockholm": {"level": 1, "label": "Exercise normal precautions", "notes": "Very safe; standard Scandinavian city precautions."},
    "Oslo": {"level": 1, "label": "Exercise normal precautions", "notes": "Very safe; very low crime rate."},
    "Helsinki": {"level": 1, "label": "Exercise normal precautions", "notes": "Extremely safe city."},
    "Reykjavik": {"level": 1, "label": "Exercise normal precautions", "notes": "One of the safest capitals in the world. Watch for unpredictable weather outdoors."},
    "Kyoto": {"level": 1, "label": "Exercise normal precautions", "notes": "Extremely safe; very low crime."},
    "Seoul": {"level": 1, "label": "Exercise normal precautions", "notes": "Very safe; standard urban precautions."},
    "Hong Kong": {"level": 2, "label": "Exercise increased caution", "notes": "Political tensions; demonstrations can occur. Otherwise very safe for tourists."},
    "Taipei": {"level": 1, "label": "Exercise normal precautions", "notes": "Very safe; low crime rate."},
    "Kuala Lumpur": {"level": 2, "label": "Exercise increased caution", "notes": "Petty crime and bag snatching; terrorism threat exists."},
    "Jakarta": {"level": 3, "label": "Reconsider travel", "notes": "Terrorism threat; high petty crime. Stick to tourist and expat areas."},
    "Hanoi": {"level": 2, "label": "Exercise increased caution", "notes": "Traffic is chaotic; bag snatching by motorbike. Generally safe otherwise."},
    "Ho Chi Minh City": {"level": 2, "label": "Exercise increased caution", "notes": "Bag snatching common; traffic accidents are a major risk."},
    "Marrakech": {"level": 2, "label": "Exercise increased caution", "notes": "Aggressive touts and scams in the medina; occasional petty crime."},
    "Casablanca": {"level": 2, "label": "Exercise increased caution", "notes": "Petty theft; avoid isolated areas at night."},
    "Accra": {"level": 2, "label": "Exercise increased caution", "notes": "Petty crime and occasional scams; safer than many West African capitals."},
    "Johannesburg": {"level": 4, "label": "Do not travel", "notes": "Very high violent crime; armed robbery, carjackings. Avoid most areas after dark."},
}

_LEVEL_COLORS = {1: "[LOW]", 2: "[MODERATE]", 3: "[HIGH]", 4: "[CRITICAL]"}


@tool
def get_travel_safety_info(destination: str) -> dict:
    """Get travel safety information and advisory level for a destination city.

    Returns an advisory level (1=Normal, 2=Increased Caution, 3=Reconsider, 4=Do Not Travel),
    a label, and specific safety notes.

    Use this when the user asks:
    - 'Is [city] safe to visit?'
    - 'What safety precautions should I take in [city]?'
    - 'Are there any travel warnings for [city]?'
    """
    city = validate_city(destination)
    info = _SAFETY_DB.get(city)
    if info:
        return {
            "city": city,
            "level": info["level"],
            "icon": _LEVEL_COLORS.get(info["level"], "[UNKNOWN]"),
            "label": info["label"],
            "notes": info["notes"],
            "disclaimer": "Based on general travel advisory knowledge. Always check your government's official travel advisory before departure.",
        }
    return {
        "city": city,
        "level": 1,
        "icon": "[LOW]",
        "label": "No specific advisory found",
        "notes": "No specific travel advisory data available for this city. Check your country's official travel advisory website for the latest information.",
        "disclaimer": "Always check your government's official travel advisory before departure.",
    }


# ---------------------------------------------------------------------------
# Visa Requirements
# ---------------------------------------------------------------------------

_VISA_DB: dict[str, dict[str, dict]] = {
    "Israeli": {
        "France": {"required": False, "notes": "Visa-free up to 90 days (Schengen). Passport valid 6 months beyond stay."},
        "Germany": {"required": False, "notes": "Visa-free up to 90 days (Schengen)."},
        "Spain": {"required": False, "notes": "Visa-free up to 90 days (Schengen)."},
        "Italy": {"required": False, "notes": "Visa-free up to 90 days (Schengen)."},
        "Netherlands": {"required": False, "notes": "Visa-free up to 90 days (Schengen)."},
        "Greece": {"required": False, "notes": "Visa-free up to 90 days (Schengen)."},
        "Portugal": {"required": False, "notes": "Visa-free up to 90 days (Schengen)."},
        "Czech Republic": {"required": False, "notes": "Visa-free up to 90 days (Schengen)."},
        "Hungary": {"required": False, "notes": "Visa-free up to 90 days (Schengen)."},
        "Poland": {"required": False, "notes": "Visa-free up to 90 days (Schengen)."},
        "Austria": {"required": False, "notes": "Visa-free up to 90 days (Schengen)."},
        "Switzerland": {"required": False, "notes": "Visa-free up to 90 days. Switzerland is not EU but participates in Schengen."},
        "United Kingdom": {"required": False, "notes": "Visa-free up to 6 months (post-Brexit electronic travel authorisation ETA required from 2025)."},
        "United States": {"required": True, "notes": "B-1/B-2 tourist visa required. Apply at the US Embassy. ESTA not available for Israeli passports."},
        "Canada": {"required": True, "notes": "Visitor visa (Temporary Resident Visa) required. Apply online via IRCC."},
        "Japan": {"required": False, "notes": "Visa-free up to 90 days for tourism."},
        "South Korea": {"required": False, "notes": "Visa-free up to 90 days."},
        "Thailand": {"required": False, "notes": "Visa-free up to 30 days (extendable once at immigration)."},
        "Singapore": {"required": False, "notes": "Visa-free up to 30 days."},
        "Dubai": {"required": False, "notes": "Visa on arrival for 30 days (UAE)."},
        "Turkey": {"required": False, "notes": "Visa-free up to 90 days. Note: tensions may affect entry."},
        "Morocco": {"required": False, "notes": "Visa-free up to 90 days."},
        "Jordan": {"required": True, "notes": "Visa on arrival at most entry points ($35-$50 JOD). Jordan Pass includes visa fee."},
        "Egypt": {"required": True, "notes": "Visa on arrival at Cairo Airport ($25). E-Visa also available online."},
        "India": {"required": True, "notes": "e-Visa required. Apply online at indianvisaonline.gov.in (usually 3-5 days processing)."},
        "China": {"required": True, "notes": "Visa required. Apply at Chinese embassy. 6-day transit visa exemption may apply."},
        "Australia": {"required": True, "notes": "ETA (Electronic Travel Authority) required - apply online, usually instant."},
        "New Zealand": {"required": False, "notes": "NZeTA required ($17 NZD) - apply online before travel."},
        "Brazil": {"required": False, "notes": "Visa-free up to 90 days for tourism."},
        "Argentina": {"required": False, "notes": "Visa-free up to 90 days."},
        "Mexico": {"required": False, "notes": "Visa-free up to 180 days."},
        "Amsterdam": {"required": False, "notes": "Amsterdam is in the Netherlands - Schengen area, visa-free 90 days."},
        "Bangkok": {"required": False, "notes": "Bangkok is in Thailand - visa-free up to 30 days."},
        "Bali": {"required": False, "notes": "Bali is in Indonesia - visa on arrival for 30 days ($35 USD)."},
        "Tokyo": {"required": False, "notes": "Tokyo is in Japan - visa-free up to 90 days."},
        "Seoul": {"required": False, "notes": "Seoul is in South Korea - visa-free up to 90 days."},
        "London": {"required": False, "notes": "UK ETA required from 2025. Visa-free up to 6 months."},
        "Barcelona": {"required": False, "notes": "Barcelona is in Spain - Schengen, visa-free 90 days."},
        "Rome": {"required": False, "notes": "Rome is in Italy - Schengen, visa-free 90 days."},
        "Prague": {"required": False, "notes": "Prague is in Czech Republic - Schengen, visa-free 90 days."},
        "Budapest": {"required": False, "notes": "Budapest is in Hungary - Schengen, visa-free 90 days."},
        "Lisbon": {"required": False, "notes": "Lisbon is in Portugal - Schengen, visa-free 90 days."},
        "Vienna": {"required": False, "notes": "Vienna is in Austria - Schengen, visa-free 90 days."},
    },
    "American": {
        "France": {"required": False, "notes": "Visa-free up to 90 days (Schengen). ETIAS required from 2025."},
        "Germany": {"required": False, "notes": "Visa-free up to 90 days (Schengen)."},
        "United Kingdom": {"required": False, "notes": "Visa-free up to 6 months. UK ETA required from 2025."},
        "Japan": {"required": False, "notes": "Visa-free up to 90 days."},
        "Australia": {"required": True, "notes": "ETA required - apply online, usually instant approval."},
        "India": {"required": True, "notes": "e-Visa required. Apply online (3-5 days)."},
        "China": {"required": True, "notes": "Visa required; apply at Chinese embassy/consulate."},
        "Brazil": {"required": False, "notes": "Visa-free up to 90 days (since 2024)."},
        "Thailand": {"required": False, "notes": "Visa-free up to 30 days."},
        "Singapore": {"required": False, "notes": "Visa-free up to 90 days."},
        "Dubai": {"required": False, "notes": "Visa-free up to 30 days."},
        "Mexico": {"required": False, "notes": "Visa-free. FMM tourist card required (often included in airfare)."},
    },
    "British": {
        "France": {"required": False, "notes": "Visa-free 90/180 days (post-Brexit). ETIAS required from 2025."},
        "Germany": {"required": False, "notes": "Visa-free 90/180 days. ETIAS from 2025."},
        "Spain": {"required": False, "notes": "Visa-free 90/180 days. ETIAS from 2025."},
        "United States": {"required": False, "notes": "ESTA required ($21). Visa-free up to 90 days under VWP."},
        "Australia": {"required": True, "notes": "ETA required - instant online approval."},
        "India": {"required": True, "notes": "e-Visa required. Apply online."},
        "Japan": {"required": False, "notes": "Visa-free up to 90 days."},
        "Dubai": {"required": False, "notes": "Visa-free up to 30 days."},
    },
}

_DEFAULT_PASSPORT_VISAS: dict[str, dict] = {
    "France": {"required": False, "notes": "Most Western passports: visa-free up to 90 days (Schengen). Check your specific passport."},
    "Germany": {"required": False, "notes": "Most Western passports: visa-free up to 90 days (Schengen)."},
    "Japan": {"required": False, "notes": "Most Western passports: visa-free up to 90 days."},
    "Thailand": {"required": False, "notes": "Most Western passports: visa-free up to 30 days."},
    "Dubai": {"required": False, "notes": "Many passports: visa-free or visa on arrival. Check your specific nationality."},
    "India": {"required": True, "notes": "Most nationalities require an e-Visa. Apply at indianvisaonline.gov.in."},
    "China": {"required": True, "notes": "Most nationalities require a visa; apply at your nearest Chinese embassy."},
    "Australia": {"required": True, "notes": "Most nationalities require an ETA or visa; apply online."},
}


@tool
def get_visa_requirements(passport_nationality: str, destination: str) -> dict:
    """Get visa requirements for a traveler based on their passport nationality and destination.

    Use this when the user asks:
    - 'Do I need a visa to visit [destination]?'
    - 'What are the visa requirements for [destination]?'
    - 'How long can I stay in [destination] visa-free?'

    passport_nationality: the user's passport country (e.g. 'Israeli', 'American', 'British').
    destination: the destination country or city (e.g. 'Japan', 'Tokyo', 'Thailand').
    """
    dest = validate_city(destination)

    nat_key = None
    for key in _VISA_DB:
        if key.lower() in passport_nationality.lower() or passport_nationality.lower() in key.lower():
            nat_key = key
            break

    if nat_key and dest in _VISA_DB[nat_key]:
        info = _VISA_DB[nat_key][dest]
        return {
            "passport": nat_key,
            "destination": dest,
            "visa_required": info["required"],
            "details": info["notes"],
            "disclaimer": "Visa rules change. Always verify at your country's official embassy or travel authority before departure.",
        }

    if dest in _DEFAULT_PASSPORT_VISAS:
        info = _DEFAULT_PASSPORT_VISAS[dest]
        return {
            "passport": passport_nationality,
            "destination": dest,
            "visa_required": info["required"],
            "details": info["notes"],
            "disclaimer": "This is general guidance. Verify exact requirements at the official embassy or consulate for your nationality.",
        }

    return {
        "passport": passport_nationality,
        "destination": dest,
        "visa_required": None,
        "details": "No specific visa data available for this combination. Please check with the embassy of your destination country or use IATA Travel Centre / Sherpa tools for precise requirements.",
        "disclaimer": "Always verify visa requirements before booking.",
    }


# ---------------------------------------------------------------------------
# Packing List Generator
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Local Customs & Etiquette
# ---------------------------------------------------------------------------

_CUSTOMS_DB: dict[str, dict] = {
    "Japan": {
        "tipping": "Do NOT tip - it is considered rude.",
        "greetings": "Bow slightly when greeting. Handshakes are becoming more common with foreigners.",
        "dress": "Smart-casual. Remove shoes when entering homes and many traditional restaurants.",
        "dining": "Say 'Itadakimasu' before eating. Do not stick chopsticks upright in rice.",
        "customs": "Queuing is taken seriously. Keep voices low in public. Eating while walking is frowned upon in many areas.",
        "useful_phrases": ["Arigatou gozaimasu (Thank you)", "Sumimasen (Excuse me)", "Hai (Yes)"],
    },
    "Thailand": {
        "tipping": "Tip 20-50 THB in restaurants, 20-100 THB for taxi/hotel.",
        "greetings": "Wai (press palms together and bow slightly). Never touch someone's head.",
        "dress": "Modest dress at temples - covered shoulders and knees. Remove shoes before entering.",
        "dining": "Sharing dishes is common. Do not waste food.",
        "customs": "The royal family is sacred - never disrespect them. Monks cannot touch women.",
        "useful_phrases": ["Sawadee kap/ka (Hello)", "Khop khun kap/ka (Thank you)", "Mai pen rai (No problem)"],
    },
    "France": {
        "tipping": "Small tip appreciated - round up or leave 5-10%. Service is included in bills.",
        "greetings": "La bise (cheek kiss) among friends. Handshake in formal settings.",
        "dress": "French dress stylishly - avoid sportswear in restaurants.",
        "dining": "Say 'Bon appétit'. Bread goes directly on the table, not on a plate. Don't ask for a doggy bag.",
        "customs": "Greet shopkeepers when entering/leaving. Punctuality is respected.",
        "useful_phrases": ["Bonjour (Hello)", "Merci (Thank you)", "S'il vous plaît (Please)", "Excusez-moi (Excuse me)"],
    },
    "Italy": {
        "tipping": "Round up the bill or leave 1-2EUR. 'Coperto' (cover charge) is common.",
        "greetings": "Cheek kiss among friends. Firm handshake in business.",
        "dress": "Cover shoulders and knees for churches. Italians dress well.",
        "dining": "Cappuccino is only for morning. Dinner is late (8-9pm). Don't rush.",
        "customs": "Queue at bars - say 'al banco' for counter service. Gesturing is part of communication.",
        "useful_phrases": ["Grazie (Thank you)", "Prego (You're welcome/Please)", "Scusi (Excuse me)", "Per favore (Please)"],
    },
    "Dubai": {
        "tipping": "10-15% in restaurants; small tips for hotel staff.",
        "greetings": "Handshakes for men; wait for women to extend hand first.",
        "dress": "Modest dress in public. Swimwear only at beaches/pools. No public displays of affection.",
        "dining": "Alcohol only in licensed venues. Avoid eating/drinking in public during Ramadan daylight hours.",
        "customs": "Using left hand for giving/receiving is impolite. Photography of people requires consent. Public swearing can lead to fines.",
        "useful_phrases": ["Marhaba (Hello)", "Shukran (Thank you)", "Afwan (You're welcome)"],
    },
    "India": {
        "tipping": "10% in restaurants; small tips for guides and drivers expected.",
        "greetings": "Namaste (press palms together). Remove shoes before entering homes and temples.",
        "dress": "Conservative dress, especially women. Cover shoulders/knees at religious sites.",
        "dining": "Right hand only for eating. Vegetarian options widely available; beef is avoided by Hindus.",
        "customs": "Bargaining is normal in markets. Head wobble means 'yes'. Cows are sacred - do not disrespect them.",
        "useful_phrases": ["Namaste (Hello/Thank you)", "Shukriya (Thank you in Hindi)", "Theek hai (OK)"],
    },
    "Germany": {
        "tipping": "Round up or add 5-10%. Common to pay each person separately.",
        "greetings": "Firm handshake. Use formal 'Sie' with strangers until invited to use 'du'.",
        "dress": "Smart-casual. Shoes must be removed in homes.",
        "dining": "Say 'Guten Appetit'. It is fine to split bills individually ('Getrennt zahlen').",
        "customs": "Punctuality is very important. Jaywalking is frowned upon. Recycling is taken seriously.",
        "useful_phrases": ["Danke (Thank you)", "Bitte (Please/You're welcome)", "Entschuldigung (Excuse me)"],
    },
    "Spain": {
        "tipping": "Optional; round up or leave small change. Not mandatory.",
        "greetings": "Two cheek kisses even among new acquaintances.",
        "dress": "Casual but stylish. Cover up for churches.",
        "dining": "Lunch (2-4pm) is the main meal. Dinner is very late (9-11pm). Tapas sharing is the norm.",
        "customs": "Siesta time (2-5pm) many businesses close. Very social culture; noise is part of life.",
        "useful_phrases": ["Gracias (Thank you)", "Por favor (Please)", "De nada (You're welcome)", "Hola (Hello)"],
    },
    "Morocco": {
        "tipping": "Expected - 10-15% in restaurants; guides and drivers expect tips.",
        "greetings": "Salam alaikum (Peace be upon you). Handshake; avoid touching the opposite gender.",
        "dress": "Modest dress essential, especially for women. Shoulders and knees covered.",
        "dining": "Right hand for eating. Accepting mint tea is polite.",
        "customs": "Bargaining in souks is expected. Photography of people requires permission. Ramadan observance respected.",
        "useful_phrases": ["Shukran (Thank you)", "Afwan (You're welcome)", "La (No)", "Naam (Yes)"],
    },
    "Israel": {
        "tipping": "10-15% in restaurants; taxis: round up.",
        "greetings": "Handshake; casual 'Shalom' or 'Ahlan'.",
        "dress": "Modest dress at religious sites (Western Wall, etc.).",
        "dining": "Kosher restaurants don't mix meat and dairy. Shabbat (Friday sunset-Saturday night) many places close.",
        "customs": "Security checks are common and thorough. Friday afternoon things shut early for Shabbat.",
        "useful_phrases": ["Shalom (Hello/Goodbye/Peace)", "Toda (Thank you)", "Bevakasha (Please/You're welcome)"],
    },
    "South Korea": {
        "tipping": "Not expected and can be refused.",
        "greetings": "Slight bow. Older people are greeted first.",
        "dress": "Conservative; avoid revealing clothing. Smart in Seoul.",
        "dining": "Elders eat first. Share dishes communally. Pouring drinks for others is polite.",
        "customs": "Remove shoes at homes. Accept things with two hands or right hand supported by left. Respect for elders is paramount.",
        "useful_phrases": ["Annyeonghaseyo (Hello)", "Gamsahamnida (Thank you)", "Juseyo (Please give me)"],
    },
    "China": {
        "tipping": "Not traditional; can be refused. Some tourist areas now accept tips.",
        "greetings": "Nod or slight bow. Handshakes becoming common.",
        "dress": "Smart casual. Modest at temples.",
        "dining": "Trying all food offered is polite. Slurping noodles is acceptable.",
        "customs": "VPN needed for Google/social media. Cash still widely used. Queue jumping is common.",
        "useful_phrases": ["Nǐ hǎo (Hello)", "Xièxiè (Thank you)", "Qǐng (Please)"],
    },
}


@tool
def get_local_customs(destination: str) -> dict:
    """Get local customs, etiquette, tipping norms, and useful phrases for a destination.

    Use this when the user asks:
    - 'What are the local customs in [city/country]?'
    - 'Should I tip in [destination]?'
    - 'What are the etiquette rules in [destination]?'
    - 'What phrases should I learn before visiting [destination]?'
    """
    dest = validate_city(destination)

    info = _CUSTOMS_DB.get(dest)
    if not info:
        city_to_country = {
            "Tokyo": "Japan", "Kyoto": "Japan", "Osaka": "Japan",
            "Bangkok": "Thailand", "Phuket": "Thailand", "Chiang Mai": "Thailand",
            "Paris": "France", "Lyon": "France", "Nice": "France", "Marseille": "France",
            "Rome": "Italy", "Milan": "Italy", "Venice": "Italy", "Florence": "Italy",
            "Barcelona": "Spain", "Madrid": "Spain", "Seville": "Spain",
            "Berlin": "Germany", "Munich": "Germany", "Hamburg": "Germany", "Frankfurt": "Germany",
            "Marrakech": "Morocco", "Casablanca": "Morocco", "Fez": "Morocco",
            "Tel Aviv": "Israel", "Jerusalem": "Israel", "Haifa": "Israel",
            "Seoul": "South Korea", "Busan": "South Korea",
            "Beijing": "China", "Shanghai": "China", "Hong Kong": "China",
            "Mumbai": "India", "Delhi": "India", "New Delhi": "India", "Goa": "India", "Jaipur": "India",
        }
        country = city_to_country.get(dest)
        if country:
            info = _CUSTOMS_DB.get(country)

    if info:
        return {"destination": dest, **info}

    return {
        "destination": dest,
        "note": "No specific customs data available for this destination.",
        "general_tips": [
            "Research local dress codes, especially for religious sites.",
            "Learn a few basic words in the local language - it goes a long way.",
            "Observe how locals behave in public and follow their lead.",
            "Check whether tipping is expected or considered rude.",
            "Be aware of local laws (photography restrictions, alcohol rules, etc.).",
        ],
    }


# ---------------------------------------------------------------------------
# Wikipedia Summary
# ---------------------------------------------------------------------------

@tool
def get_wikipedia_summary(topic: str) -> str:
    """Fetch a short, factual overview of any travel-related topic from Wikipedia.

    Use this when the user asks about:
    - A specific city or country ("tell me about Kyoto", "what is Amsterdam known for?")
    - A specific attraction or landmark ("what is the Louvre?", "tell me about the Colosseum")
    - A historical site, museum, or natural wonder ("what is Machu Picchu?")

    Do NOT use for questions about flights, hotels, prices, or trip planning.
    """
    formatted = topic.strip().replace(" ", "_")
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{formatted}"

    try:
        response = requests.get(url, timeout=5, headers={"User-Agent": "navigation-ai/1.0"})
        if response.status_code == 200:
            data = response.json()
            return json.dumps({
                "topic": topic,
                "summary": data.get("extract", "No summary available."),
                "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
            })
        if response.status_code == 404:
            return json.dumps({
                "error": f"No Wikipedia page found for '{topic}'. Try a more specific name or different spelling."
            })
        return json.dumps({"error": f"Wikipedia API returned status {response.status_code}"})
    except requests.exceptions.RequestException as e:
        return json.dumps({"error": f"Request failed: {e}"})


@tool
def find_destinations_by_vibe(category: str) -> list[dict]:
    """Find cities that have activities matching a given category.

    Takes ONE argument only: category (str). Do NOT pass a city parameter.
    Available categories: Culture, Entertainment, Family, History, Nature, Nightlife, Sightseeing.
    Use this to answer vibe-based questions like 'where should I go for nightlife?'
    or 'I want a nature-focused trip, where should I go?'
    Example: find_destinations_by_vibe(category="Culture")
    """
    category = category.strip()
    return _provider.search_by_activity_category(category)


@tool
def get_reachable_destinations(origin: str, max_flight_hours: float | None = None) -> list[dict]:
    """Get all destinations with available flights from a given origin city.

    Use max_flight_hours to filter by actual flight duration:
    - Short haul (under ~2 hours):  max_flight_hours=2
    - Medium haul (under ~5 hours): max_flight_hours=5
    - Long haul (under ~8 hours):   max_flight_hours=8
    Leave max_flight_hours as None to return all reachable destinations.
    Returns city, country, price range, and shortest available flight duration.
    """
    origin = validate_city(origin)
    if max_flight_hours is not None:
        validate_positive_number(max_flight_hours, "max_flight_hours")
    return _provider.get_all_destinations_from_origin(origin, max_flight_hours)


@tool
def find_destinations_by_tag(tag: str) -> list[dict]:
    """Find cities that match a given travel style or vibe tag.

    Available tags: alternative, art, beach, budget-friendly, canals, city-break,
    cultural, cycling, entertainment, expensive, fashion, foodie, historic, iconic,
    liberal, luxury, mediterranean, modern, multicultural, nature-nearby, nightlife,
    rainy, romantic, safe, shopping, student-friendly, sunny, technology, theatre,
    unique, walkable.

    Use this when the user describes a vibe, atmosphere, or lifestyle preference
    (e.g. 'romantic', 'beach', 'budget-friendly', 'foodie').
    Prefer find_destinations_by_tag over find_destinations_by_vibe for style/atmosphere queries;
    use find_destinations_by_vibe for activity-category queries (Nature, Culture, Family, etc.).
    """
    tag = tag.strip().lower()
    return _provider.search_by_tag(tag)


@tool
def find_destinations_within_budget_auto(origin: str, total_budget: float) -> list[dict]:
    """Find destinations affordable from a given origin within a total budget, using each city's recommended stay length.

    Unlike find_destinations_within_budget, this tool does NOT require trip_days.
    It automatically looks up the recommended minimum stay for each destination and uses that
    to calculate the minimum trip cost (cheapest_flight + cheapest_hotel × recommended_days).
    Returns recommended_days in each result so the user can be informed of the assumed duration.

    Use this when the user mentions a budget and an origin but does NOT specify trip duration.
    """
    origin = validate_city(origin)
    validate_positive_number(total_budget, "total_budget")
    return _provider.find_within_budget_auto_duration(origin, total_budget)


@tool
def find_destinations_within_budget(origin: str, total_budget: float, trip_days: int) -> list[dict]:
    """Find destinations affordable from a given origin within a total budget for a given number of days.

    Calculates the minimum possible trip cost as: cheapest available flight + (cheapest hotel per night × days).
    Returns only destinations where this minimum total is within the budget, ordered cheapest first.
    Each result includes the cheapest flight price, cheapest hotel per night, estimated minimum total,
    and shortest available flight duration.

    Use this whenever the user mentions a budget alongside an origin, e.g.:
    'I have $1000 and I'm flying from Tel Aviv for 5 days — where can I go?'
    """
    origin = validate_city(origin)
    validate_positive_number(total_budget, "total_budget")
    validate_positive_number(trip_days, "trip_days")
    return _provider.find_within_budget(origin, total_budget, trip_days)


@tool
def get_trip_duration_advisor(city: str | list[str]) -> dict | list[dict]:
    """Get advisor info on how many days to spend in one or more cities.

    Pass a single city name for a single-city question, or a list of city names
    to fetch all durations in one call (e.g. ["Rome", "Amsterdam"] for a comparison query).

    Returns the suggested minimum and maximum number of days, with an explanation
    of what each range covers.
    Use this when the user asks 'how many days should I spend in X?',
    'is N days enough for X?', or wants help splitting time across multiple cities.
    """
    if isinstance(city, list):
        cities = [validate_city(c) for c in city]
        return _provider.get_recommended_duration(cities)
    return _provider.get_recommended_duration(validate_city(city))


@tool
def get_city_overview(city: str) -> dict:
    """Get a rich profile of a destination city.

    Returns:
    - Activity categories available and sample activity names
    - Best months to visit and the reason why
    - Average temperature for each season (Spring, Summer, Autumn, Winter)

    Use this to answer questions like 'what kind of trip is Tokyo?',
    'when should I visit Paris?', or 'what is there to do in Amsterdam?'
    """
    city = validate_city(city)
    return _provider.get_city_profile(city)


advisor_tools = [
    find_destinations_by_vibe,
    find_destinations_by_tag,
    get_reachable_destinations,
    find_destinations_within_budget_auto,
    find_destinations_within_budget,
    get_city_overview,
    get_trip_duration_advisor,
    fetch_activities,
    get_best_time_to_visit,
    get_average_weather,
    get_currency_exchange,
    get_visa_requirements,
    get_travel_safety_info,
    get_packing_list,
    get_local_customs,
    get_wikipedia_summary,
    search_concerts,
]
