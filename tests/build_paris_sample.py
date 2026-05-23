"""Pull a Paris travel sample using the two live APIs and write the result to
disk so you can eyeball what's reachable.

Outputs:
    out/paris_sample.json   - structured data, ready for routing/planning
    out/paris_sample.md     - human-readable summary

What's in it:
    1. Flights on 2026-06-10  (Travelpayouts, TLV -> PAR)
    2. Hotels in Paris        (Google Places, ratings + price tier)
    3. Attractions in Paris   (Google Places, ratings + lat/lng + price tier)
    4. Restaurants in Paris   (Google Places, ratings + lat/lng + price tier)

Run:
    tavenv/bin/python tests/build_paris_sample.py
"""
import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

TP_KEY = os.getenv("TRAVELPAYOUTS_API_KEY")
GM_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
if not TP_KEY or not GM_KEY:
    print("Missing TRAVELPAYOUTS_API_KEY or GOOGLE_MAPS_API_KEY in .env")
    sys.exit(1)

OUT_DIR = ROOT / "out"
OUT_DIR.mkdir(exist_ok=True)

PRICE_LEVEL = {
    0: "Free",
    1: "$ – inexpensive",
    2: "$$ – moderate",
    3: "$$$ – expensive",
    4: "$$$$ – very expensive",
}


# ---------------------------------------------------------------------------
# 1. Flights — TLV -> PAR on 2026-06-10
# ---------------------------------------------------------------------------
def fetch_flights(origin: str, destination: str, depart: str) -> list[dict]:
    r = requests.get(
        "https://api.travelpayouts.com/aviasales/v3/prices_for_dates",
        params={
            "origin": origin,
            "destination": destination,
            "departure_at": depart,  # exact-day filter
            "currency": "usd",
            "limit": 20,
            "sorting": "price",
            "one_way": "true",
            "token": TP_KEY,
        },
        timeout=20,
    )
    r.raise_for_status()
    items = r.json().get("data", [])
    return [
        {
            "airline": it.get("airline"),
            "flight_number": it.get("flight_number"),
            "price_usd": it.get("price"),
            "departure_at": it.get("departure_at"),
            "origin_airport": it.get("origin_airport"),
            "destination_airport": it.get("destination_airport"),
            "duration_minutes": it.get("duration"),
            "transfers": it.get("transfers"),
            "seller": it.get("gate"),
            "booking_link": "https://www.aviasales.com" + it.get("link", ""),
        }
        for it in items
    ]


# ---------------------------------------------------------------------------
# Google Places helpers
# ---------------------------------------------------------------------------
BASE = "https://maps.googleapis.com/maps/api"


def geocode(address: str) -> tuple[float, float, str]:
    r = requests.get(f"{BASE}/geocode/json", params={"address": address, "key": GM_KEY}, timeout=10)
    r.raise_for_status()
    res = r.json()["results"][0]
    loc = res["geometry"]["location"]
    return loc["lat"], loc["lng"], res["formatted_address"]


def places_text_search(query: str, limit: int = 10) -> list[dict]:
    r = requests.get(
        f"{BASE}/place/textsearch/json", params={"query": query, "key": GM_KEY}, timeout=15
    )
    r.raise_for_status()
    return [_simplify_place(p) for p in r.json().get("results", [])[:limit]]


def places_nearby(lat: float, lng: float, kind: str, radius: int = 4000, limit: int = 10) -> list[dict]:
    r = requests.get(
        f"{BASE}/place/nearbysearch/json",
        params={
            "location": f"{lat},{lng}",
            "radius": radius,
            "type": kind,
            "key": GM_KEY,
        },
        timeout=15,
    )
    r.raise_for_status()
    items = r.json().get("results", [])
    # Prefer well-known places (more reviews) and best-rated.
    items.sort(
        key=lambda p: ((p.get("user_ratings_total") or 0), (p.get("rating") or 0)),
        reverse=True,
    )
    return [_simplify_place(p) for p in items[:limit]]


# New Places API (v1) — POST with FieldMask. Required for `priceRange` (real
# currency amounts) which the legacy endpoints don't return.
PRICE_LEVEL_V1 = {
    "PRICE_LEVEL_FREE": (0, "Free"),
    "PRICE_LEVEL_INEXPENSIVE": (1, "$ – inexpensive"),
    "PRICE_LEVEL_MODERATE": (2, "$$ – moderate"),
    "PRICE_LEVEL_EXPENSIVE": (3, "$$$ – expensive"),
    "PRICE_LEVEL_VERY_EXPENSIVE": (4, "$$$$ – very expensive"),
}


def places_v1_text_search(query: str, limit: int = 15, drop_types: set[str] = frozenset()) -> list[dict]:
    fields = ",".join([
        "places.displayName",
        "places.id",
        "places.formattedAddress",
        "places.location",
        "places.rating",
        "places.userRatingCount",
        "places.priceLevel",
        "places.priceRange",
        "places.types",
        "places.regularOpeningHours.openNow",
        "places.editorialSummary",
    ])
    r = requests.post(
        "https://places.googleapis.com/v1/places:searchText",
        json={"textQuery": query, "maxResultCount": 20},
        headers={
            "Content-Type": "application/json",
            "X-Goog-Api-Key": GM_KEY,
            "X-Goog-FieldMask": fields,
        },
        timeout=20,
    )
    r.raise_for_status()
    out: list[dict] = []
    for p in r.json().get("places", []):
        types = p.get("types", [])
        if drop_types & set(types):
            continue
        out.append(_simplify_place_v1(p))
        if len(out) >= limit:
            break
    return out


def _simplify_place_v1(p: dict) -> dict:
    loc = p.get("location") or {}
    pl_enum = p.get("priceLevel")
    pl_int, pl_label = PRICE_LEVEL_V1.get(pl_enum, (None, None))
    pr = p.get("priceRange") or {}
    sp = pr.get("startPrice") or {}
    ep = pr.get("endPrice") or {}

    def fmt(side: dict) -> str | None:
        if not side:
            return None
        units = side.get("units")
        curr = side.get("currencyCode")
        return f"{units} {curr}" if units and curr else None

    start = fmt(sp)
    end = fmt(ep)
    price_range_text = None
    if start and end:
        price_range_text = f"{start} – {end}"
    elif start:
        price_range_text = f"from {start}"
    elif end:
        price_range_text = f"up to {end}"

    pid = p.get("id")
    return {
        "name": (p.get("displayName") or {}).get("text"),
        "place_id": pid,
        "address": p.get("formattedAddress"),
        "rating": p.get("rating"),
        "review_count": p.get("userRatingCount"),
        "price_level": pl_int,
        "price_tier": pl_label,
        "price_range_text": price_range_text,
        "price_range_raw": pr or None,
        "open_now": (p.get("regularOpeningHours") or {}).get("openNow"),
        "lat": loc.get("latitude"),
        "lng": loc.get("longitude"),
        "google_maps_url": f"https://www.google.com/maps/place/?q=place_id:{pid}",
        "types": p.get("types", []),
        "editorial_summary": (p.get("editorialSummary") or {}).get("text"),
    }


def _simplify_place(p: dict) -> dict:
    loc = (p.get("geometry") or {}).get("location") or {}
    pl = p.get("price_level")
    return {
        "name": p.get("name"),
        "place_id": p.get("place_id"),
        "address": p.get("formatted_address") or p.get("vicinity"),
        "rating": p.get("rating"),
        "review_count": p.get("user_ratings_total"),
        "price_level": pl,
        "price_tier": PRICE_LEVEL.get(pl) if pl is not None else None,
        "price_range_text": None,
        "open_now": (p.get("opening_hours") or {}).get("open_now"),
        "lat": loc.get("lat"),
        "lng": loc.get("lng"),
        "google_maps_url": f"https://www.google.com/maps/place/?q=place_id:{p.get('place_id')}",
        "types": p.get("types", []),
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
print("Fetching flights TLV -> PAR on 2026-06-10 …")
flights = fetch_flights("TLV", "PAR", "2026-06-10")

print("Geocoding Paris …")
lat, lng, paris_addr = geocode("Paris, France")

print("Fetching hotels …")
hotels = places_text_search("hotels in Paris", limit=10)

print("Fetching attractions near Paris …")
attractions = places_nearby(lat, lng, "tourist_attraction", radius=4500, limit=10)

print("Fetching restaurants near Paris (new Places API for priceRange) …")
restaurants = places_v1_text_search(
    "restaurants in Paris",
    limit=15,
    drop_types={"lodging"},  # filter out hotels-with-restaurants
)

sample = {
    "generated_for": "2026-06-10 trip to Paris",
    "paris_center": {"lat": lat, "lng": lng, "address": paris_addr},
    "flights_TLV_PAR_2026_06_10": flights,
    "hotels_in_paris": hotels,
    "attractions_in_paris": attractions,
    "restaurants_in_paris": restaurants,
}

json_path = OUT_DIR / "paris_sample.json"
json_path.write_text(json.dumps(sample, indent=2, ensure_ascii=False))
print(f"\nWrote {json_path}  ({json_path.stat().st_size:,} bytes)")


# ---------------------------------------------------------------------------
# Pretty markdown summary
# ---------------------------------------------------------------------------
def fmt_minutes(m):
    if not m:
        return "?"
    h, mm = divmod(int(m), 60)
    return f"{h}h{mm:02d}"


lines: list[str] = []
lines.append(f"# Paris sample — 2026-06-10\n")
lines.append(f"Paris center geocoded to **{lat:.4f}, {lng:.4f}** ({paris_addr}).\n")

lines.append("\n## Flights TLV → PAR on 2026-06-10  (Travelpayouts, sorted by price)\n")
lines.append("| Price USD | Airline | Flight | Depart | Stops | Duration | Route | Seller |")
lines.append("|---:|---|---|---|---:|---|---|---|")
for f in flights:
    lines.append(
        f"| ${f['price_usd']} | {f['airline']} | {f['flight_number']} | "
        f"{f['departure_at'][:16].replace('T', ' ')} | {f['transfers']} | "
        f"{fmt_minutes(f['duration_minutes'])} | {f['origin_airport']}→{f['destination_airport']} | {f['seller']} |"
    )
if not flights:
    lines.append("| _no offers found for that date_ | | | | | | | |")


def place_table(title: str, items: list[dict], with_range: bool = False) -> None:
    lines.append(f"\n## {title}\n")
    if with_range:
        lines.append("| Name | Rating ★ | Reviews | Price tier | Price range | Lat | Lng | Map |")
        lines.append("|---|---:|---:|---|---|---:|---:|---|")
        for p in items:
            lines.append(
                f"| {p['name']} | {p['rating']} | {p['review_count']} | "
                f"{p['price_tier'] or '—'} | {p.get('price_range_text') or '—'} | "
                f"{p['lat']:.4f} | {p['lng']:.4f} | "
                f"[map]({p['google_maps_url']}) |"
            )
    else:
        lines.append("| Name | Rating ★ | Reviews | Price tier | Lat | Lng | Map |")
        lines.append("|---|---:|---:|---|---:|---:|---|")
        for p in items:
            lines.append(
                f"| {p['name']} | {p['rating']} | {p['review_count']} | "
                f"{p['price_tier'] or '—'} | "
                f"{p['lat']:.4f} | {p['lng']:.4f} | "
                f"[map]({p['google_maps_url']}) |"
            )


place_table("Hotels in Paris  (Google Places — Text Search)", hotels)
place_table("Top tourist attractions near Paris center  (Nearby Search)", attractions)
place_table(
    "Restaurants in Paris  (new Places API v1 — `priceRange` in EUR)",
    restaurants,
    with_range=True,
)

lines.append(
    "\n_Note:_ Google's `price_tier` is a 0–4 bucket the place got from review data — "
    "it's an indicator, not an actual nightly/per-meal USD. Hotels often have it blank.\n"
)

md_path = OUT_DIR / "paris_sample.md"
md_path.write_text("\n".join(lines))
print(f"Wrote {md_path}  ({md_path.stat().st_size:,} bytes)\n")

print("Quick stats:")
print(f"  flights:     {len(flights)}")
print(f"  hotels:      {len(hotels)}")
print(f"  attractions: {len(attractions)}")
print(f"  restaurants: {len(restaurants)}")
