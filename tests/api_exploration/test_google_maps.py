"""Probe Google Maps Platform endpoints to decide which SQLite-backed tools we
can swap for live data.

Run:
    tavenv/bin/python tests/test_google_maps.py

What we test (legacy stable endpoints — all take ?key=...):
  1. Geocoding              -> city -> lat/lng       (used by every place query)
  2. Places Text Search     -> "museums in Paris"    -> replaces fetch_activities
  3. Places Nearby Search   -> tourist attractions   -> replaces fetch_activities
  4. Place Details          -> hours, rating, phone  -> enrich activities
  5. Place Photo            -> photo_reference -> URL (UI only)
  6. Distance Matrix        -> trip duration/km      -> replaces calculate_trip cost legs
  7. Places Text Search for hotels -> partial replacement for fetch_hotels
     (Google gives ratings & price_level 0-4, NOT a per-night rate)

The new Places API (places.googleapis.com/v1/...) is more capable but requires
POST + FieldMask headers; we keep this probe on the legacy GET endpoints because
they're enough to prove what data is reachable.
"""
import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
if not API_KEY:
    print("ERROR: GOOGLE_MAPS_API_KEY not found in .env")
    sys.exit(1)
print(f"Using key: {API_KEY[:6]}…{API_KEY[-4:]}\n")

BASE = "https://maps.googleapis.com/maps/api"


def section(t: str) -> None:
    print("\n" + "=" * 70)
    print(f" {t}")
    print("=" * 70)


def get(url: str, **params) -> dict:
    r = requests.get(url, params={**params, "key": API_KEY}, timeout=15)
    print(f"  HTTP {r.status_code}  ({len(r.content)}b)")
    try:
        data = r.json()
    except ValueError:
        print(f"  Non-JSON: {r.text[:200]}")
        return {}
    status = data.get("status")
    if status and status != "OK":
        print(f"  status={status}  error_message={data.get('error_message')}")
    return data


# ---------------------------------------------------------------------------
# 1. Geocoding — city → coordinates
# ---------------------------------------------------------------------------
section("1. Geocoding: city name -> lat/lng")
geo: dict[str, tuple[float, float]] = {}
for city in ("Paris", "Tel Aviv", "Tokyo", "Barcelona"):
    d = get(f"{BASE}/geocode/json", address=city)
    if d.get("results"):
        loc = d["results"][0]["geometry"]["location"]
        geo[city] = (loc["lat"], loc["lng"])
        formatted = d["results"][0]["formatted_address"]
        print(f"  {city:10s} -> {loc['lat']:.4f},{loc['lng']:.4f}  ({formatted})")

PARIS = geo.get("Paris", (48.8566, 2.3522))

# ---------------------------------------------------------------------------
# 2. Places Text Search — "museums in Paris"
# ---------------------------------------------------------------------------
section("2. Places Text Search: 'museums in Paris' (replaces fetch_activities)")
d = get(f"{BASE}/place/textsearch/json", query="museums in Paris")
results = d.get("results", [])[:5]
print(f"  Got {len(d.get('results', []))} results. First 5:")
museum_place_ids = []
for p in results:
    pid = p.get("place_id")
    museum_place_ids.append(pid)
    print(f"    - {p.get('name'):<40s} rating={p.get('rating')}  price_level={p.get('price_level')}  open_now={p.get('opening_hours',{}).get('open_now')}")

# ---------------------------------------------------------------------------
# 3. Places Nearby Search — top tourist attractions near Paris center
# ---------------------------------------------------------------------------
section("3. Places Nearby Search: tourist_attraction near Paris (replaces fetch_activities)")
d = get(
    f"{BASE}/place/nearbysearch/json",
    location=f"{PARIS[0]},{PARIS[1]}",
    radius=5000,
    type="tourist_attraction",
)
nearby = d.get("results", [])[:5]
print(f"  Got {len(d.get('results', []))} results. First 5:")
for p in nearby:
    print(f"    - {p.get('name'):<40s} rating={p.get('rating')}  user_ratings_total={p.get('user_ratings_total')}")

# ---------------------------------------------------------------------------
# 4. Place Details — opening hours, phone, website for one museum
# ---------------------------------------------------------------------------
section("4. Place Details (enrich an activity row)")
if museum_place_ids:
    d = get(
        f"{BASE}/place/details/json",
        place_id=museum_place_ids[0],
        fields="name,formatted_address,formatted_phone_number,website,rating,opening_hours,price_level,editorial_summary,photos",
    )
    res = d.get("result", {})
    print(f"  Name:    {res.get('name')}")
    print(f"  Address: {res.get('formatted_address')}")
    print(f"  Phone:   {res.get('formatted_phone_number')}")
    print(f"  Website: {res.get('website')}")
    print(f"  Rating:  {res.get('rating')}   price_level={res.get('price_level')}")
    hours = (res.get("opening_hours") or {}).get("weekday_text") or []
    if hours:
        print("  Hours:")
        for line in hours:
            print(f"    {line}")
    photos = res.get("photos") or []
    print(f"  Photos available: {len(photos)}")
    if photos:
        ref = photos[0]["photo_reference"]
        photo_url = f"{BASE}/place/photo?maxwidth=400&photo_reference={ref}&key={API_KEY}"
        print(f"  Example photo URL (HEAD-checked below): {photo_url[:120]}…")
        head = requests.head(photo_url, allow_redirects=False, timeout=10)
        print(f"  HEAD -> {head.status_code}  Location: {head.headers.get('Location','')[:120]}…")

# ---------------------------------------------------------------------------
# 5. Distance Matrix — Tel Aviv -> Paris (driving), and Paris -> Versailles
# ---------------------------------------------------------------------------
section("5. Distance Matrix (driving distance & duration)")
d = get(
    f"{BASE}/distancematrix/json",
    origins="Paris",
    destinations="Versailles, France|Louvre Museum, Paris|Eiffel Tower, Paris",
    mode="driving",
)
rows = d.get("rows", [])
dests = d.get("destination_addresses", [])
if rows:
    for dest, el in zip(dests, rows[0].get("elements", [])):
        if el.get("status") == "OK":
            print(f"  Paris -> {dest[:60]:<60s} {el['distance']['text']:>10s}  {el['duration']['text']}")
        else:
            print(f"  Paris -> {dest}  status={el.get('status')}")

# ---------------------------------------------------------------------------
# 6. Places Text Search for hotels (partial replacement for fetch_hotels)
# ---------------------------------------------------------------------------
section("6. Places Text Search: 'hotels in Paris' (partial fetch_hotels)")
d = get(f"{BASE}/place/textsearch/json", query="hotels in Paris")
results = d.get("results", [])[:5]
print(f"  Got {len(d.get('results', []))} results. First 5:")
for p in results:
    print(f"    - {p.get('name'):<40s} rating={p.get('rating')}  price_level={p.get('price_level')}  ratings={p.get('user_ratings_total')}")
print("  NOTE: Google gives `price_level` 0-4 (a tier), NOT a per-night price.")

# ---------------------------------------------------------------------------
# Capability matrix
# ---------------------------------------------------------------------------
section("Replacement matrix — synthetic SQLite -> Google Maps Platform")
matrix = [
    ("fetch_activities",        "YES  ", "Places Text Search / Nearby Search + Place Details for hours, rating, website, photos"),
    ("fetch_hotels",            "PART ", "Places Text Search gives names/ratings/price_level (1-4), NOT nightly USD"),
    ("get_hotel_filter_options","PART ", "Aggregate price_level + ratings from a Places search"),
    ("fetch_flights",           "NO   ", "Google Maps has no flight pricing — keep Travelpayouts for this"),
    ("get_best_time_to_visit",  "NO   ", "Not in Maps Platform"),
    ("get_average_weather",     "NO   ", "Not in Maps Platform — use Open-Meteo or similar"),
    ("calculate_trip_cost",     "BONUS", "Distance Matrix can give per-leg km + duration for budgeting"),
    ("(new) city -> lat/lng",   "YES  ", "Geocoding — useful glue everywhere"),
]
print(f"  {'tool':<28} {'replaceable':<12} backend")
print(f"  {'-'*28} {'-'*12} {'-'*60}")
for tool, ok, backend in matrix:
    print(f"  {tool:<28} {ok:<12} {backend}")
print("\nDONE.")
