"""Probe Travelpayouts endpoints to decide which SQLite-backed tools we can
swap for live data.

Run:
    tavenv/bin/python tests/test_travelpayouts.py

Findings (verified on 2026-05-23 with the user's TRAVELPAYOUTS_API_KEY):

  FLIGHT tools — REPLACEABLE with live Travelpayouts data:
    fetch_flights              -> GET aviasales/v3/prices_for_dates
    find_connecting_flights    -> same endpoint; filter by `transfers` field
    get_flight_filter_options  -> derive airlines/price range from prices feed
    (also useful) get_latest_prices -> v2/prices/latest

  HOTEL tools — NOT REPLACEABLE with this token.
    The documented Hotellook engine (engine.hotellook.com, yasen.hotellook.com)
    now returns 404 on EVERY documented path — including their own doc examples
    (lookup.json, cache.json, search/start.json, static/*). The public engine
    appears to be retired; the remaining hotels API requires a partner
    `marker` (affiliate ID) rather than a bare data-API token.

  ACTIVITIES / WEATHER / BEST-TIME-TO-VISIT — not covered by Travelpayouts at all.
"""
import json
import os
import sys
from pathlib import Path
from datetime import date, timedelta

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

API_KEY = os.getenv("TRAVELPAYOUTS_API_KEY") or os.getenv("TRAVELPAYOUT_API_KEY")
if not API_KEY:
    print("ERROR: TRAVELPAYOUTS_API_KEY not found in .env")
    sys.exit(1)

print(f"Using API key: {API_KEY[:6]}…{API_KEY[-4:]}\n")


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def show(resp: requests.Response, limit: int = 2) -> None:
    print(f"  HTTP {resp.status_code}  ({len(resp.content)} bytes)")
    if resp.status_code != 200:
        print(f"  Body: {resp.text[:300]}")
        return
    try:
        data = resp.json()
    except ValueError:
        print(f"  Non-JSON body: {resp.text[:200]}")
        return
    if isinstance(data, list):
        print(f"  Returned {len(data)} items. First {min(limit, len(data))}:")
        for item in data[:limit]:
            print("    " + json.dumps(item, indent=2)[:400].replace("\n", "\n    "))
    elif isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            print(f"  data: {len(data['data'])} items. First {min(limit, len(data['data']))}:")
            for item in data["data"][:limit]:
                print("    " + json.dumps(item, indent=2)[:400].replace("\n", "\n    "))
            print(f"  success={data.get('success')}, currency={data.get('currency')}")
        else:
            text = json.dumps(data, indent=2)
            print("  " + text[:800].replace("\n", "\n  "))


def city_to_iata(city: str) -> str | None:
    """Travelpayouts autocomplete: city name → IATA code."""
    r = requests.get(
        "https://autocomplete.travelpayouts.com/places2",
        params={"term": city, "locale": "en", "types[]": "city"},
        timeout=10,
    )
    if r.status_code != 200 or not r.json():
        return None
    return r.json()[0].get("code")


# ---------------------------------------------------------------------------
# 1. City → IATA (Travelpayouts uses IATA codes for flight queries)
# ---------------------------------------------------------------------------
section("1. Autocomplete: city name → IATA code")
for c in ("Paris", "Tel Aviv", "New York", "Tokyo", "Barcelona"):
    print(f"  {c:12s} -> {city_to_iata(c)}")

ORIGIN_CITY, DEST_CITY = "Tel Aviv", "Paris"
ORIGIN = city_to_iata(ORIGIN_CITY) or "TLV"
DEST = city_to_iata(DEST_CITY) or "PAR"
print(f"\n  Route used below: {ORIGIN_CITY}={ORIGIN} -> {DEST_CITY}={DEST}")

depart_month = (date.today() + timedelta(days=21)).strftime("%Y-%m")

# ---------------------------------------------------------------------------
# 2. Flights: prices_for_dates → replaces fetch_flights
# ---------------------------------------------------------------------------
section("2. Flights: prices_for_dates  (replaces fetch_flights)")
r = requests.get(
    "https://api.travelpayouts.com/aviasales/v3/prices_for_dates",
    params={
        "origin": ORIGIN,
        "destination": DEST,
        "departure_at": depart_month,
        "currency": "usd",
        "limit": 5,
        "sorting": "price",
        "token": API_KEY,
    },
    timeout=15,
)
show(r)

# ---------------------------------------------------------------------------
# 3. Flights: get_latest_prices (cached recent fares)
# ---------------------------------------------------------------------------
section("3. Flights: v2/prices/latest  (cached latest fares)")
r = requests.get(
    "https://api.travelpayouts.com/v2/prices/latest",
    params={"origin": ORIGIN, "destination": DEST, "currency": "usd", "limit": 5, "token": API_KEY},
    timeout=15,
)
show(r)

# ---------------------------------------------------------------------------
# 4. Connecting flights via the `transfers` field
# ---------------------------------------------------------------------------
section("4. Connecting flights via `transfers` (replaces find_connecting_flights)")
r = requests.get(
    "https://api.travelpayouts.com/aviasales/v3/prices_for_dates",
    params={
        "origin": ORIGIN,
        "destination": DEST,
        "departure_at": depart_month,
        "currency": "usd",
        "limit": 30,
        "sorting": "price",
        "one_way": "true",
        "token": API_KEY,
    },
    timeout=15,
)
if r.status_code == 200:
    items = r.json().get("data", [])
    by_stops: dict[int, list] = {}
    for it in items:
        by_stops.setdefault(it.get("transfers", 0), []).append(it)
    print(f"  Offers by stops: { {k: len(v) for k, v in sorted(by_stops.items())} }")
    multi = [it for it in items if (it.get("transfers") or 0) >= 1]
    if multi:
        print("  Cheapest multi-stop offer:")
        print("    " + json.dumps(multi[0], indent=2).replace("\n", "\n    "))
else:
    show(r)

# ---------------------------------------------------------------------------
# 5. Filter dimensions (replaces get_flight_filter_options)
# ---------------------------------------------------------------------------
section("5. Flight filter dimensions (airlines + price range)")
if r.status_code == 200:
    items = r.json().get("data", [])
    airlines = sorted({i.get("airline") for i in items if i.get("airline")})
    prices = [i.get("price") for i in items if i.get("price") is not None]
    print(f"  Airlines on route: {airlines}")
    print(f"  Price min/max:    {min(prices) if prices else None} / {max(prices) if prices else None}")

# ---------------------------------------------------------------------------
# 6. Hotels — verify the engine is dead
# ---------------------------------------------------------------------------
section("6. Hotels: confirm Hotellook engine is unreachable")
check_in = (date.today() + timedelta(days=21)).isoformat()
check_out = (date.today() + timedelta(days=24)).isoformat()
hotel_probes = [
    ("engine cache.json (https)",
        "https://engine.hotellook.com/api/v2/cache.json",
        {"location": DEST_CITY, "currency": "usd", "checkIn": check_in,
         "checkOut": check_out, "adults": 2, "limit": 5, "token": API_KEY}),
    ("engine cache.json (http→follow)",
        "http://engine.hotellook.com/api/v2/cache.json",
        {"location": DEST_CITY, "checkIn": check_in, "checkOut": check_out,
         "limit": 5, "token": API_KEY}),
    ("engine lookup.json (doc example)",
        "http://engine.hotellook.com/api/v2/lookup.json",
        {"query": "moscow", "lang": "ru", "lookFor": "both", "limit": 1, "token": API_KEY}),
    ("api.travelpayouts v1/hotels",
        "https://api.travelpayouts.com/v1/hotels",
        {"location": DEST_CITY, "token": API_KEY}),
    ("yasen widget_locations",
        "http://yasen.hotellook.com/tp/public/widget_locations.json",
        {"query": "paris", "language": "en", "limit": 3, "token": API_KEY}),
]
for label, url, params in hotel_probes:
    try:
        rr = requests.get(url, params=params, timeout=10)
        print(f"  {label:42s} -> HTTP {rr.status_code}  ({len(rr.content)}b)")
    except Exception as e:
        print(f"  {label:42s} -> EXC {e}")

# ---------------------------------------------------------------------------
# 7. Capability matrix
# ---------------------------------------------------------------------------
section("Replacement matrix — synthetic SQLite -> live Travelpayouts")
matrix = [
    ("fetch_flights",              "YES  ", "aviasales/v3/prices_for_dates"),
    ("find_connecting_flights",    "YES  ", "same endpoint; filter on `transfers >= 1`"),
    ("get_flight_filter_options",  "YES  ", "derive airlines & price range from prices feed"),
    ("fetch_hotels",               "NO   ", "Hotellook engine returns 404 on all documented paths"),
    ("get_hotel_filter_options",   "NO   ", "depends on fetch_hotels"),
    ("fetch_activities",           "NO   ", "Travelpayouts has no activities/POI data"),
    ("get_best_time_to_visit",     "NO   ", "Not covered (use a weather/seasonality API instead)"),
    ("get_average_weather",        "NO   ", "Not covered (use Open-Meteo or similar)"),
]
print(f"  {'tool':<28} {'replaceable':<12} backend")
print(f"  {'-'*28} {'-'*12} {'-'*48}")
for tool, ok, backend in matrix:
    print(f"  {tool:<28} {ok:<12} {backend}")

print("\nDONE.")
