"""Probe many Travelpayouts flight endpoints + date scopes for TLV → PAR.

Goal: figure out why exact-day searches return so few (or weird) offers and
which endpoint actually gives a usable answer for travel planning.

We try:
  - aviasales/v3/prices_for_dates    (exact day, month)
  - aviasales/v3/grouped_prices      (grouped by date)
  - aviasales/v3/get_latest_prices   (cached)
  - v2/prices/latest                 (older cached feed)
  - v2/prices/month-matrix           (per-day cheapest for a whole month)
  - v1/prices/cheap                  (cheapest per gate)
  - v1/prices/calendar               (calendar of cheap fares)
  - v1/prices/direct                 (direct only)

We also try `direct=true` on the dates endpoint and decode airline codes
against the public airlines.json so "IZ" actually has a name.
"""
import json
import os
import sys
import time
from pathlib import Path
from datetime import date, timedelta

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

KEY = os.getenv("TRAVELPAYOUTS_API_KEY")
if not KEY:
    print("Missing TRAVELPAYOUTS_API_KEY")
    sys.exit(1)

ORIGIN, DEST = "TLV", "PAR"
DEPART_DAY = "2026-06-10"
DEPART_MONTH = "2026-06"


# ---------------------------------------------------------------------------
# Airline-code lookup (so we know what "IZ" is)
# ---------------------------------------------------------------------------
def airline_names() -> dict[str, str]:
    r = requests.get("https://api.travelpayouts.com/data/en/airlines.json", timeout=15)
    r.raise_for_status()
    out: dict[str, str] = {}
    for a in r.json():
        code = a.get("iata") or a.get("code")
        if code:
            out[code] = a.get("name") or code
    return out


AIRLINES = airline_names()
print(f"Loaded {len(AIRLINES)} airline names\n")


# ---------------------------------------------------------------------------
# Endpoint wrappers
# ---------------------------------------------------------------------------
def call(url: str, params: dict, label: str) -> tuple[int, dict, float]:
    t = time.perf_counter()
    r = requests.get(url, params={**params, "token": KEY}, timeout=20)
    ms = (time.perf_counter() - t) * 1000
    try:
        body = r.json()
    except ValueError:
        body = {"_text": r.text[:300]}
    print(f"  [{ms:5.0f}ms] {label:<60s} HTTP {r.status_code}")
    return r.status_code, body, ms


def summarise_v3(body: dict, label: str) -> None:
    items = body.get("data") or []
    if not items:
        print(f"    -> 0 items.  success={body.get('success')}  error={body.get('error')}")
        return
    prices = [it.get("price") for it in items if it.get("price") is not None]
    airlines = sorted({it.get("airline") for it in items if it.get("airline")})
    transfers = sorted({it.get("transfers") for it in items if it.get("transfers") is not None})
    dur = [it.get("duration") for it in items if it.get("duration")]
    print(f"    -> {len(items)} offers  price ${min(prices)}-${max(prices)}  stops={transfers}  duration={min(dur)}-{max(dur)}min")
    print(f"       airlines: {', '.join(f'{a} ({AIRLINES.get(a, a)})' for a in airlines)}")
    cheapest = min(items, key=lambda x: x.get("price", 1e9))
    print(
        f"       cheapest: ${cheapest['price']}  {cheapest.get('airline')} "
        f"{cheapest.get('flight_number')}  {cheapest.get('departure_at','')[:16]}  "
        f"stops={cheapest.get('transfers')}  dur={cheapest.get('duration')}min  "
        f"gate={cheapest.get('gate')}"
    )


def summarise_v2_latest(body: dict, label: str) -> None:
    items = body.get("data") or []
    if not items:
        print(f"    -> 0 items.  success={body.get('success')}")
        return
    prices = [it.get("value") for it in items if it.get("value") is not None]
    gates = sorted({it.get("gate") for it in items if it.get("gate")})
    print(f"    -> {len(items)} offers  price ${min(prices)}-${max(prices)}  gates={gates[:6]}")
    cheapest = min(items, key=lambda x: x.get("value", 1e9))
    print(
        f"       cheapest: ${cheapest['value']}  changes={cheapest.get('number_of_changes')}  "
        f"depart={cheapest.get('depart_date')}  return={cheapest.get('return_date')}  "
        f"gate={cheapest.get('gate')}  found={cheapest.get('found_at','')[:16]}"
    )


def summarise_v2_cheap(body: dict, label: str) -> None:
    data = body.get("data") or {}
    if not data:
        print(f"    -> empty data block.  success={body.get('success')}  error={body.get('error')}")
        return
    n = 0
    for dest, by_stop in data.items():
        for stops, offer in by_stop.items():
            n += 1
            if n <= 3:
                print(
                    f"       {dest} stops={stops}: ${offer.get('price')}  {offer.get('airline')}  "
                    f"flight={offer.get('flight_number')}  depart={offer.get('departure_at','')[:16]}  "
                    f"return={offer.get('return_at','')[:16]}"
                )
    print(f"    -> {n} offer rows across {len(data)} destinations")


def summarise_calendar(body: dict) -> None:
    data = body.get("data") or {}
    if not data:
        print(f"    -> empty.  success={body.get('success')}  error={body.get('error')}")
        return
    days = sorted(data.keys())
    cheap = min(data.values(), key=lambda x: x.get("price", 1e9))
    print(f"    -> {len(days)} day buckets, cheapest in window:")
    print(
        f"       ${cheap.get('price')}  depart {cheap.get('departure_at','')[:10]}  "
        f"airline={cheap.get('airline')}  stops={cheap.get('transfers')}  gate={cheap.get('gate')}"
    )
    # Show the spread for the first 12 days
    sample = [(d, data[d].get("price")) for d in days[:12]]
    print("       day -> price (first 12 buckets):  " + ", ".join(f"{d[5:]}=${p}" for d, p in sample))


# ---------------------------------------------------------------------------
# 1. Exact day, with and without direct=true
# ---------------------------------------------------------------------------
print("=" * 75)
print(f" 1. aviasales/v3/prices_for_dates  exact day {DEPART_DAY}")
print("=" * 75)
for direct in ("false", "true"):
    s, b, _ = call(
        "https://api.travelpayouts.com/aviasales/v3/prices_for_dates",
        {
            "origin": ORIGIN, "destination": DEST,
            "departure_at": DEPART_DAY, "currency": "usd",
            "limit": 30, "sorting": "price", "one_way": "true",
            "direct": direct,
        },
        f"prices_for_dates  direct={direct}",
    )
    if s == 200:
        summarise_v3(b, f"direct={direct}")


# ---------------------------------------------------------------------------
# 2. Whole month — same endpoint, just departure_at=YYYY-MM
# ---------------------------------------------------------------------------
print("\n" + "=" * 75)
print(f" 2. aviasales/v3/prices_for_dates  whole month {DEPART_MONTH}")
print("=" * 75)
for direct in ("false", "true"):
    s, b, _ = call(
        "https://api.travelpayouts.com/aviasales/v3/prices_for_dates",
        {
            "origin": ORIGIN, "destination": DEST,
            "departure_at": DEPART_MONTH, "currency": "usd",
            "limit": 30, "sorting": "price", "one_way": "true",
            "direct": direct,
        },
        f"prices_for_dates  month  direct={direct}",
    )
    if s == 200:
        summarise_v3(b, f"month direct={direct}")


# ---------------------------------------------------------------------------
# 3. grouped_prices — group by departure_at to see per-day cheapest
# ---------------------------------------------------------------------------
print("\n" + "=" * 75)
print(f" 3. aviasales/v3/grouped_prices  group_by=departure_at  {DEPART_MONTH}")
print("=" * 75)
s, b, _ = call(
    "https://api.travelpayouts.com/aviasales/v3/grouped_prices",
    {
        "origin": ORIGIN, "destination": DEST,
        "departure_at": DEPART_MONTH, "currency": "usd",
        "group_by": "departure_at", "trip_class": 0, "limit": 30,
    },
    "grouped_prices  departure_at",
)
if s == 200:
    data = b.get("data") or {}
    print(f"    -> {len(data)} buckets")
    # Show cheapest per day
    rows = []
    for k, v in data.items():
        if isinstance(v, dict):
            rows.append((k, v.get("price"), v.get("airline"), v.get("transfers"), v.get("flight_number")))
    rows.sort(key=lambda r: r[0])
    for day, price, airline, stops, fn in rows[:20]:
        print(f"       {day}  ${price}  {airline} {fn}  stops={stops}  ({AIRLINES.get(airline, airline)})")


# ---------------------------------------------------------------------------
# 4. get_latest_prices — cached fares feed
# ---------------------------------------------------------------------------
print("\n" + "=" * 75)
print(" 4. v2/prices/latest  (cached recent fares)")
print("=" * 75)
s, b, _ = call(
    "https://api.travelpayouts.com/v2/prices/latest",
    {"origin": ORIGIN, "destination": DEST, "currency": "usd", "limit": 30,
     "period_type": "year", "show_to_affiliates": "false"},
    "v2/prices/latest",
)
if s == 200:
    summarise_v2_latest(b, "latest")


# ---------------------------------------------------------------------------
# 5. v2/prices/month-matrix — cheapest for each day of the month
# ---------------------------------------------------------------------------
print("\n" + "=" * 75)
print(f" 5. v2/prices/month-matrix  month={DEPART_MONTH}")
print("=" * 75)
s, b, _ = call(
    "https://api.travelpayouts.com/v2/prices/month-matrix",
    {"origin": ORIGIN, "destination": DEST, "month": DEPART_MONTH,
     "currency": "usd", "show_to_affiliates": "true"},
    "v2/prices/month-matrix",
)
if s == 200:
    rows = b.get("data") or []
    if rows:
        rows.sort(key=lambda r: r.get("depart_date", ""))
        print(f"    -> {len(rows)} rows (one per day)")
        for r in rows[:31]:
            print(
                f"       {r.get('depart_date')}  ${r.get('value')}  "
                f"changes={r.get('number_of_changes')}  gate={r.get('gate')}  "
                f"trip_class={r.get('trip_class')}  return={r.get('return_date')}"
            )
    else:
        print(f"    -> empty.  error={b.get('error')}")


# ---------------------------------------------------------------------------
# 6. v1/prices/cheap — cheapest per OTA per stops
# ---------------------------------------------------------------------------
print("\n" + "=" * 75)
print(" 6. v1/prices/cheap")
print("=" * 75)
s, b, _ = call(
    "https://api.travelpayouts.com/v1/prices/cheap",
    {"origin": ORIGIN, "destination": DEST, "currency": "usd",
     "depart_date": DEPART_MONTH, "return_date": "2026-06"},
    "v1/prices/cheap",
)
if s == 200:
    summarise_v2_cheap(b, "cheap")


# ---------------------------------------------------------------------------
# 7. v1/prices/calendar — calendar of cheap fares
# ---------------------------------------------------------------------------
print("\n" + "=" * 75)
print(f" 7. v1/prices/calendar  depart_date={DEPART_MONTH}")
print("=" * 75)
s, b, _ = call(
    "https://api.travelpayouts.com/v1/prices/calendar",
    {"origin": ORIGIN, "destination": DEST, "currency": "usd",
     "depart_date": DEPART_MONTH, "calendar_type": "departure_date"},
    "v1/prices/calendar",
)
if s == 200:
    summarise_calendar(b)


# ---------------------------------------------------------------------------
# 8. v1/prices/direct — direct flights only
# ---------------------------------------------------------------------------
print("\n" + "=" * 75)
print(" 8. v1/prices/direct")
print("=" * 75)
s, b, _ = call(
    "https://api.travelpayouts.com/v1/prices/direct",
    {"origin": ORIGIN, "destination": DEST, "currency": "usd",
     "depart_date": DEPART_MONTH, "return_date": "2026-06"},
    "v1/prices/direct",
)
if s == 200:
    summarise_v2_cheap(b, "direct")

# ---------------------------------------------------------------------------
# 9. Try the specific CDG airport (Paris has CDG, ORY, BVA)
# ---------------------------------------------------------------------------
print("\n" + "=" * 75)
print(" 9. Same exact-day query, but destination=CDG explicitly")
print("=" * 75)
for dest_code in ("CDG", "ORY", "BVA"):
    s, b, _ = call(
        "https://api.travelpayouts.com/aviasales/v3/prices_for_dates",
        {
            "origin": ORIGIN, "destination": dest_code,
            "departure_at": DEPART_DAY, "currency": "usd",
            "limit": 30, "sorting": "price", "one_way": "true",
        },
        f"prices_for_dates  TLV->{dest_code}  {DEPART_DAY}",
    )
    if s == 200:
        summarise_v3(b, dest_code)

print("\nDONE.")
