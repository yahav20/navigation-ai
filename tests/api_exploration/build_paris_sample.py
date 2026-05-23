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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from dotenv import load_dotenv

# Local import — Wikipedia/Wikivoyage enrichment for attractions
sys.path.insert(0, str(Path(__file__).resolve().parent))
from enrich_attractions_wiki import enrich_attractions, attractions_markdown_section  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
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

# Trip dates — used for flight search AND hotel rate lookup.
DEPART = "2026-06-10"
RETURN = "2026-06-13"


# ---------------------------------------------------------------------------
# Xotelo (TripAdvisor-backed, no auth) — real per-night USD rates per OTA
# ---------------------------------------------------------------------------
XOTELO = "https://data.xotelo.com/api"
# Paris TripAdvisor location key
PARIS_LOCATION_KEY = "g187147"


def xotelo_list_hotels(location_key: str, limit: int = 10, sort: str = "popularity") -> list[dict]:
    r = requests.get(
        f"{XOTELO}/list",
        params={"location_key": location_key, "limit": limit, "sort": sort},
        timeout=20,
    )
    r.raise_for_status()
    return ((r.json().get("result") or {}).get("list") or [])


def xotelo_rates(hotel_key: str, chk_in: str, chk_out: str, currency: str = "USD") -> list[dict]:
    r = requests.get(
        f"{XOTELO}/rates",
        params={"hotel_key": hotel_key, "chk_in": chk_in, "chk_out": chk_out, "currency": currency},
        timeout=20,
    )
    if r.status_code != 200:
        return []
    return ((r.json().get("result") or {}).get("rates") or [])


def fetch_paris_hotels(chk_in: str, chk_out: str, limit: int = 10, with_rates: bool = False) -> list[dict]:
    """Pull hotels from Xotelo's /list.

    /list is fast (~4s once) and gives the historical USD `price_ranges.min/max`
    per night — usually enough for planning. /rates adds the per-OTA breakdown
    for the specific dates but stalls ~8-10s per hotel when TripAdvisor has no
    cached quote, so it's off by default.
    """
    out: list[dict] = []
    for h in xotelo_list_hotels(PARIS_LOCATION_KEY, limit=limit, sort="popularity"):
        hotel_key = h.get("key")
        geo = h.get("geo") or {}
        review = h.get("review_summary") or {}
        pr = h.get("price_ranges") or {}

        rates: list[dict] = []
        live_rate = None
        if with_rates and hotel_key:
            rates = xotelo_rates(hotel_key, chk_in, chk_out)
            live_rate = min(
                (r.get("rate") for r in rates if isinstance(r.get("rate"), (int, float))),
                default=None,
            )

        out.append({
            "name": h.get("name"),
            "hotel_key": hotel_key,
            "type": h.get("accommodation_type"),
            "rating": review.get("rating"),
            "review_count": review.get("count"),
            "price_min_usd": pr.get("minimum"),
            "price_max_usd": pr.get("maximum"),
            "best_rate_usd": live_rate,
            "rates_by_ota": [
                {"name": r.get("name"), "rate_usd": r.get("rate")}
                for r in rates
            ],
            "lat": geo.get("latitude"),
            "lng": geo.get("longitude"),
            "mentions": h.get("mentions", []),
            "image": h.get("image"),
            "tripadvisor_url": h.get("url"),
        })
    return out



# ---------------------------------------------------------------------------
# 1. Flights — TLV -> PAR (Aviasales v3, three queries combined)
#
# The exact-day query on a city code ("PAR") returns only the single cheapest
# cached offer. We get richer results by:
#   - separately asking for direct flights on the date
#   - asking for cheapest connecting per Paris airport (CDG, ORY, BVA)
#   - pulling the monthly per-day calendar so the agent can suggest shifting
# ---------------------------------------------------------------------------
AVIASALES = "https://api.travelpayouts.com/aviasales/v3"
PARIS_AIRPORTS = ("CDG", "ORY", "BVA")
_airline_cache: dict[str, str] | None = None


def airline_name(code: str | None) -> str:
    """Resolve a 2-letter IATA airline code to its name. Cached after first call."""
    global _airline_cache
    if not code:
        return ""
    if _airline_cache is None:
        try:
            r = requests.get("https://api.travelpayouts.com/data/en/airlines.json", timeout=15)
            r.raise_for_status()
            _airline_cache = {a.get("iata") or a.get("code"): a.get("name") for a in r.json() if a.get("iata") or a.get("code")}
        except Exception:
            _airline_cache = {}
    return _airline_cache.get(code) or code


def _normalize_offer(it: dict) -> dict:
    code = it.get("airline")
    return {
        "airline_code": code,
        "airline_name": airline_name(code),
        "flight_number": it.get("flight_number"),
        "price_usd": it.get("price"),
        "departure_at": it.get("departure_at"),
        "origin_airport": it.get("origin_airport"),
        "destination_airport": it.get("destination_airport"),
        "duration_minutes": it.get("duration"),
        "transfers": it.get("transfers"),
        "seller": it.get("gate"),
        "booking_link": "https://www.aviasales.com" + (it.get("link") or ""),
    }


def _prices_for_dates(origin: str, dest: str, depart: str, direct: bool, limit: int = 5) -> list[dict]:
    r = requests.get(
        f"{AVIASALES}/prices_for_dates",
        params={
            "origin": origin, "destination": dest,
            "departure_at": depart, "currency": "usd",
            "limit": limit, "sorting": "price", "one_way": "true",
            "direct": "true" if direct else "false",
            "token": TP_KEY,
        },
        timeout=15,
    )
    r.raise_for_status()
    return [_normalize_offer(it) for it in r.json().get("data", [])]


def _grouped_by_day(origin: str, dest: str, month: str) -> list[dict]:
    """grouped_prices group_by=departure_at — one cheapest offer per day."""
    r = requests.get(
        f"{AVIASALES}/grouped_prices",
        params={
            "origin": origin, "destination": dest,
            "departure_at": month, "currency": "usd",
            "group_by": "departure_at", "trip_class": 0,
            "token": TP_KEY,
        },
        timeout=15,
    )
    r.raise_for_status()
    out: list[dict] = []
    for day, offer in (r.json().get("data") or {}).items():
        if isinstance(offer, dict):
            code = offer.get("airline")
            out.append({
                "date": day,
                "price_usd": offer.get("price"),
                "airline_code": code,
                "airline_name": airline_name(code),
                "flight_number": offer.get("flight_number"),
                "transfers": offer.get("transfers"),
                "duration_minutes": offer.get("duration"),
                "destination_airport": offer.get("destination_airport"),
            })
    out.sort(key=lambda r: r["date"])
    return out


def fetch_flight_options(origin: str, destination: str, depart: str) -> dict:
    """Combine direct + per-airport cheapest connecting + monthly calendar."""
    direct = _prices_for_dates(origin, destination, depart, direct=True, limit=5)

    # Per-Paris-airport cheapest connecting (each call returns the single
    # cheapest cached offer for that airport on that day)
    connecting: list[dict] = []
    seen_keys: set[tuple] = set()
    for ap in PARIS_AIRPORTS:
        for offer in _prices_for_dates(origin, ap, depart, direct=False, limit=3):
            key = (offer["flight_number"], offer["departure_at"])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            connecting.append(offer)
    connecting.sort(key=lambda o: o["price_usd"] or 1e9)

    month = depart[:7]
    calendar = _grouped_by_day(origin, destination, month)

    return {
        "depart_date": depart,
        "origin": origin,
        "destination": destination,
        "direct_on_date": direct,
        "connecting_on_date": connecting,
        "monthly_cheapest_by_day": calendar,
    }


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
print(f"Fetching flight options TLV -> PAR for {DEPART} (direct + connecting + month) …")
flight_options = fetch_flight_options("TLV", "PAR", DEPART)
print(f"  direct on date:       {len(flight_options['direct_on_date'])}")
print(f"  connecting on date:   {len(flight_options['connecting_on_date'])}")
print(f"  monthly day-by-day:   {len(flight_options['monthly_cheapest_by_day'])}")

print("Geocoding Paris …")
lat, lng, paris_addr = geocode("Paris, France")

print(f"Fetching hotels via Xotelo (TripAdvisor) for {DEPART}..{RETURN} …")
# `with_rates=True` adds per-OTA prices for the trip dates, but adds ~50s
# (each /rates call stalls 8-10s when TripAdvisor has no cached quote).
# The min/max range from /list is enough for planning.
hotels = fetch_paris_hotels(DEPART, RETURN, limit=10, with_rates=False)

print("Fetching attractions near Paris …")
attractions = places_nearby(lat, lng, "tourist_attraction", radius=4500, limit=10)

print("Enriching attractions with Wikipedia + Wikivoyage …")
enrich_attractions(attractions, verbose=False)

print("Fetching restaurants near Paris (new Places API for priceRange) …")
restaurants = places_v1_text_search(
    "restaurants in Paris",
    limit=15,
    drop_types={"lodging"},  # filter out hotels-with-restaurants
)

sample = {
    "generated_for": "2026-06-10 trip to Paris",
    "paris_center": {"lat": lat, "lng": lng, "address": paris_addr},
    "flights_TLV_PAR_2026_06_10": flight_options,
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

def _flight_row(f: dict) -> str:
    depart = (f.get("departure_at") or "")[:16].replace("T", " ")
    return (
        f"| ${f['price_usd']} | {f['airline_code']} ({f['airline_name']}) | "
        f"{f['flight_number']} | {depart} | {f['transfers']} | "
        f"{fmt_minutes(f['duration_minutes'])} | "
        f"{f['origin_airport']}→{f['destination_airport']} | {f['seller']} |"
    )


lines.append(f"\n## Flights TLV → PAR  (Travelpayouts / Aviasales)\n")

lines.append(f"### Direct flights on {DEPART}\n")
lines.append("| Price | Airline | Flight | Depart | Stops | Duration | Route | Seller |")
lines.append("|---:|---|---|---|---:|---|---|---|")
if flight_options["direct_on_date"]:
    for f in flight_options["direct_on_date"]:
        lines.append(_flight_row(f))
else:
    lines.append("| _no direct flights cached for this date_ | | | | | | | |")

lines.append(f"\n### Cheapest connecting options on {DEPART}  (per Paris airport)\n")
lines.append("| Price | Airline | Flight | Depart | Stops | Duration | Route | Seller |")
lines.append("|---:|---|---|---|---:|---|---|---|")
if flight_options["connecting_on_date"]:
    for f in flight_options["connecting_on_date"]:
        lines.append(_flight_row(f))
else:
    lines.append("| _no connecting offers cached_ | | | | | | | |")

lines.append(f"\n### Cheapest fare per day across {DEPART[:7]}  (flex your date)\n")
lines.append("| Date | Price | Airline | Flight | Stops | Duration | To |")
lines.append("|---|---:|---|---|---:|---|---|")
target_day = flight_options["depart_date"]
for d in flight_options["monthly_cheapest_by_day"]:
    marker = " ← target" if d["date"] == target_day else ""
    lines.append(
        f"| {d['date']}{marker} | ${d['price_usd']} | "
        f"{d['airline_code']} ({d['airline_name']}) | {d['flight_number']} | "
        f"{d['transfers']} | {fmt_minutes(d['duration_minutes'])} | {d.get('destination_airport') or '—'} |"
    )
if flight_options["monthly_cheapest_by_day"]:
    cheapest = min(flight_options["monthly_cheapest_by_day"], key=lambda x: x["price_usd"] or 1e9)
    lines.append(f"\n_Cheapest day in window: **{cheapest['date']}** at **${cheapest['price_usd']}** ({cheapest['airline_name']}, {cheapest['transfers']} stops)._")


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


has_live_rates = any(h.get("best_rate_usd") is not None for h in hotels)

lines.append(f"\n## Hotels in Paris  (Xotelo / TripAdvisor)\n")
if has_live_rates:
    lines.append(f"_Per-night USD for {DEPART}..{RETURN}; range is the historical min/max from TripAdvisor._\n")
    lines.append("| Name | ★ | Reviews | Best rate/night | Min–max USD | Lat | Lng | OTAs |")
    lines.append("|---|---:|---:|---:|---|---:|---:|---|")
    for h in hotels:
        rating = h.get("rating") or "—"
        reviews = h.get("review_count") or "—"
        best = f"${h['best_rate_usd']:.0f}" if h.get("best_rate_usd") is not None else "—"
        mn, mx = h.get("price_min_usd"), h.get("price_max_usd")
        rng = f"${mn}–${mx}" if mn and mx else "—"
        otas = ", ".join(f"{r['name']} ${r['rate_usd']:.0f}" for r in (h.get("rates_by_ota") or [])[:5])
        lat_s = f"{h['lat']:.4f}" if h.get("lat") else "—"
        lng_s = f"{h['lng']:.4f}" if h.get("lng") else "—"
        lines.append(f"| {h['name']} | {rating} | {reviews} | {best} | {rng} | {lat_s} | {lng_s} | {otas or '—'} |")
else:
    lines.append("_Per-night USD range from TripAdvisor's historical data (live OTA rates skipped — slow)._\n")
    lines.append("| Name | ★ | Reviews | Min–max USD / night | Lat | Lng |")
    lines.append("|---|---:|---:|---|---:|---:|")
    for h in hotels:
        rating = h.get("rating") or "—"
        reviews = h.get("review_count") or "—"
        mn, mx = h.get("price_min_usd"), h.get("price_max_usd")
        rng = f"${mn}–${mx}" if mn and mx else "—"
        lat_s = f"{h['lat']:.4f}" if h.get("lat") else "—"
        lng_s = f"{h['lng']:.4f}" if h.get("lng") else "—"
        lines.append(f"| {h['name']} | {rating} | {reviews} | {rng} | {lat_s} | {lng_s} |")

lines.append("")
lines.extend(attractions_markdown_section(attractions))
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
print(f"  flights direct/connecting/calendar: "
      f"{len(flight_options['direct_on_date'])}/"
      f"{len(flight_options['connecting_on_date'])}/"
      f"{len(flight_options['monthly_cheapest_by_day'])}")
print(f"  hotels:      {len(hotels)}")
print(f"  attractions: {len(attractions)}")
print(f"  restaurants: {len(restaurants)}")
