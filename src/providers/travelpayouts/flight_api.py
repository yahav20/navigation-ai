"""Travelpayouts (Aviasales v3) flight search wrapper.

Exposes a single function `search_flights(origin_city, destination_city, departure_at)`
that returns a list of flight offers in the same shape the rest of the agent
already understands (see `src/providers/sqlite/flight_queries.py` for the
direct-flight schema and `src/agent/nodes/travel_agent.py` for how connecting
routes are consumed).

`departure_at` accepts either `YYYY-MM-DD` (specific date) or `YYYY-MM` (whole
month). The wrapper resolves city names to IATA codes via Travelpayouts'
autocomplete endpoint, and resolves airline codes to display names via the
public airlines.json feed.
"""
from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests

_ROUTE_BLOB_RE = re.compile(r"t=([^&]+)")
_ROUTE_TAIL_RE = re.compile(r"([A-Z]{6,})$")

_AVIASALES = "https://api.travelpayouts.com/aviasales/v3"
_AUTOCOMPLETE = "https://autocomplete.travelpayouts.com/places2"
_AIRLINES_FEED = "https://api.travelpayouts.com/data/en/airlines.json"

_TIMEOUT = 15
_DEFAULT_LIMIT = 5

# Drop offers that take noticeably longer than a realistic connection should.
# Direct flights are bounded so a single bad row doesn't slip through; multi-stop
# caps allow overnight layovers without admitting 36+ hour outliers.
_MAX_HOURS_BY_TRANSFERS = {0: 16, 1: 18, 2: 24}
_MAX_HOURS_FALLBACK = 30

# Value score for ranking: balances price against duration and number of layovers.
# Treat ~$30/hour of travel time as roughly equivalent to dollars, and add a flat
# penalty per transfer so direct & well-routed connections win over cheap 2-stops.
_DURATION_COST_PER_MIN = 0.5
_TRANSFER_PENALTY = 50.0

_iata_cache: dict[str, str] = {}
_airline_cache: dict[str, str] | None = None


def _api_key() -> str | None:
    return os.getenv("TRAVELPAYOUTS_API_KEY")


def _iata_for_city(name: str) -> str | None:
    """Resolve a city name to a 3-letter IATA code via Travelpayouts autocomplete."""
    if not name:
        return None
    key = name.strip().lower()
    if key in _iata_cache:
        return _iata_cache[key]
    try:
        r = requests.get(
            _AUTOCOMPLETE,
            params={"term": name, "locale": "en", "types[]": "city"},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        results = r.json() or []
    except (requests.RequestException, ValueError):
        return None
    code = None
    for item in results:
        candidate = item.get("code")
        if candidate and len(candidate) == 3:
            code = candidate
            break
    if code:
        _iata_cache[key] = code
    return code


def _airline_name(code: str | None) -> str:
    """Resolve a 2-letter IATA airline code to its display name."""
    global _airline_cache
    if not code:
        return ""
    if _airline_cache is None:
        try:
            r = requests.get(_AIRLINES_FEED, timeout=_TIMEOUT)
            r.raise_for_status()
            _airline_cache = {
                (a.get("iata") or a.get("code")): a.get("name")
                for a in r.json()
                if a.get("iata") or a.get("code")
            }
        except (requests.RequestException, ValueError):
            _airline_cache = {}
    return _airline_cache.get(code) or code


def _prices_for_dates(
    origin_iata: str,
    destination_iata: str,
    departure_at: str,
    *,
    direct: bool,
    token: str,
    limit: int,
) -> list[dict]:
    try:
        r = requests.get(
            f"{_AVIASALES}/prices_for_dates",
            params={
                "origin": origin_iata,
                "destination": destination_iata,
                "departure_at": departure_at,
                "currency": "usd",
                "limit": limit,
                "sorting": "price",
                "one_way": "true",
                "direct": "true" if direct else "false",
                "token": token,
            },
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        payload = r.json() or {}
    except (requests.RequestException, ValueError):
        return []
    return payload.get("data") or []


def _parse_route_iatas(link: str | None) -> list[str]:
    """Extract the IATA sequence encoded in a Travelpayouts /search booking link.

    The link's `t=` param ends with consecutive uppercase letters that are the
    airport codes in order, e.g. `TLVATHORY` -> [TLV, ATH, ORY]. The format is
    undocumented but verified stable across direct, 1-stop, and 2-stop offers.
    Returns an empty list if the link is malformed.
    """
    if not link:
        return []
    blob_match = _ROUTE_BLOB_RE.search(link)
    if not blob_match:
        return []
    blob = blob_match.group(1).split("_", 1)[0]
    tail_match = _ROUTE_TAIL_RE.search(blob)
    if not tail_match:
        return []
    codes = tail_match.group(1)
    if len(codes) % 3 != 0:
        return []
    return [codes[i : i + 3] for i in range(0, len(codes), 3)]


def _normalize_offer(offer: dict, origin_city: str, destination_city: str) -> dict:
    """Shape any prices_for_dates offer (direct or connecting) the same way.

    The v3 endpoint does NOT itemize legs of a connecting flight — the
    `airline` / `flight_number` it returns refer only to the first leg, and the
    intermediate stop airport is not exposed. We therefore avoid synthesizing
    fake leg data and just surface the final origin/destination airports plus
    the layover count.
    """
    code = offer.get("airline")
    route_iatas = _parse_route_iatas(offer.get("link"))
    # Intermediate airports = everything except the first (origin) and last (final dest).
    stop_airports = route_iatas[1:-1] if len(route_iatas) >= 2 else []
    return {
        "origin_city": origin_city,
        "destination_city": destination_city,
        "origin_airport": offer.get("origin_airport") or "",
        "destination_airport": offer.get("destination_airport") or "",
        "stop_airports": stop_airports,
        "airline": _airline_name(code) or code or "",
        "flight_number": offer.get("flight_number") or "",
        "price": offer.get("price"),
        "availability": "available",
        "departure_time": offer.get("departure_at") or "",
        "arrival_time": "",  # not provided by this endpoint
        # Searches are one-way (one_way=true), so `duration_to` is the outbound
        # leg's travel time. `duration` can include round-trip/total figures that
        # make a legitimate one-way offer look implausibly long and get filtered
        # out by `_is_reasonable`. Prefer the leg duration, falling back to total.
        "duration_minutes": offer.get("duration_to") or offer.get("duration"),
        "transfers": int(offer.get("transfers") or 0),
    }


def _is_reasonable(offer: dict) -> bool:
    """Drop offers whose duration is implausible for the number of layovers."""
    duration = offer.get("duration_minutes")
    if not isinstance(duration, (int, float)):
        return True  # don't filter when we don't know
    transfers = offer.get("transfers") or 0
    cap_hours = _MAX_HOURS_BY_TRANSFERS.get(transfers, _MAX_HOURS_FALLBACK)
    return duration <= cap_hours * 60


def _value_score(offer: dict) -> float:
    """Score combining price, duration, and transfer penalty (lower is better)."""
    price = offer.get("price")
    if not isinstance(price, (int, float)):
        return 1e9
    duration = offer.get("duration_minutes") or 0
    transfers = offer.get("transfers") or 0
    return float(price) + duration * _DURATION_COST_PER_MIN + transfers * _TRANSFER_PENALTY


def _dedupe_offers(offers: list[dict]) -> list[dict]:
    seen: set[tuple[Any, Any]] = set()
    out: list[dict] = []
    for offer in offers:
        key = (offer.get("flight_number"), offer.get("departure_at"))
        if key in seen:
            continue
        seen.add(key)
        out.append(offer)
    return out


def search_flights(
    origin_city: str,
    destination_city: str,
    departure_at: str,
    *,
    limit: int = _DEFAULT_LIMIT,
) -> list[dict]:
    """Return flight offers in the agent's existing payload schema.

    Direct flights are emitted with `flight_number`/`price`/`airline` keys.
    Connecting flights are emitted with a `route` array and `total_price`.
    Returns an empty list on missing key, unresolvable cities, or upstream error.
    """
    token = _api_key()
    if not token:
        return []

    # Resolve both IATAs in parallel — autocomplete calls are independent
    # network round-trips (cached after first lookup).
    with ThreadPoolExecutor(max_workers=2) as pool_exec:
        origin_iata, destination_iata = pool_exec.map(
            _iata_for_city, (origin_city, destination_city)
        )
    if not origin_iata or not destination_iata:
        return []

    # Pull a wider pool than we ultimately return so the duration filter,
    # value ranking, and per-flight dedupe all have room to operate before
    # we cut to `limit`.
    pool = max(limit * 5, 40)
    # The mixed (direct=false) and direct-only feeds are independent — fetch
    # them concurrently to halve the wall-clock latency. The direct-only call
    # backstops the mixed feed truncating cached non-stops.
    with ThreadPoolExecutor(max_workers=2) as pool_exec:
        mixed_future = pool_exec.submit(
            _prices_for_dates,
            origin_iata, destination_iata, departure_at,
            direct=False, token=token, limit=pool,
        )
        direct_future = pool_exec.submit(
            _prices_for_dates,
            origin_iata, destination_iata, departure_at,
            direct=True, token=token, limit=max(5, limit),
        )
    raw = mixed_future.result() + direct_future.result()
    raw = _dedupe_offers(raw)

    normalized = [
        _normalize_offer(offer, origin_city, destination_city)
        for offer in raw
        if isinstance(offer, dict)
    ]
    normalized = [o for o in normalized if _is_reasonable(o)]
    normalized.sort(key=_value_score)

    # Dedupe by (airline, flight_number) keeping the cheapest/best-ranked
    # variant. Travelpayouts returns the same operating flight on many
    # departure dates, and without this the agent sees one airline 5x instead
    # of 5 different airlines.
    seen_flights: set[tuple[Any, Any]] = set()
    diverse: list[dict] = []
    for offer in normalized:
        key = (offer.get("airline"), offer.get("flight_number"))
        if key in seen_flights:
            continue
        seen_flights.add(key)
        diverse.append(offer)
    return diverse[:limit]
