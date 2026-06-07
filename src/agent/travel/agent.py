"""Deterministic travel agent node: curates a structured plan for the formatter."""

import json
import os
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

from agent.core.llm import silent
from agent.core.models import TravelPlan, TravelPlanCuration
from agent.core.state import AgentState
from agent.shared.pricing import flight_group_price, group_label, hotel_group_price
from agent.shared.travelers import compute_default_rooms
from tools.dependencies import data_provider
from security import SECURITY_RULES

_DAY_GAP_PENALTY = 30.0
_MAX_PAIRINGS = 10


SEASONS = ("Spring", "Summer", "Autumn", "Winter")

_CURATION_PROMPT = """You are Atlas, a deterministic travel agent.

You receive two blocks:
- <trip_context>: the trip constraints — origin, destination, total_budget, budget_applied,
  budget_optional, trip_days, trip_start, is_adjustment, and user_preferences. Honor these.
- <travel_payload>: the options to curate from — a pre-computed `pairings` array, hotels,
  activities, restaurants, weather, and best_time.
Your job is to curate the trip and return a structured plan.

Rules:
1. Use ONLY items present in the payload. Never invent flights, hotels, activities, restaurants, prices, or dates.
2. Produce exactly 3 round-trip `flight_pairings` (fewer only if `pairings` has fewer than 3 entries). For each pairing, pick one entry from the payload's `pairings` array — every entry there is already a valid (outbound, return) pair with the dates filtered and ranked for you. Each `pairings[i]` contains:
   - `outbound`: the outbound FlightPick (origin → destination).
   - `return_flight`: the return FlightPick (destination → origin).
   - `total_price`: outbound.price + return_flight.price (already summed).
   - `day_gap`: actual days between departure and return (may be null if dates were unparseable).
   The list is sorted by score = total_price + penalty for `day_gap` deviating from `trip_days`. Aim for VARIETY across the 3 picks (cheapest, fastest, different airlines/times) — do NOT just take the first three. Copy `outbound`, `return_flight`, and `total_price` straight through and write a fresh `description` line.
   When `day_gap` is set, prefer pairings whose gap is close to `trip_days`. If the closest available gap is far off, mention that briefly in the description ("note: return is 6 days out instead of 4 due to schedule").
3. Pick **exactly 3 hotels** from the payload's **`hotels` array** (fewer only if that array has fewer than 3 entries). Do NOT pick from `activities` — those are sightseeing items, not accommodation. Aim for **price variety**: one budget-friendly, one mid-range, one premium — so the user has real choices. When budget_applied=true (see <trip_context>), all 3 must fit within the budget (flight_outbound + cheapest_return + price_per_night × trip_days ≤ total_budget). Use the `price_per_night` value from the payload as-is — never write $0 for a hotel.
4. Pick up to 5 activities from the payload's `activities` list (a list of activity names, already curated for this destination). Respect user_preferences (e.g. dietary_restrictions, preferred_location). Write a short description for each pick.
5. Pick up to 3 restaurants from the payload's `restaurants` list. If the list is empty, output an empty `restaurants` array. For each pick fill: `name` (from payload), `price_tier` (from `price_level_text` or null), `rating` (from payload or null), and a short `description`.
6. For each pick, write a short one-line description of why it fits.
7. For every FlightPick inside a pairing (both `outbound` and `return_flight`):
   - Set `stops` to the source flight's `transfers` value when present, otherwise to `len(route) - 1` when the source has a `route` array, otherwise 0.
   - Set `duration_minutes` to the source flight's `duration_minutes` when present, or to the sum of `route[i].duration_minutes` across legs for multi-leg routes. Leave null if no leg reports a duration.
   - Set `departure_time` to the source flight's `departure_time` (or `route[0].departure_time` for multi-leg). Leave null if the payload has no departure time.
   - Set `destination_airport` to the source flight's `destination_airport` when present (IATA code). Leave null if absent.
   - Set `stop_airports` to the source flight's `stop_airports` array (intermediate IATA codes; empty for direct flights). Copy verbatim.
   - Only fill `legs` when the source has an itemized `route` array (multi-leg). Copy each entry as {from_city, to_city, airline, flight_number}. For single-segment offers from the live API (no `route` array), leave `legs` empty — `stops` already conveys whether it is direct or has layovers.
8. `intro` and `sign_off` should be one short sentence each, friendly but concise. When trip_start is set, you may mention it in `intro` (e.g. "for your trip starting around 2026-06").
9. All prices in USD. Use the values from the payload as-is.
10. Framing based on is_adjustment:
   - is_adjustment=true: the user is updating an existing plan. `intro` MUST start with "✅ Trip Updated Successfully!" and acknowledge the changes. `sign_off` should reassure them the new plan reflects their requested updates.
   - is_adjustment=false: this is a brand-new plan. `intro` should welcome them to their plan (no "updated" framing). `sign_off` should encourage them to confirm or refine.
"""

_NO_FLIGHTS_NEW = (
    "Based on our search, we could not find available outbound or return flights "
    "between {origin} and {destination} at this time."
)
_NO_FLIGHTS_ADJUSTMENT = (
    "⚠️ **Update Failed:** I tried to update your trip, but I couldn't find both "
    "outbound and return flights between {origin} and {destination} for the new request."
)
_NO_HOTELS_NEW = (
    "I found flights for this route, but no hotels that match the available "
    "inventory and budget constraints."
)
_NO_HOTELS_ADJUSTMENT = (
    "⚠️ **Update Failed:** I found flights for your updated request, but no "
    "hotels fit the newly adjusted budget or preferences."
)


_LODGING_TYPES = {"lodging", "hotel"}

# Fields the LLM actually consumes per section (output models: HotelPick,
# RestaurantPick, FlightPick). Everything else in the source rows is input-only
# noise — dropped before serialising the prompt. See _slim_for_llm below.
_HOTEL_KEEP = {"name", "stars", "price_per_night"}
_RESTAURANT_KEEP = {"name", "rating", "price_level_text", "price_level"}
# FlightPick carries no city fields (origin/destination are implied by the trip);
# availability/arrival_time are unused or always blank.
_FLIGHT_DROP = {"availability", "origin_city", "destination_city", "origin_airport", "arrival_time"}


def _strip_empty(item: dict) -> dict:
    """Drop keys whose value is blank/null. Most hotel/restaurant/flight rows
    have many empty columns ("", None, [], {}) that carry no signal. Keeps 0 and
    False — a $0 price or a real boolean flag is meaningful."""
    return {k: v for k, v in item.items() if v not in ("", None, [], {})}


def _round_numbers(obj: Any, ndigits: int = 2) -> Any:
    """Recursively round floats so the prompt carries clean values (e.g.
    4.4399999999999995 -> 4.44) instead of float-repr noise. bool is a subclass
    of int, so it's left untouched."""
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_numbers(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_numbers(v, ndigits) for v in obj]
    return obj


def _slim_flight(flight: dict) -> dict:
    return _strip_empty({k: v for k, v in flight.items() if k not in _FLIGHT_DROP})


def _slim_pairing(pairing: dict) -> dict:
    slim = dict(pairing)
    for side in ("outbound", "return_flight"):
        if isinstance(slim.get(side), dict):
            slim[side] = _slim_flight(slim[side])
    return slim


# Set ATLAS_DEBUG_TRAVEL_AGENT=1 to print a payload-build vs LLM-call timing split
# and dump the exact curation messages to out/travel_agent_prompt.json for replay.
_DEBUG = os.getenv("ATLAS_DEBUG_TRAVEL_AGENT") == "1"
_PROMPT_DUMP_PATH = Path(__file__).resolve().parents[3] / "out" / "travel_agent_prompt.json"


def _is_message(item: object) -> bool:
    return isinstance(item, dict) and bool(item.get("message"))


def _is_lodging(activity: dict) -> bool:
    cats = activity.get("categories") or []
    if isinstance(cats, str):
        try:
            cats = json.loads(cats)
        except Exception:
            cats = []
    if _LODGING_TYPES & {str(c).lower() for c in cats}:
        return True
    name = (activity.get("name") or "").lower().strip()
    return name.startswith("hotel ") or name.startswith("hôtel ")


def _valid_items(items: object) -> list[dict]:
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict) and not _is_message(item)]


def _flight_price(flight: dict) -> float | None:
    price = flight.get("total_price", flight.get("price"))
    if isinstance(price, (int, float)):
        return float(price)
    return None


def _flight_label(flight: dict) -> str:
    if flight.get("flight_number"):
        return str(flight["flight_number"])
    route = flight.get("route")
    if isinstance(route, list):
        labels = [str(leg.get("flight")) for leg in route if isinstance(leg, dict) and leg.get("flight")]
        if labels:
            return " + ".join(labels)
    return "flight option"


def _hotel_price(hotel: dict) -> float | None:
    price = hotel.get("price_per_night")
    if isinstance(price, (int, float)):
        return float(price)
    return None


def _apply_hotel_preferences(hotels: list[dict], preferences: dict) -> list[dict]:
    result = hotels
    min_stars = preferences.get("min_hotel_stars")
    if isinstance(min_stars, (int, float)):
        result = [hotel for hotel in result if hotel.get("stars", 0) >= min_stars]

    dietary = (preferences.get("dietary_restrictions") or "").lower()
    if "kosher" in dietary:
        kosher_hotels = [hotel for hotel in result if hotel.get("is_kosher")]
        if kosher_hotels:
            result = kosher_hotels

    return result


def _hotel_max_price(preferences: dict) -> int | None:
    max_price = preferences.get("max_hotel_price_per_night")
    if isinstance(max_price, (int, float)):
        return int(max_price)
    return None


def _fetch_weather(destination: str) -> dict:
    weather = {}
    for season in SEASONS:
        item = data_provider.get_average_weather(destination, season)
        if isinstance(item, dict) and item.get("season") and item.get("temperature"):
            weather[str(item["season"])] = item["temperature"]
    return weather


def _cheapest_flight_price(flights: list[dict]) -> float | None:
    prices = [p for p in (_flight_price(f) for f in flights) if p is not None]
    return min(prices) if prices else None


def _flight_date(flight: dict) -> date | None:
    iso = flight.get("departure_time") or ""
    if not iso:
        route = flight.get("route")
        if isinstance(route, list) and route:
            iso = route[0].get("departure_time", "") or ""
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso).date()
    except (ValueError, TypeError):
        return None


def _flight_identity(flight: dict) -> tuple[Any, Any]:
    return (flight.get("airline"), flight.get("flight_number") or _flight_label(flight))


def _build_pairings(
    outbound: list[dict],
    return_flights: list[dict],
    trip_days: int,
    *,
    limit: int = _MAX_PAIRINGS,
) -> list[dict]:
    target = max(int(trip_days), 1)
    candidates: list[dict] = []

    for ob in outbound:
        ob_price = _flight_price(ob)
        if ob_price is None:
            continue
        ob_date = _flight_date(ob)
        for rb in return_flights:
            rb_price = _flight_price(rb)
            if rb_price is None:
                continue
            rb_date = _flight_date(rb)

            day_gap: int | None
            if ob_date and rb_date:
                day_gap = (rb_date - ob_date).days
                if day_gap < 1:
                    continue
                gap_penalty = abs(day_gap - target) * _DAY_GAP_PENALTY
            else:
                day_gap = None
                gap_penalty = 0.0

            total_price = ob_price + rb_price
            candidates.append({
                "outbound": ob,
                "return_flight": rb,
                "total_price": round(total_price, 2),
                "day_gap": day_gap,
                "_score": total_price + gap_penalty,
            })

    candidates.sort(key=lambda c: c["_score"])

    seen: set[tuple] = set()
    diverse: list[dict] = []
    for cand in candidates:
        key = (_flight_identity(cand["outbound"]), _flight_identity(cand["return_flight"]))
        if key in seen:
            continue
        seen.add(key)
        cand.pop("_score", None)
        diverse.append(cand)
        if len(diverse) >= limit:
            break
    return diverse


def _build_costs(
    flights: list[dict],
    return_flights: list[dict],
    hotels: list[dict],
    trip_days: int,
    budget: float | None,
    *,
    budget_optional: bool,
    num_adults: int = 1,
    num_children: int = 0,
    num_rooms: int = 1,
) -> dict[str, Any]:
    options = []
    apply_budget = isinstance(budget, (int, float)) and not budget_optional
    return_floor_per_person = _cheapest_flight_price(return_flights) or 0.0

    for flight in flights:
        flight_cost = _flight_price(flight)
        if flight_cost is None:
            continue

        for hotel in hotels:
            nightly_rate = _hotel_price(hotel)
            if nightly_rate is None:
                continue

            # Per-adult baseline (1 adult, solo — keeps existing behaviour intact)
            hotel_solo  = nightly_rate * trip_days
            total_solo  = flight_cost + return_floor_per_person + hotel_solo

            # Group total using pricing formulas
            hotel_group  = hotel_group_price(nightly_rate, num_rooms, trip_days)
            group_out    = flight_group_price(flight_cost, num_adults, num_children)
            group_ret    = flight_group_price(return_floor_per_person, num_adults, num_children)
            total_group  = group_out + group_ret + hotel_group

            options.append({
                "flight": _flight_label(flight),
                "hotel": hotel.get("name", "hotel option"),
                "breakdown": {
                    "flight_outbound": flight_cost,
                    "flight_return": return_floor_per_person,
                    "hotel_total": hotel_solo,
                    "hotel_group_total": hotel_group,
                    "days": trip_days,
                    "num_adults": num_adults,
                    "num_children": num_children,
                    "num_rooms": num_rooms,
                },
                "total_estimate": total_solo,
                "group_total_estimate": total_group,
                "currency": "USD",
                "within_budget": total_group <= budget if apply_budget else None,
            })

    # Sort by solo cost (as before) so lowest_total_estimate is the true cheapest
    # per-adult baseline. lowest_group_estimate then comes from the SAME option,
    # making the two numbers directly comparable (group = solo + extra-adult flights).
    # Budget filtering still uses the group cost (within_budget flag).
    options = sorted(options, key=lambda item: item["total_estimate"])
    affordable_options = [item for item in options if item["within_budget"] is True]
    visible_options = affordable_options if apply_budget else options

    return {
        "options": visible_options[:10],
        "lowest_total_estimate": visible_options[0]["total_estimate"] if visible_options else None,
        "lowest_group_estimate": visible_options[0]["group_total_estimate"] if visible_options else None,
        "budget_applied": apply_budget,
        "return_flight_floor": return_floor_per_person,
    }


def _affordable_hotel_names(
    hotels: list[dict],
    flights: list[dict],
    return_flights: list[dict],
    trip_days: int,
    budget: float,
    *,
    num_adults: int = 1,
    num_children: int = 0,
    num_rooms: int = 1,
) -> set[str]:
    flight_costs = [c for c in (_flight_price(f) for f in flights) if c is not None]
    if not flight_costs:
        return set()
    cheapest_outbound = min(flight_costs)
    cheapest_return   = _cheapest_flight_price(return_flights) or 0.0
    group_out = flight_group_price(cheapest_outbound, num_adults, num_children)
    group_ret = flight_group_price(cheapest_return, num_adults, num_children)
    names: set[str] = set()
    for hotel in hotels:
        nightly = _hotel_price(hotel)
        if nightly is None:
            continue
        if group_out + group_ret + hotel_group_price(nightly, num_rooms, trip_days) <= budget:
            names.add(hotel.get("name", ""))
    names.discard("")
    return names


def _filter_hotels_by_budget(
    hotels: list[dict],
    flights: list[dict],
    return_flights: list[dict],
    trip_days: int,
    costs: dict[str, Any],
    budget: float | None,
    *,
    num_adults: int = 1,
    num_children: int = 0,
    num_rooms: int = 1,
) -> list[dict]:
    if not costs.get("budget_applied") or not isinstance(budget, (int, float)):
        return hotels
    keep = _affordable_hotel_names(
        hotels, flights, return_flights, trip_days, float(budget),
        num_adults=num_adults, num_children=num_children, num_rooms=num_rooms,
    )
    return [hotel for hotel in hotels if hotel.get("name") in keep]


def build_travel_prompt_payload(state: AgentState) -> dict:
    """Build the complete structured payload for the deterministic travel agent."""
    destination = state.get("destination_city")
    trip_days = int(state.get("trip_days") or 3)
    preferences = state.get("user_preferences") or {}

    flights = _valid_items(state.get("flight_options", []))
    return_flights = _valid_items(state.get("return_flight_options", []))
    hotels = []
    activities = []
    weather = {}
    best_time = {}

    restaurants: list[dict] = []

    if destination:
        _api_hotels = _valid_items(data_provider.fetch_hotels(destination, approach="api"))
        if _api_hotels:
            hotels = _apply_hotel_preferences(_api_hotels, preferences)
        else:
            hotels = _apply_hotel_preferences(
                _valid_items(data_provider.fetch_hotels(destination, approach="local")),
                preferences,
            )

        _api_activities = [
            a for a in _valid_items(data_provider.fetch_activities(destination, approach="api"))
            if not _is_lodging(a)
        ]
        activities = _api_activities if _api_activities else _valid_items(
            data_provider.fetch_activities(destination, approach="local")
        )

        restaurants = _valid_items(data_provider.fetch_restaurants(destination))
        weather = _fetch_weather(destination)

        best_time_result = data_provider.get_best_time_to_visit(destination)
        if isinstance(best_time_result, dict) and not _is_message(best_time_result):
            best_time = best_time_result

    budget = state.get("total_budget")
    budget_value = budget if isinstance(budget, (int, float)) else None

    num_adults   = int(state.get("num_adults")   or 1)
    num_children = int(state.get("num_children") or 0)
    num_rooms    = int(state.get("num_rooms") or compute_default_rooms(num_adults, num_children))

    costs = _build_costs(
        flights,
        return_flights,
        hotels,
        trip_days,
        budget_value,
        budget_optional=bool(state.get("budget_optional", False)),
        num_adults=num_adults,
        num_children=num_children,
        num_rooms=num_rooms,
    )
    hotels = _filter_hotels_by_budget(
        hotels, flights, return_flights, trip_days, costs, budget_value,
        num_adults=num_adults, num_children=num_children, num_rooms=num_rooms,
    )

    pairings = _build_pairings(flights, return_flights, trip_days)

    # Align estimates with the cheapest bookable pairing + cheapest hotel.
    # _build_costs uses return_floor_per_person (global cheapest return) which
    # may be from a different option than the cheapest displayed pairing, causing
    # a small but visible gap between the shown flight prices and the estimate.
    # Overriding here ensures: solo = cheapest_pairing.total_price + cheapest_hotel×days.
    if pairings and hotels:
        cheapest_p = min(pairings, key=lambda p: p["total_price"])
        ob_p = _flight_price(cheapest_p["outbound"])
        rb_p = _flight_price(cheapest_p["return_flight"])
        hotel_nights = [_hotel_price(h) for h in hotels if _hotel_price(h) is not None]
        if ob_p is not None and rb_p is not None and hotel_nights:
            min_nightly = min(hotel_nights)
            costs["lowest_total_estimate"] = round(
                ob_p + rb_p + min_nightly * trip_days, 2
            )
            costs["lowest_group_estimate"] = round(
                flight_group_price(ob_p, num_adults, num_children)
                + flight_group_price(rb_p, num_adults, num_children)
                + hotel_group_price(min_nightly, num_rooms, trip_days),
                2,
            )

    return {
        "origin": state.get("current_city"),
        "destination": destination,
        "total_budget": budget,
        "budget_optional": bool(state.get("budget_optional", False)),
        "trip_days": trip_days,
        "trip_start": state.get("trip_start"),
        "user_preferences": preferences,
        "flights": flights,
        "return_flights": return_flights,
        "pairings": pairings,
        "hotels": hotels,
        "activities": activities,
        "restaurants": restaurants,
        "weather": weather,
        "best_time": best_time,
        "costs": costs,
        "is_adjustment": bool(state.get("is_adjustment", False)),
        "travelers_label": group_label(num_adults, num_children),
        "num_adults": num_adults,
        "num_children": num_children,
        "num_rooms": num_rooms,
    }


class TravelAgentNode:
    """Curate the deterministic payload into a structured TravelPlan for the formatter."""

    def __init__(self, response_model: Runnable) -> None:
        """Wrap the response model with structured output for curation."""
        self.curation_model = silent(response_model.with_structured_output(TravelPlanCuration))

    def __call__(self, state: AgentState) -> dict:
        """Return either `travel_plan` (success) or a fallback `AIMessage` (no data)."""
        if not state.get("messages"):
            return {}

        _t0 = time.perf_counter()
        payload = build_travel_prompt_payload(state)
        _build_ms = (time.perf_counter() - _t0) * 1000
        origin = payload.get("origin") or "your origin"
        destination = payload.get("destination") or "your destination"
        is_adjustment = payload["is_adjustment"]

        if not payload["flights"] or not payload["return_flights"]:
            template = _NO_FLIGHTS_ADJUSTMENT if is_adjustment else _NO_FLIGHTS_NEW
            text = template.format(origin=origin, destination=destination)
            return {"messages": [AIMessage(content=text, name="travel_agent")]}

        if not payload["hotels"]:
            text = _NO_HOTELS_ADJUSTMENT if is_adjustment else _NO_HOTELS_NEW
            return {"messages": [AIMessage(content=text, name="travel_agent")]}

        # The trip constraints go in their own labelled block at the top so the
        # model sees budget/dates/preferences clearly, separated from the options.
        trip_context = {
            "origin": payload.get("origin"),
            "destination": payload.get("destination"),
            "total_budget": payload.get("total_budget"),
            "budget_optional": bool(payload.get("budget_optional", False)),
            "budget_applied": payload["costs"].get("budget_applied"),
            "trip_days": payload["trip_days"],
            "trip_start": payload.get("trip_start"),
            "is_adjustment": is_adjustment,
            "user_preferences": payload.get("user_preferences") or {},
        }

        # The data block — just the options to curate from. Reductions, none of
        # which remove anything the model may use (see output models in models.py):
        #   - flights/return_flights: fully embedded inside each `pairings` entry.
        #   - activities: the model returns only name + its own description, so
        #     the other ~20 fields per activity are dead weight — names only.
        #   - hotels/restaurants/flights: keep only consumed fields, strip blanks.
        #   - pairings: capped at _MAX_PAIRINGS upstream.
        data_payload = {
            "pairings": [
                _slim_pairing(pairing)
                for pairing in payload["pairings"]
                if isinstance(pairing, dict)
            ],
            "hotels": [
                _strip_empty({k: v for k, v in hotel.items() if k in _HOTEL_KEEP})
                for hotel in payload["hotels"]
                if isinstance(hotel, dict)
            ],
            "activities": [
                activity["name"]
                for activity in payload["activities"]
                if isinstance(activity, dict) and activity.get("name")
            ],
            "restaurants": [
                _strip_empty({k: v for k, v in rest.items() if k in _RESTAURANT_KEEP})
                for rest in payload["restaurants"]
                if isinstance(rest, dict)
            ],
            "weather": payload["weather"],
            "best_time": payload["best_time"],
        }

        # Round all floats so prices/ratings read cleanly in the prompt
        # (e.g. 4.4399999999999995 -> 4.44) and flow through to the rendered plan.
        trip_context = _round_numbers(trip_context)
        data_payload = _round_numbers(data_payload)

        messages = [
            {"role": "system", "content": _CURATION_PROMPT},
            {
                "role": "user",
                "content": (
                    "<trip_context>\n"
                    f"{json.dumps(trip_context, indent=2, sort_keys=True)}\n"
                    "</trip_context>\n\n"
                    "<travel_payload>\n"
                    f"{json.dumps(data_payload, indent=2, sort_keys=True)}\n"
                    "</travel_payload>"
                ),
            },
        ]

        if _DEBUG:
            _PROMPT_DUMP_PATH.parent.mkdir(parents=True, exist_ok=True)
            _PROMPT_DUMP_PATH.write_text(json.dumps(messages, indent=2), encoding="utf-8")

        _t1 = time.perf_counter()
        curation: TravelPlanCuration = self.curation_model.invoke(messages)
        _llm_ms = (time.perf_counter() - _t1) * 1000

        if _DEBUG:
            _payload_chars = len(messages[1]["content"])
            print(
                f"[travel_agent] payload_build={_build_ms:,.0f}ms  "
                f"llm_invoke={_llm_ms:,.0f}ms  "
                f"payload_chars={_payload_chars:,}  "
                f"prompt -> {_PROMPT_DUMP_PATH}",
                flush=True,
            )

        budget = payload.get("total_budget")
        plan = TravelPlan(
            **curation.model_dump(),
            origin=payload.get("origin"),
            destination=payload.get("destination"),
            trip_days=payload["trip_days"],
            trip_start=payload.get("trip_start"),
            total_budget=float(budget) if isinstance(budget, (int, float)) else None,
            weather=payload["weather"],
            best_time=payload["best_time"],
            lowest_total_estimate=payload["costs"].get("lowest_total_estimate"),
            lowest_group_estimate=payload["costs"].get("lowest_group_estimate"),
            travelers_label=payload.get("travelers_label"),
        )

        return {"travel_plan": plan.model_dump(),
                "has_existing_trip_context": True}
