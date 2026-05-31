"""Deterministic travel agent node: curates a structured plan for the formatter."""

import json
from datetime import date, datetime
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

from agent.models import TravelPlan, TravelPlanCuration
from agent.state import AgentState
from tools.dependencies import data_provider
from security import SECURITY_RULES

# Penalty (USD per day) added to a pairing's score for each day its outbound→return
# gap deviates from the requested trip_days. Tuned so a $50 cheaper pairing one day
# off-target beats the on-target pairing, but a 5-day-off pairing has to be much
# cheaper to win. Keeps the agent honest about trip length without dropping the
# only available options when the date grid is sparse.
_DAY_GAP_PENALTY = 30.0
_MAX_PAIRINGS = 12


SEASONS = ("Spring", "Summer", "Autumn", "Winter")

_CURATION_PROMPT = """You are Atlas, a deterministic travel agent.

You receive a JSON payload inside <travel_payload> containing flights (outbound),
return_flights, a pre-computed `pairings` array, hotels, activities, restaurants,
weather, best_time, costs, trip_days, trip_start, and an `is_adjustment` flag.
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
3. Pick **exactly 3 hotels** from the payload's **`hotels` array** (fewer only if that array has fewer than 3 entries). Do NOT pick from `activities` — those are sightseeing items, not accommodation. Aim for **price variety**: one budget-friendly, one mid-range, one premium — so the user has real choices. When costs.budget_applied=true, all 3 must fit within the budget (flight_outbound + cheapest_return + price_per_night × trip_days ≤ total_budget). Use the `price_per_night` value from the payload as-is — never write $0 for a hotel.
4. Pick up to 5 activities. Activities with `source: "api"` or `source: "hybrid"` carry live ratings from Google Maps — **prefer these** over `source: "local"` fixtures. Local fixtures are fallback only when no API activity exists in the payload. Respect user_preferences (e.g. dietary_restrictions, preferred_location).
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


def _is_message(item: object) -> bool:
    return isinstance(item, dict) and bool(item.get("message"))


def _is_lodging(activity: dict) -> bool:
    """Return True when a Google Maps activity is a hotel/lodging that should be excluded.

    Checks both the place type tags returned by Google AND the name prefix, because
    Google sometimes returns real hotels under tourist_attraction without a 'lodging' tag.
    """
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
        # Only apply the filter if at least one kosher hotel exists; otherwise keep all
        # and let the agent explain the limitation rather than returning an empty list.
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
    """Best-effort extraction of the departure date from an offer."""
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
    """Identity key for deduping pairings by which outbound/return flights they use."""
    return (flight.get("airline"), flight.get("flight_number") or _flight_label(flight))


def _build_pairings(
    outbound: list[dict],
    return_flights: list[dict],
    trip_days: int,
    *,
    limit: int = _MAX_PAIRINGS,
) -> list[dict]:
    """Pre-compute outbound + return pairings scored by price + date-gap deviation.

    The score penalises pairings whose actual day gap differs from `trip_days`,
    so the agent gets a list that's already date-aware. When dates can't be
    parsed (DB-fallback rows, weird ISO strings), the pair is still produced
    but with `day_gap` null and zero day-gap penalty — total_price wins by default.
    """
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
                    continue  # return must be after outbound
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

    # Dedupe by (outbound flight, return flight) keeping the best-scored entry.
    seen: set[tuple] = set()
    diverse: list[dict] = []
    for cand in candidates:
        key = (_flight_identity(cand["outbound"]), _flight_identity(cand["return_flight"]))
        if key in seen:
            continue
        seen.add(key)
        cand.pop("_score", None)  # internal; don't leak to the LLM
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
) -> dict[str, Any]:
    options = []
    apply_budget = isinstance(budget, (int, float)) and not budget_optional
    return_floor = _cheapest_flight_price(return_flights) or 0.0

    for flight in flights:
        flight_cost = _flight_price(flight)
        if flight_cost is None:
            continue

        for hotel in hotels:
            nightly_rate = _hotel_price(hotel)
            if nightly_rate is None:
                continue

            hotel_total = nightly_rate * trip_days
            total = flight_cost + return_floor + hotel_total
            options.append({
                "flight": _flight_label(flight),
                "hotel": hotel.get("name", "hotel option"),
                "breakdown": {
                    "flight_outbound": flight_cost,
                    "flight_return": return_floor,
                    "hotel_total": hotel_total,
                    "days": trip_days,
                },
                "total_estimate": total,
                "currency": "USD",
                "within_budget": total <= budget if apply_budget else None,
            })

    options = sorted(options, key=lambda item: item["total_estimate"])
    affordable_options = [item for item in options if item["within_budget"] is True]
    visible_options = affordable_options if apply_budget else options

    return {
        "options": visible_options[:10],
        "lowest_total_estimate": visible_options[0]["total_estimate"] if visible_options else None,
        "budget_applied": apply_budget,
        "return_flight_floor": return_floor,
    }


def _affordable_hotel_names(
    hotels: list[dict],
    flights: list[dict],
    return_flights: list[dict],
    trip_days: int,
    budget: float,
) -> set[str]:
    flight_costs = [c for c in (_flight_price(f) for f in flights) if c is not None]
    if not flight_costs:
        return set()
    cheapest_outbound = min(flight_costs)
    cheapest_return = _cheapest_flight_price(return_flights) or 0.0
    names: set[str] = set()
    for hotel in hotels:
        nightly = _hotel_price(hotel)
        if nightly is None:
            continue
        if cheapest_outbound + cheapest_return + nightly * trip_days <= budget:
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
) -> list[dict]:
    if not costs.get("budget_applied") or not isinstance(budget, (int, float)):
        return hotels
    keep = _affordable_hotel_names(hotels, flights, return_flights, trip_days, float(budget))
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
        # Hotels: use "api" approach (reads from api_hotels cache / Xotelo live).
        # Fall back to local fixtures only when the API returns nothing.
        _api_hotels = _valid_items(data_provider.fetch_hotels(destination, approach="api"))
        if _api_hotels:
            hotels = _apply_hotel_preferences(_api_hotels, preferences)
        else:
            hotels = _apply_hotel_preferences(
                _valid_items(data_provider.fetch_hotels(destination, approach="local")),
                preferences,
            )

        # Activities: use "api" approach (reads from api_attractions cache / Google Maps live).
        # Fall back to local fixtures only when the API returns nothing.
        # Exclude lodging entries — Google Maps sometimes returns hotels as tourist attractions,
        # which confuses the LLM into treating them as activities with price=0.
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

    # Build costs using all candidate hotels (before budget filter)
    costs = _build_costs(
        flights,
        return_flights,
        hotels,
        trip_days,
        budget_value,
        budget_optional=bool(state.get("budget_optional", False)),
    )
    hotels = _filter_hotels_by_budget(hotels, flights, return_flights, trip_days, costs, budget_value)

    pairings = _build_pairings(flights, return_flights, trip_days)

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
    }


class TravelAgentNode:
    """Curate the deterministic payload into a structured TravelPlan for the formatter."""

    def __init__(self, response_model: Runnable) -> None:
        """Wrap the response model with structured output for curation."""
        self.curation_model = response_model.with_structured_output(TravelPlanCuration)

    def __call__(self, state: AgentState) -> dict:
        """Return either `travel_plan` (success) or a fallback `AIMessage` (no data)."""
        if not state.get("messages"):
            return {}

        payload = build_travel_prompt_payload(state)
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

        curation: TravelPlanCuration = self.curation_model.invoke([
            {"role": "system", "content": _CURATION_PROMPT},
            {
                "role": "user",
                "content": (
                    "<travel_payload>\n"
                    f"{json.dumps(payload, indent=2, sort_keys=True)}\n"
                    "</travel_payload>"
                ),
            },
        ])

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
        )

        return {"travel_plan": plan.model_dump(),
                "has_existing_trip_context": True}  # signal to router that we now have active trip context after producing a plan
