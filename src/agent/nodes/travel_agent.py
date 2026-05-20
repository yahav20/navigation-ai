"""Deterministic travel agent node: curates a structured plan for the formatter."""

import json
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

from agent.models import TravelPlan, TravelPlanCuration
from agent.state import AgentState
from tools.dependencies import data_provider
from security import SECURITY_RULES


SEASONS = ("Spring", "Summer", "Autumn", "Winter")

_CURATION_PROMPT = """You are Atlas, a deterministic travel agent.

You receive a JSON payload inside <travel_payload> containing flights, hotels, activities, weather, best_time, costs, and an `is_adjustment` flag. Your job is to curate the trip and return a structured plan.

Rules:
1. Use ONLY items present in the payload. Never invent flights, hotels, activities, prices, or dates.
2. Pick 1-3 flights and 1-3 hotels from the payload. For each candidate, check whether it is a good match for the budget (when costs.budget_applied=true). "Good match" is NOT just "cheaper than budget" — when the budget is generous relative to the option's cost, premium/higher-priced options are also a good fit. Example: a $600/night hotel for 4 nights ($2400) is a good fit for a $5000 budget, even if cheaper hotels exist. Don't drop an option just because a cheaper one exists — drop it only if it genuinely doesn't fit the budget.
3. Pick up to 5 activities that respect user_preferences (e.g. dietary_restrictions, preferred_location).
4. For each pick, write a short one-line description of why it fits.
4a. For each flight pick: if the source flight has a `route` array (connecting flight), copy each leg into `legs` as {from_city, to_city, airline, flight_number}. For direct flights (no `route`), leave `legs` empty.
5. `intro` and `sign_off` should be one short sentence each, friendly but concise.
6. All prices in USD. Use the values from the payload as-is.
7. Framing based on is_adjustment:
   - is_adjustment=true: the user is updating an existing plan. `intro` MUST start with "✅ Trip Updated Successfully!" and acknowledge the changes. `sign_off` should reassure them the new plan reflects their requested updates.
   - is_adjustment=false: this is a brand-new plan. `intro` should welcome them to their plan (no "updated" framing). `sign_off` should encourage them to confirm or refine.
"""

_NO_FLIGHTS_NEW = (
    "Based on our search, we could not find available flights from "
    "{origin} to {destination} at this time."
)
_NO_FLIGHTS_ADJUSTMENT = (
    "⚠️ **Update Failed:** I tried to update your trip, but I couldn't find any "
    "available flights from {origin} to {destination} for the new request."
)
_NO_HOTELS_NEW = (
    "I found flights for this route, but no hotels that match the available "
    "inventory and budget constraints."
)
_NO_HOTELS_ADJUSTMENT = (
    "⚠️ **Update Failed:** I found flights for your updated request, but no "
    "hotels fit the newly adjusted budget or preferences."
)


def _is_message(item: object) -> bool:
    return isinstance(item, dict) and bool(item.get("message"))


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


def _build_costs(
    flights: list[dict],
    hotels: list[dict],
    trip_days: int,
    budget: float | None,
    *,
    budget_optional: bool,
) -> dict[str, Any]:
    options = []
    apply_budget = isinstance(budget, (int, float)) and not budget_optional

    for flight in flights:
        flight_cost = _flight_price(flight)
        if flight_cost is None:
            continue

        for hotel in hotels:
            nightly_rate = _hotel_price(hotel)
            if nightly_rate is None:
                continue

            hotel_total = nightly_rate * trip_days
            total = flight_cost + hotel_total
            options.append({
                "flight": _flight_label(flight),
                "hotel": hotel.get("name", "hotel option"),
                "breakdown": {
                    "flight": flight_cost,
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
    }


def _affordable_hotel_names(
    hotels: list[dict],
    flights: list[dict],
    trip_days: int,
    budget: float,
) -> set[str]:
    flight_costs = [c for c in (_flight_price(f) for f in flights) if c is not None]
    if not flight_costs:
        return set()
    cheapest_flight = min(flight_costs)
    names: set[str] = set()
    for hotel in hotels:
        nightly = _hotel_price(hotel)
        if nightly is None:
            continue
        if cheapest_flight + nightly * trip_days <= budget:
            names.add(hotel.get("name", ""))
    names.discard("")
    return names


def _filter_hotels_by_budget(
    hotels: list[dict],
    flights: list[dict],
    trip_days: int,
    costs: dict[str, Any],
    budget: float | None,
) -> list[dict]:
    if not costs.get("budget_applied") or not isinstance(budget, (int, float)):
        return hotels
    keep = _affordable_hotel_names(hotels, flights, trip_days, float(budget))
    return [hotel for hotel in hotels if hotel.get("name") in keep]


def build_travel_prompt_payload(state: AgentState) -> dict:
    """Build the complete structured payload for the deterministic travel agent."""
    destination = state.get("destination_city")
    trip_days = int(state.get("trip_days") or 3)
    preferences = state.get("user_preferences") or {}

    flights = _valid_items(state.get("flight_options", []))
    hotels = []
    activities = []
    weather = {}
    best_time = {}

    if destination:
        hotels = _valid_items(
            data_provider.fetch_hotels(destination, max_price=_hotel_max_price(preferences)),
        )
        hotels = _apply_hotel_preferences(hotels, preferences)
        activities = _valid_items(data_provider.fetch_activities(destination))
        weather = _fetch_weather(destination)

        best_time_result = data_provider.get_best_time_to_visit(destination)
        if isinstance(best_time_result, dict) and not _is_message(best_time_result):
            best_time = best_time_result

    budget = state.get("total_budget")
    budget_value = budget if isinstance(budget, (int, float)) else None
    costs = _build_costs(
        flights,
        hotels,
        trip_days,
        budget_value,
        budget_optional=bool(state.get("budget_optional", False)),
    )
    hotels = _filter_hotels_by_budget(hotels, flights, trip_days, costs, budget_value)

    return {
        "origin": state.get("current_city"),
        "destination": destination,
        "total_budget": budget,
        "budget_optional": bool(state.get("budget_optional", False)),
        "trip_days": trip_days,
        "user_preferences": preferences,
        "flights": flights,
        "hotels": hotels,
        "activities": activities,
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

        if not payload["flights"]:
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
            total_budget=float(budget) if isinstance(budget, (int, float)) else None,
            weather=payload["weather"],
            best_time=payload["best_time"],
            lowest_total_estimate=payload["costs"].get("lowest_total_estimate"),
        )

        return {"travel_plan": plan.model_dump()}
