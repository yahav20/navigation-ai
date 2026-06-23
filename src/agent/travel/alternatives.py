"""Suggest reachable alternative destinations when the original route is empty."""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from agent.core.llm import silent
from agent.core.state import AgentState
from providers.flights import search_flights_with_fallback
from tools.dependencies import data_provider


class AlternativeSuggestion(BaseModel):
    """Single alternative city suggestion picked by the LLM."""

    city: str = Field(description="The suggested alternative city")
    country: str = Field(description="The country the city is in")
    reason: str = Field(description="Brief explanation of why this is a good alternative")


class AlternativeDestinations(BaseModel):
    """Container for alternative destination suggestions."""

    suggestions: list[AlternativeSuggestion] = Field(
        default_factory=list,
        description="1-3 alternative destinations the traveler can actually book",
    )


class AlternativeDestinationNode:
    """Pick reachable alternative cities when the requested route has no flights."""

    def __init__(self, extraction_model: BaseChatModel) -> None:
        """Store the extraction model used to rank alternative cities."""
        self.extraction_model = extraction_model

    def __call__(self, state: AgentState) -> dict:
        destination = state.get("destination_city")
        origin = state.get("current_city")
        if not destination or not origin:
            return {"alternative_destinations": []}

        candidates = data_provider.get_reachable_destinations_by_distance(
            origin, destination, 10,
        )

        usable = [c for c in candidates if isinstance(c, dict) and c.get("city")]
        for _c in usable:
            pass
        if not usable:
            return {"alternative_destinations": []}

        candidate_lines = "\n".join(
            f"- {c['city']}, {c.get('country', 'Unknown')} "
            f"({round(c.get('distance_km', 0))} km from {destination})"
            for c in usable
        )

        prompt = f"""The traveler wanted to fly from "{origin}" to "{destination}" but no flights to "{destination}" are available.
The candidates below are cities that ARE reachable from "{origin}" by flight, ordered by how close they are to "{destination}".

Pick 2-3 that would make genuinely good alternative destinations — favoring options that are
both reasonably close to "{destination}" (same region/continent) AND comparable in culture,
vibe, or traveler appeal. Discard candidates that are dramatically farther than "{destination}"
itself would have been (e.g. trans-continental detours). Briefly explain each pick.

Each pick must be a DIFFERENT city — never repeat the same city across suggestions.

Candidates:
{candidate_lines}
"""

        picker = silent(self.extraction_model.with_structured_output(AlternativeDestinations))
        try:
            result: AlternativeDestinations = picker.invoke(prompt)
            shortlist = [s.model_dump() for s in result.suggestions]
        except Exception:  # noqa: BLE001
            return {"alternative_destinations": []}

        budget = state.get("total_budget")
        budget_optional = state.get("budget_optional", False)
        apply_budget = bool(budget) and not budget_optional
        trip_days = state.get("trip_days") or 2
        trip_start = state.get("trip_start")

        enriched = []
        seen_cities = set()
        for pick in shortlist:
            city = pick.get("city")
            if not city:
                continue
            key = city.strip().lower()
            if key in seen_cities:
                continue
            seen_cities.add(key)

            raw_flights = search_flights_with_fallback(origin, city, trip_start) or []
            flights = [
                f for f in raw_flights
                if isinstance(f, dict) and "flight_number" in f
                and isinstance(f.get("price"), (int, float))
            ]

            raw_hotels = data_provider.fetch_hotels(city) or []
            hotels = [
                h for h in raw_hotels
                if isinstance(h, dict)
                and isinstance(h.get("price_per_night"), (int, float))
            ]

            if apply_budget and flights and hotels:
                cheapest_flight = min(f["price"] for f in flights)
                cheapest_hotel = min(h["price_per_night"] for h in hotels)
                flights = [f for f in flights if f["price"] + cheapest_hotel * trip_days <= budget]
                hotels = [h for h in hotels if cheapest_flight + h["price_per_night"] * trip_days <= budget]

            if not flights or not hotels:
                continue

            enriched.append({**pick, "flights": flights, "hotels": hotels})

        return {"alternative_destinations": enriched}


class FormatterAlternativeNode:
    """Render the alternative-destination Markdown response deterministically."""

    def __call__(self, state: AgentState) -> dict:
        """Format the alternative-destination response as Markdown."""
        alternatives = state.get("alternative_destinations") or []
        original_destination = state.get("destination_city", "your destination")
        origin = state.get("current_city", "your origin")
        budget = state.get("total_budget")

        if not alternatives:
            budget_line = (
                f" within your **${budget:g}** budget"
                if isinstance(budget, (int, float)) and budget
                else ""
            )
            text = (
                f"Unfortunately, we could not find any flights from **{origin}** to "
                f"**{original_destination}**, and no reachable alternative destinations "
                f"from **{origin}**{budget_line} are available in our database.\n\n"
                "Try a different origin or destination, or adjust your budget."
            )
            return {"messages": [AIMessage(content=text)]}

        text = _render_markdown(origin, original_destination, budget, alternatives)
        return {"messages": [AIMessage(content=text)],
                 "has_existing_trip_context": True}


def _render_markdown(origin: str, original_destination: str, budget: float | None,
                      alternatives: list[dict]) -> str:
    """Render the alternative-destinations payload as the agreed Markdown template."""
    budget_text = f"${budget:g}" if isinstance(budget, (int, float)) and budget else "Not specified"

    lines = [
        f"Good news — while **{original_destination}** isn't reachable from **{origin}** "
        "right now, we found some great alternatives for you.",
        "",
        f"Unfortunately, we could not find any flights from **{origin}** to "
        f"**{original_destination}**. Below are reachable alternatives that fit your trip.",
        "",
        "---",
        "",
        "### 🌍 **Suggested Alternatives**",
        "",
        f"**Total Budget:** {budget_text}",
        "",
        "---",
        "",
    ]

    for i, alt in enumerate(alternatives, start=1):
        city = alt.get("city", "Unknown")
        country = alt.get("country", "Unknown")
        reason = alt.get("reason", "")
        flights = alt.get("flights") or []
        hotels = alt.get("hotels") or []

        lines.append(f"#### ✈️ **Option {i} — {city}, {country}**")
        lines.append("")
        lines.append(f"*Why this alternative:* {reason}")
        lines.append("")
        lines.append(f"**Flights from {origin}:**")
        if flights:
            lines.extend(
                f"* **Airline:** {f.get('airline', 'Unknown')}  \n"
                f"  **Flight Number:** {f.get('flight_number', 'Unknown')}  \n"
                f"  **Price:** ${f.get('price', 0):g}"
                for f in flights
            )
        else:
            lines.append("No flights available.")
        lines.append("")
        lines.append(f"**Hotels in {city}:**")
        if hotels:
            lines.extend(
                f"* **{h.get('name', 'Unknown')}** — {h.get('stars', '?')} stars, "
                f"${h.get('price_per_night', 0):g}/night"
                for h in hotels
            )
        else:
            lines.append("No hotels within your budget for this city.")
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append(
        "Let us know if any of these spark your interest, or if you'd like to adjust "
        "your budget or pick a different region!",
    )
    return "\n".join(lines)
