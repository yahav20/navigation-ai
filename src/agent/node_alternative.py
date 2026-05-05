from typing import List
from pydantic import BaseModel, Field
from agent.state import AgentState
from agent.node import get_models
from tools.tools import data_provider

_, extraction_model = get_models()


class AlternativeSuggestion(BaseModel):
    city: str = Field(description="The suggested alternative city")
    country: str = Field(description="The country the city is in")
    reason: str = Field(description="Brief explanation of why this is a good alternative")


class AlternativeDestinations(BaseModel):
    suggestions: List[AlternativeSuggestion] = Field(
        default_factory=list,
        description="1-3 alternative destinations the traveler can actually book",
    )


def alternative_destination_node(state: AgentState):
    """
    Triggered when fetch_flights returns no results.
    Pulls candidate cities the traveler can actually reach from their
    origin (i.e. cities that have at least one flight from `current_city`),
    ranked by closeness to the originally requested destination. The LLM
    then picks 2-3 that are similar in character and reasonably nearby.
    Skipped entirely when no origin is known.
    """
    destination = state.get("destination_city")
    origin = state.get("current_city")
    if not destination or not origin:
        return {"alternative_destinations": []}

    candidates = data_provider.get_reachable_destinations_by_distance(
        origin, destination, 10
    )

    # If lookup hit a message-only payload (city not found, JSON provider, etc.)
    usable = [c for c in candidates if isinstance(c, dict) and c.get("city")]
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

Candidates:
{candidate_lines}
"""

    picker = extraction_model.with_structured_output(AlternativeDestinations)
    try:
        result: AlternativeDestinations = picker.invoke(prompt)
        shortlist = [s.model_dump() for s in result.suggestions]
    except Exception as exc:
        print(f"--- alternative_destination_node failed: {exc} ---")
        shortlist = []

    return {"alternative_destinations": shortlist}
