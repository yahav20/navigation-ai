"""HITL node: show the travel plan and ask the user whether to build the itinerary."""
from langgraph.types import interrupt

from agent.core.state import AgentState


class TravelConfirmationNode:
    """Pause after the travel formatter and ask the user what to do next.

    Presents two choices:
      yes  → proceed to plan_check which resolves hotel/flight and launches
              the itinerary builder (itinerary_mode="with_travel_data").
      no   → end the turn; the user keeps the travel plan without an itinerary.

    Skips the interrupt entirely when the state has no travel plan (e.g. the
    formatter ran but travel_agent produced nothing), so the graph goes straight
    to summary in that edge case.
    """

    def __call__(self, state: AgentState) -> dict:
        """Interrupt and ask; return intent update on resume."""
        travel_plan = state.get("travel_plan") or {}
        destination = state.get("destination_city") or "your destination"

        if not travel_plan.get("hotels") or not state.get("has_flights"):
            return {}

        user_choice: str = interrupt({
            "question": (
                f"Here's your travel plan to **{destination}** with flights and hotels. "
                "Would you like me to build a full day-by-day itinerary based on these selections?"
            ),
            "options": [
                ("yes", "Yes, build the itinerary"),
                ("no",  "No thanks, this is enough for now"),
            ],
        })

        if user_choice.strip().lower() in ("yes", "y"):
            # Signal plan_check to treat existing flights+hotels as confirmed.
            # plan_check will call _resolve_travel_data and set itinerary_mode.
            return {"intent": "build_itinerary"}

        return {}
