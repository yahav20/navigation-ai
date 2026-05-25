"""Extract travel metadata from the conversation history."""
# src/agent/nodes/metadata.py
from langchain_core.language_models import BaseChatModel

from agent.models import TravelMetadata
from agent.state import AgentState


class MetadataNode:
    """Pull travel metadata fields out of the recent chat history."""

    def __init__(self, extraction_model: BaseChatModel) -> None:
        """Store the extraction model used to read structured metadata."""
        self.extraction_model = extraction_model

    def __call__(self, state: AgentState) -> dict:
        """Return updated agent state with any newly extracted metadata."""
        updates = {"step_count": state.get("step_count", 0)}

        if not state.get("messages"):
            return updates

        messages = state.get("messages", [])
        recent_messages = [
            msg for msg in messages[-10:]
            if getattr(msg, "type", "") in ("human", "ai")
            and not getattr(msg, "tool_calls", None)
        ][-6:]

        extractor = self.extraction_model.with_structured_output(TravelMetadata)

        current_trip_days = state.get("trip_days")
        existing_summary = state.get("summary", "")

        metadata: TravelMetadata = extractor.invoke([
            {
                "role": "system",
                "content": f"""
                Extract travel metadata from the conversation.
                Only fill a field if it is explicitly mentioned or very clear.
                Do not guess. If a field is missing, return null.
                Extract: current_city, destination_city, budget, trip_days.

                IMPORTANT — current_city and destination_city must be CITIES, not countries.
                If the user names a country, resolve it to the primary departure/arrival city:
                  "Israel" / "Israeli" → "Tel Aviv"
                  "France" → "Paris"
                  "UK" / "England" / "Britain" → "London"
                  "USA" / "United States" / "America" → "New York City" (or whatever city they imply)
                  "Japan" → "Tokyo"
                  "Germany" → "Berlin"
                  "Netherlands" / "Holland" → "Amsterdam"

                CONVERSATION MEMORY (from previous turns):
                {existing_summary or "No previous context."}

                Use this memory to resolve references like "there", "that city", "the same place",
                or "the first one" — e.g. if memory says the agent recommended Berlin and the user
                says "let's go there", extract destination_city as "Berlin".

                IMPORTANT — trip_days resolution:
                The current trip duration in state is: {current_trip_days} days.
                If the user says something relative like "increase by 2 days", "add 3 days",
                "make it longer by 1 day", or "reduce by 2 days", compute the new absolute
                value from the current state value and return that number.
                Example: current is 5, user says "add 2 days" → return 7.
                Example: current is 5, user says "reduce by 1 day" → return 4.
                If the user gives an absolute number like "6 days", return that directly.
                """,
            },
            *recent_messages,
        ])

        old_origin = state.get("current_city", "").lower() if state.get("current_city") else ""
        old_dest = state.get("destination_city", "").lower() if state.get("destination_city") else ""
        old_budget = state.get("total_budget")
        old_days = state.get("trip_days")

        if metadata.current_city is not None:
            new_origin = metadata.current_city.split(",")[0].strip()
            updates["current_city"] = new_origin
            # If origin changed, invalidate old plans
            if old_origin and new_origin.lower() != old_origin:
                updates["travel_plan"] = {}
                updates["itinerary_plan"] = {}
                updates["flight_options"] = []
                updates["has_flights"] = False

        if metadata.destination_city is not None:
            new_dest = metadata.destination_city.split(",")[0].strip()
            updates["destination_city"] = new_dest
            # If destination changed, invalidate old plans
            if old_dest and new_dest.lower() != old_dest:
                updates["travel_plan"] = {}
                updates["itinerary_plan"] = {}
                updates["alternative_destinations"] = []
                updates["flight_options"] = []
                updates["has_flights"] = False

        if metadata.budget is not None:
            updates["total_budget"] = metadata.budget
            if old_budget is not None and metadata.budget != old_budget:
                updates["travel_plan"] = {}
                updates["itinerary_plan"] = {}
                updates["flight_options"] = []
                updates["has_flights"] = False

        if metadata.trip_days is not None:
            updates["trip_days"] = metadata.trip_days
            if old_days is not None and metadata.trip_days != old_days:
                updates["travel_plan"] = {}
                updates["itinerary_plan"] = {}
                updates["flight_options"] = []
                updates["has_flights"] = False

        return updates