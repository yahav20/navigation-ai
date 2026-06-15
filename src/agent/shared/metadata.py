"""Extract travel metadata from the conversation history."""
from datetime import date

from langchain_core.language_models import BaseChatModel

from agent.core.llm import silent
from agent.core.models import TravelMetadata
from agent.core.state import AgentState
from agent.shared.travelers import apply_traveler_updates


def _same_month_refinement(old: str | None, new: str) -> bool:
    """True when old and new describe the same month at different precision
    (e.g. '2026-06' vs '2026-06-14') — a refinement, not a reschedule.

    Guards against spuriously invalidating an already-found travel plan when the
    extractor re-reads a concrete flight date from the rendered plan in history
    while state still holds the user's original month-level start.
    """
    if not old:
        return False
    short, long = sorted((old, new), key=len)
    return len(short) == 7 and len(long) == 10 and long.startswith(short)


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

        extractor = silent(self.extraction_model.with_structured_output(TravelMetadata))

        current_trip_days = state.get("trip_days")
        current_adults = state.get("num_adults")
        current_children = state.get("num_children")
        existing_summary = state.get("summary", "")
        today_iso = date.today().isoformat()

        metadata: TravelMetadata = extractor.invoke([
            {
                "role": "system",
                "content": f"""
                Extract travel metadata from the conversation.
                Only fill a field if it is explicitly mentioned or very clear.
                Do not guess. If a field is missing, return null.
                Extract: current_city, destination_city, budget, trip_days, trip_start,
                num_adults, num_children, num_rooms, num_rooms_delta.

                Today's date is {today_iso}.

                For current_city and destination_city, extract the place the user
                actually named — a city OR a country — verbatim. Do NOT resolve a
                country to a city yourself; a later step handles that. For example,
                if the user says "start in Israel", return current_city "Israel".

                CONVERSATION MEMORY (from previous turns):
                {existing_summary or "No previous context."}

                Use this memory to resolve references like "there", "that city", "the same place",
                or "the first one" — BUT ONLY when the memory points to a SINGLE unambiguous
                destination. If the memory mentions MULTIPLE possible destinations (e.g. the agent
                recommended Berlin and London as options) and the user's message uses a vague
                reference without naming a specific city, return null for destination_city so
                the system can ask the user to clarify which one they meant.

                IMPORTANT — trip_days resolution:
                The current trip duration in state is: {current_trip_days} days.
                If the user says something relative like "increase by 2 days", "add 3 days",
                "make it longer by 1 day", or "reduce by 2 days", compute the new absolute
                value from the current state value and return that number.
                Example: current is 5, user says "add 2 days" → return 7.
                Example: current is 5, user says "reduce by 1 day" → return 4.
                If the user gives an absolute number like "6 days", return that directly.

                IMPORTANT — trip_start resolution:
                Convert phrases about when the trip happens into a calendar string.
                - Specific day ("June 10", "leaving August 3rd", "on 2026-07-15") -> YYYY-MM-DD.
                - Month or season only ("in June", "next month", "around August", "late autumn")
                  -> YYYY-MM. Resolve relative phrases against today's date above.
                - If the user says nothing about timing, return null.
                Never invent a date when none was implied.

                IMPORTANT — travellers (adults / children) resolution:
                The current group in state is: {current_adults} adults, {current_children} children.
                - Return ABSOLUTE counts. If the user gives an absolute number
                  ("2 adults and 1 kid", "we're 4 people"), return that directly.
                  An unqualified count of people/travellers means adults unless
                  children are explicitly named.
                - If the user says something relative ("2 more adults joined",
                  "4 people are joining", "one fewer child"), compute the new
                  absolute value from the current state value and return that.
                  Example: current 2 adults, "4 more people joined" -> num_adults 6.
                - Return null for adults/children the user did not mention.

                IMPORTANT — hotel rooms resolution:
                - Set num_rooms ONLY when the user names a specific room TOTAL
                  ("book 3 rooms", "we need 2 rooms"). Never calculate rooms from
                  the number of people — the system derives that itself.
                - Set num_rooms_delta to +1 for "add a room"/"one more room" and -1
                  for "remove a room"/"one less room".
                - Return null for both when the user says nothing about rooms.
                """,
            },
            *recent_messages,
        ])

        old_origin = state.get("current_city", "").lower() if state.get("current_city") else ""
        old_dest = state.get("destination_city", "").lower() if state.get("destination_city") else ""
        old_budget = state.get("total_budget")
        old_start = state.get("trip_start")

        # On a build_itinerary turn the user is building ON the existing plan, not
        # changing it. extract_metadata still re-reads recent messages — which now
        # include the bot's own rendered plan (concrete flight dates etc.) — so a
        # spurious re-extraction must NOT wipe the just-created travel data. Field
        # updates below still run; only the destructive reset is suppressed. (If no
        # plan exists yet, there is nothing to lose, so this is safe either way.)
        suppress_invalidation = state.get("intent") == "build_itinerary"

        def _invalidate_flights(reset_alternatives: bool = False) -> None:
            if suppress_invalidation:
                return
            updates["travel_plan"] = {}
            updates["itinerary_plan"] = {}
            updates["flight_options"] = []
            updates["return_flight_options"] = []
            updates["has_flights"] = False
            if reset_alternatives:
                updates["alternative_destinations"] = []

        if metadata.current_city is not None:
            new_origin = metadata.current_city.split(",")[0].strip()
            updates["current_city"] = new_origin
            if old_origin and new_origin.lower() != old_origin:
                _invalidate_flights()

        if metadata.destination_city is not None:
            new_dest = metadata.destination_city.split(",")[0].strip()
            updates["destination_city"] = new_dest
            if old_dest and new_dest.lower() != old_dest:
                _invalidate_flights(reset_alternatives=True)

        if metadata.budget is not None:
            updates["total_budget"] = metadata.budget
            if old_budget is not None and metadata.budget != old_budget:
                _invalidate_flights()

        if metadata.trip_days is not None:
            updates["trip_days"] = metadata.trip_days
            # trip_days-only changes keep the same route/hotel — no re-search needed
            # (matches AdjustmentsNode's documented behavior).

        if metadata.trip_start is not None and not _same_month_refinement(old_start, metadata.trip_start):
            updates["trip_start"] = metadata.trip_start
            if old_start is not None and metadata.trip_start != old_start:
                _invalidate_flights()

        # Travellers / rooms. Scoped first step: these intentionally do NOT
        # invalidate flights or trigger a re-search — they only touch the new
        # group-size and room fields (downstream cost effects come later).
        updates.update(apply_traveler_updates(
            state,
            new_adults=metadata.num_adults,
            new_children=metadata.num_children,
            new_rooms_abs=metadata.num_rooms,
            rooms_delta=metadata.num_rooms_delta,
        ))

        return updates
