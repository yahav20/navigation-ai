"""Detect and process user adjustments to travel parameters."""
from langchain_core.language_models import BaseChatModel

from agent.core.llm import silent
from agent.core.models import TravelAdjustments
from agent.core.state import AgentState
from agent.shared.budget import resolve_budget
from agent.shared.travelers import apply_traveler_updates


class AdjustmentsNode:
    """Check if the user wants to change destination, origin, or budget during the conversation."""

    def __init__(self, extraction_model: BaseChatModel) -> None:
        """Bind the Adjustments extraction model."""
        self.extraction_model = silent(extraction_model.with_structured_output(TravelAdjustments))

    def __call__(self, state: AgentState) -> dict:
        """Return state updates for explicit trip-parameter changes."""
        messages = state.get("messages", [])
        if not messages:
            return {}

        last_msg = messages[-1]
        if getattr(last_msg, "type", "") != "human":
            return {}

        if not state.get("destination_city") and not state.get("current_city"):
            return {}

        prompt = f"""
        Current Trip State:
        - Origin: {state.get("current_city", "None")}
        - Destination: {state.get("destination_city", "None")}
        - Budget: {state.get("total_budget", "None")}
        - Trip Days: {state.get("trip_days", "None")}
        - Adults: {state.get("num_adults", "None")}
        - Children: {state.get("num_children", "None")}
        - Hotel Rooms: {state.get("num_rooms", "None")}

        User's latest message: "{last_msg.content}"

        Is the user explicitly asking to adjust any of these parameters?
        For budget: if the user states an absolute total directly ("make it $9000"),
        return new_budget. If the user says something relative in dollars ("add $10000",
        "raise it by $500", "reduce by $300"), return ONLY the signed dollar amount as
        new_budget_delta (e.g. "add $10000" -> 10000) — do NOT add it to the current
        budget yourself. If relative as a percent ("increase budget by 20%"), return
        ONLY the signed percent as new_budget_delta_pct (e.g. -> 20) — do NOT compute
        the resulting dollar amount yourself.
        For travellers/rooms return ABSOLUTE counts (resolve "2 more adults"
        against the current value). Set new_num_rooms only for an explicit room
        TOTAL and rooms_delta for add/remove-a-room phrasing. Never calculate
        rooms from the number of people.
        """

        adjustment: TravelAdjustments = self.extraction_model.invoke(prompt)

        if not adjustment.is_adjustment:
            return {}

        updates = {}
        summary_parts = []

        if adjustment.new_destination:
            updates["destination_city"] = adjustment.new_destination
            summary_parts.append(f"Destination changed to {adjustment.new_destination}")

        if adjustment.new_origin:
            updates["current_city"] = adjustment.new_origin
            summary_parts.append(f"Origin changed to {adjustment.new_origin}")

        new_budget = resolve_budget(
            state.get("total_budget"),
            absolute=adjustment.new_budget,
            delta=adjustment.new_budget_delta,
            delta_pct=adjustment.new_budget_delta_pct,
        )
        if new_budget is not None:
            updates["total_budget"] = new_budget
            updates["budget_optional"] = False
            summary_parts.append(f"Budget changed to {new_budget}")

        if adjustment.new_trip_days is not None:
            updates["trip_days"] = adjustment.new_trip_days
            summary_parts.append(f"Trip days changed to {adjustment.new_trip_days}")

        # Travellers / rooms are isolated: they update the new group-size fields
        # but must NOT trigger the re-search side effects below (no summary
        # overwrite, no flight/plan resets). A room-only edit changes rooms and
        # nothing else; a people change also auto-recomputes rooms.
        traveler_updates = apply_traveler_updates(
            state,
            new_adults=adjustment.new_num_adults,
            new_children=adjustment.new_num_children,
            new_rooms_abs=adjustment.new_num_rooms,
            rooms_delta=adjustment.rooms_delta,
        )

        if not updates:
            # No trip-parameter change to re-search for; return any traveler
            # updates on their own so nothing else is disturbed.
            return traveler_updates

        # 1. Overwrite summary to force the agent to search again
        #
        # NOTE: We deliberately do NOT prune the message history here. Deleting
        # messages from state also erases them from any client that renders the
        # live graph state (e.g. agent-chat-ui), making the visible conversation
        # collapse mid-turn. The rolling `summary` below — plus the flight/plan
        # state resets — are what force the re-search; the transcript stays
        # intact. (Same policy the summary node documents.)
        update_reasons = ", ".join(summary_parts)

        # Only reset travel data when destination/origin/budget changed — those
        # require a new flight search. A trip_days-only change uses the same
        # route and hotel, so existing flight and travel data remain valid.
        needs_travel_reset = bool(
            adjustment.new_destination
            or adjustment.new_origin
            or new_budget is not None
        )

        if needs_travel_reset:
            updates["summary"] = f"USER ADJUSTMENTS MADE: {update_reasons}. SYSTEM MUST SEARCH FOR NEW FLIGHTS AND HOTELS."
            updates["flight_options"] = []
            updates["return_flight_options"] = []
            updates["has_flights"] = False
            updates["travel_plan"] = {}
            # Clear itinerary_mode so after_enrichment routes to flight_search, not
            # plan_check. Without this, users who already had an itinerary and then
            # change destination get silently re-routed to plan_check, bypassing the
            # flight search and the no-flights → alternatives path entirely.
            updates["itinerary_mode"] = None
        else:
            updates["summary"] = f"USER ADJUSTMENTS MADE: {update_reasons}."

        # 2. Force re-enrichment
        updates["enrichment_complete"] = False

        # 3. Mark that this was an adjustment, not a brand new request
        updates["is_adjustment"] = True
        updates["alternative_destinations"] = []
        updates["itinerary_plan"] = {}   # always clear — must rebuild with updated params

        # Fold in any traveller/room changes made in the same message.
        updates.update(traveler_updates)

        return updates
