"""Detect and process user adjustments to travel parameters."""
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import RemoveMessage

from agent.models import TravelAdjustments
from agent.state import AgentState


class AdjustmentsNode:
    """Check if the user wants to change destination, origin, or budget during the conversation."""

    def __init__(self, extraction_model: BaseChatModel) -> None:
        """Bind the Adjustments extraction model."""
        self.extraction_model = extraction_model.with_structured_output(TravelAdjustments)

    def __call__(self, state: AgentState) -> dict:
        """Return state updates for explicit trip-parameter changes."""
        messages = state.get("messages", [])
        if not messages:
            return {}

        last_msg = messages[-1]
        if getattr(last_msg, "type", "") != "human":
            return {}

        # We only look for adjustments if we already have some base parameters
        if not state.get("destination_city") and not state.get("current_city"):
            return {}

        prompt = f"""
        Current Trip State:
        - Origin: {state.get("current_city", "None")}
        - Destination: {state.get("destination_city", "None")}
        - Budget: {state.get("total_budget", "None")}
        - Trip Days: {state.get("trip_days", "None")}

        User's latest message: "{last_msg.content}"

        Is the user explicitly asking to adjust any of these parameters?
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

        if adjustment.new_budget is not None:
            updates["total_budget"] = adjustment.new_budget
            updates["budget_optional"] = False
            summary_parts.append(f"Budget changed to {adjustment.new_budget}")

        if adjustment.new_trip_days is not None:
            updates["trip_days"] = adjustment.new_trip_days
            summary_parts.append(f"Trip days changed to {adjustment.new_trip_days}")

        if not updates:
            return {}

        # 1. Clear old messages except the latest user message
        messages_to_delete = [
            RemoveMessage(id=m.id)
            for m in messages[:-1]
            if m.id is not None
        ]
        updates["messages"] = messages_to_delete

        # 2. Overwrite summary to force the agent to search again
        update_reasons = ", ".join(summary_parts)
        updates["summary"] = f"USER ADJUSTMENTS MADE: {update_reasons}. SYSTEM MUST SEARCH FOR NEW FLIGHTS AND HOTELS."

        # 3. Force re-enrichment
        updates["enrichment_complete"] = False

        # 4. Mark that this was an adjustment, not a brand new request
        updates["is_adjustment"] = True

        # Clear alternative destinations since parameters changed
        updates["alternative_destinations"] = []
        updates["flight_options"] = []
        updates["has_flights"] = False
        updates["travel_plan"] = {}
        updates["itinerary_plan"] = {}

        return updates
