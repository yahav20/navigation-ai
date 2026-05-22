"""Router node to classify user intent and direct the graph flow."""
from langchain_core.language_models import BaseChatModel
from agent.state import AgentState
from agent.models import IntentClassification

class RouterNode:
    """Classify user intent to route to the appropriate sub-graph."""

    def __init__(self, classification_model: BaseChatModel) -> None:
        self.classification_model = classification_model.with_structured_output(IntentClassification)

    def __call__(self, state: AgentState) -> dict:
        messages = state.get("messages", [])
        if not messages:
            return {"intent": "other"}

        last_msg = messages[-1]
        
        # If it's a system or tool message, do not route again
        if getattr(last_msg, "type", "") != "human":
            return {}

        # Provide existing context to the router so it knows if we are mid-planning
        has_active_trip = bool(state.get("destination_city") or state.get("current_city"))
        last_intent = state.get("intent", "")
        is_rec_flow = last_intent == "recommendations" and not has_active_trip

        if has_active_trip:
            trip_status = "ACTIVE TRIP IN PROGRESS"
        elif is_rec_flow:
            trip_status = "ACTIVE RECOMMENDATION FLOW (user is browsing destination options — budget/day changes refine the search)"
        else:
            trip_status = "NO ACTIVE TRIP (Start from scratch)"

        prompt = f"""
        Analyze the user's latest message and classify their core intent.

        Current System Context: {trip_status}
        - Origin: {state.get("current_city", "None")}
        - Destination: {state.get("destination_city", "None")}
        - Budget: {state.get("total_budget", "None")}

        Examples:
        - "I live in Paris and want to fly somewhere." -> recommendations
        - "Book me a trip to Rome for 3 days" -> new_travel_plan
        - "Plan a day-by-day schedule for my 3 days in Rome" -> build_itinerary
        - "Change my budget to $500." -> update_travel_plan
        - "Do I need a visa for Japan?" -> general_chat

        User message: "{last_msg.content}"
        """

        classification: IntentClassification = self.classification_model.invoke(prompt)

        final_intent = classification.intent

        # Deterministic Guardrail: Can't start a new direct plan without a destination
        if final_intent == "new_travel_plan" and not classification.has_explicit_destination:
            final_intent = "recommendations"

        # Guardrail: If user tries to update but no active trip exists, convert to new plan
        # Exception: if we were in a recommendations flow, keep it as recommendations
        if final_intent == "update_travel_plan" and not has_active_trip:
            if state.get("intent") == "recommendations":
                final_intent = "recommendations"
            else:
                final_intent = "new_travel_plan"

        return {"intent": final_intent}
