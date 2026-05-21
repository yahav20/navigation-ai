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
        is_advisor_flow = last_intent == "advisor" and not has_active_trip

        if has_active_trip:
            trip_status = "ACTIVE TRIP IN PROGRESS"
        elif is_advisor_flow:
            trip_status = "ACTIVE ADVISOR FLOW (user is browsing destination options — budget/day changes refine the search)"
        else:
            trip_status = "NO ACTIVE TRIP (Start from scratch)"

        prompt = f"""
        Analyze the user's latest message and classify their core intent.

        Current System Context: {trip_status}
        - Origin: {state.get("current_city", "None")}
        - Destination: {state.get("destination_city", "None")}
        - Budget: {state.get("total_budget", "None")}

        TRANSITION RULE (critical): If the system is in ACTIVE ADVISOR FLOW and the user
        says something like "plan this trip", "let's go", "book this", "sounds good let's do it",
        "I want to go there", or any phrase that signals they are ready to commit to a trip —
        classify as 'new_travel_plan', NOT 'advisor'. The user is transitioning from
        browsing destinations to starting actual trip planning.

        User message: "{last_msg.content}"
        """

        classification: IntentClassification = self.classification_model.invoke(prompt)

        # Guardrail: If user tries to update but no active trip exists, convert to new plan
        # Exception: if we were in an advisor flow, keep it as advisor
        final_intent = classification.intent
        if final_intent == "update_travel_plan" and not has_active_trip:
            if state.get("intent") == "advisor":
                final_intent = "advisor"
            else:
                final_intent = "new_travel_plan"

        return {"intent": final_intent}
