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
        - Days: {state.get("trip_days", "None")}

        INTENT DEFINITIONS & DIFFERENCES:
        1. 'advisor': User wants destination ideas or travel info but does NOT have a specific destination yet, OR is asking about activities, weather, or food for a city. ("Where should I go for a beach trip?", "What's there to do in Tokyo?")
        2. 'new_travel_plan' (Info & Feasibility): User explicitly names a destination and wants to check flights, hotels, or costs, BUT they DO NOT ask for a daily schedule. ("I want to visit Rome, what are the flights?", "Let's check options for Madrid")
        3. 'build_itinerary' (Full Execution): User EXPLICITLY asks for a detailed, day-by-day itinerary, schedule, OR asks to "replan" the entire trip (even if they are changing destination or budget at the same time). Examples: "Build a 3-day itinerary for Rome", "replan for Paris with $700", "create full plan".
        4. 'update_travel_plan': User wants to change parameters (destination, budget, origin) of a trip, BUT DOES NOT ask for a daily schedule or a full replan. They just want to check feasibility/flights. Examples: "Change destination to Paris", "Increase budget to $1000".
        5. 'general_chat': Greetings, casual chat, or general travel questions not related to active planning.

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

        # Guardrail 1: Both high-level planning and micro-planning require a destination.
        # If the user asks for a plan or itinerary but we don't have a destination, route to advisor.
        if final_intent in ["new_travel_plan", "build_itinerary"]:
            has_dest = classification.has_explicit_destination or state.get("destination_city")
            if not has_dest:
                final_intent = "advisor"

        # Guardrail 2: If user tries to update but no active trip exists, convert to new plan
        if final_intent == "update_travel_plan" and not has_active_trip:
            if state.get("intent") == "advisor":
                final_intent = "advisor"
            else:
                final_intent = "new_travel_plan"

        # Guardrail 4: Override update to build_itinerary if planning is explicitly requested
        if final_intent == "update_travel_plan":
            content_lower = last_msg.content.lower()
            # If the user used planning/replanning trigger words, force route to Planner
            trigger_words = ["replan", "full plan", "schedule", "לוז", "תכנון מלא", "תבנה לי"]
            if any(word in content_lower for word in trigger_words):
                final_intent = "build_itinerary"

        return {"intent": final_intent}