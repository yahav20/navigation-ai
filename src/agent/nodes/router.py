"""Router node to classify user intent and direct the graph flow."""
from langchain_core.language_models import BaseChatModel

from agent.models import IntentClassification
from agent.state import AgentState


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

        has_existing_trip_context = state.get("has_existing_trip_context", False)
        last_intent = state.get("intent", "")
        is_rec_flow = last_intent == "recommendations" and not has_existing_trip_context

        if has_existing_trip_context:
            trip_status = "ACTIVE TRIP IN PROGRESS"
        elif is_rec_flow:
            trip_status = "ACTIVE RECOMMENDATION FLOW (user is browsing destination options)"
        else:
            trip_status = "NO ACTIVE TRIP (Start from scratch)"

        # Include whether flights are already fetched — helps the router
        # correctly classify follow-up itinerary requests
        has_flight_data = bool(state.get("flight_options"))
        has_destination  = bool(state.get("destination_city"))

        prompt = f"""
Analyze the user's latest message and classify their core intent.

Current System Context: {trip_status}
- Origin:      {state.get("current_city", "None")}
- Destination: {state.get("destination_city", "None")}
- Budget:      {state.get("total_budget", "None")}
- Flights already fetched: {has_flight_data}
- Days planned: {state.get("trip_days", "None")}

Classification rules:
- "I live in Paris and want to fly somewhere."            → recommendations
- "Book me a trip to Rome for 3 days"                    → new_travel_plan
- "Plan a day-by-day schedule for my 3 days in Rome"     → itinerary
- "Ok now please plan a full trip" (after flights shown) → itinerary
- "Plan the full itinerary" (after alternatives shown)   → itinerary
- "Change my budget to $500."                            → update_travel_plan
- "Do I need a visa for Japan?"                          → general_chat

User message: "{last_msg.content}"
"""

        classification: IntentClassification = self.classification_model.invoke(prompt)
        final_intent = classification.intent

        # ── Guardrail 1: can't do new_travel_plan without a destination ──
        if final_intent == "new_travel_plan" and not classification.has_explicit_destination:
            final_intent = "recommendations"

        # ── Guardrail 2: update without active trip → new plan ──
        if final_intent == "update_travel_plan" and not has_existing_trip_context:
            if state.get("intent") == "recommendations":
                final_intent = "recommendations"
            else:
                final_intent = "new_travel_plan"

        # ── Set build_itinerary flag ──────────────────────────────────────
        # This flag is read by after_flight_search and after_travel_agent
        # to decide whether to branch into the itinerary sub-graph.
        build_itinerary = final_intent in ("itinerary", "build_itinerary")

        return {
            "intent": final_intent,
            "build_itinerary": build_itinerary,
        }