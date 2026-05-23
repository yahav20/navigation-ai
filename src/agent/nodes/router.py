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
        if getattr(last_msg, "type", "") != "human":
            return {}

        has_existing_trip  = state.get("has_existing_trip_context", False)
        has_itinerary      = bool(state.get("itinerary_plan"))
        has_flights        = bool(state.get("flight_options"))
        last_intent        = state.get("intent", "")
        is_rec_flow        = last_intent == "recommendations" and not has_existing_trip

        if has_itinerary:
            context_line = "ACTIVE ITINERARY — user may want to update preferences or re-plan"
        elif has_existing_trip:
            context_line = "ACTIVE TRIP IN PROGRESS"
        elif is_rec_flow:
            context_line = "ACTIVE RECOMMENDATION FLOW"
        else:
            context_line = "NO ACTIVE TRIP"

        prompt = f"""Classify the user's intent into exactly one category.

Context: {context_line}
Origin: {state.get("current_city", "None")}
Destination: {state.get("destination_city", "None")}
Budget: {state.get("total_budget", "None")}
Days: {state.get("trip_days", "None")}
Itinerary already built: {has_itinerary}
Flights already fetched: {has_flights}

CATEGORIES:
- new_travel_plan      : user wants to plan a brand-new trip to a specific destination
- update_travel_plan   : user wants to change trip parameters (dates, budget, origin/destination)
- itinerary            : user wants a day-by-day schedule (first time, or fresh destination)
- update_itinerary     : user wants to UPDATE or ADJUST an EXISTING itinerary
                         (e.g. "I'm vegetarian", "add more museums", "I prefer 4-star hotels")
                         Only use this when an itinerary is already built.
- recommendations      : user wants destination suggestions
- general_chat         : questions, visa info, weather, small talk

EXAMPLES:
"Book me a trip to Rome for 3 days"                          → new_travel_plan
"Plan a day-by-day schedule for my 3 days in Paris"         → itinerary
"Ok now plan a full trip"  (flights already shown)          → itinerary
"Plan a 3-day itinerary for Paris from Tel Aviv, budget is 1k"  → itinerary 
"I'm actually vegetarian, please update my itinerary"       → update_itinerary
"Add more outdoor activities to my plan"                    → update_itinerary
"Change my budget to $1000"                                 → update_travel_plan
"I want to fly somewhere warm"                              → recommendations
"Do I need a visa for Japan?"                               → general_chat

User message: "{last_msg.content}"
"""

        classification: IntentClassification = self.classification_model.invoke(prompt)
        final_intent = classification.intent

        # Guardrail: new_travel_plan without destination → recommendations
        if final_intent == "new_travel_plan" and not classification.has_explicit_destination:
            final_intent = "recommendations"

        # Guardrail: update_travel_plan without active trip → new plan
        if final_intent == "update_travel_plan" and not has_existing_trip:
            final_intent = "recommendations" if is_rec_flow else "new_travel_plan"

        # Guardrail: update_itinerary but no itinerary built yet → treat as itinerary
        if final_intent == "update_itinerary" and not has_itinerary:
            final_intent = "itinerary"

        build_itinerary = final_intent in ("itinerary", "build_itinerary", "update_itinerary")

        return {
            "intent": final_intent,
            "build_itinerary": build_itinerary,
        }