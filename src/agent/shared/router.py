"""Router node to classify user intent and direct the graph flow."""
from langchain_core.language_models import BaseChatModel
from agent.core.llm import silent
from agent.core.state import AgentState
from agent.core.models import IntentClassification

class RouterNode:
    """Classify user intent to route to the appropriate sub-graph."""

    def __init__(self, classification_model: BaseChatModel) -> None:
        self.classification_model = silent(classification_model.with_structured_output(IntentClassification))

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

        INTENT DEFINITIONS & DIFFERENCES (when in doubt, use 'advisor'):
        1. 'advisor': The DEFAULT for all travel-related queries. Use for: destination ideas,
           comparing destinations, activities in a city, city overviews, budget exploration,
           travel recommendations, currency exchange ("how much is $500 in euros?"),
           visa requirements ("do I need a visa for Japan?"), travel safety ("is Rio safe?"),
           packing questions ("what should I pack for Tokyo?"), local customs and etiquette,
           greetings ("hello", "hi", "thanks"), and capability questions ("what can you do?").
        2. 'new_travel_plan': User commits to a SPECIFIC destination and wants flights/hotels/costs.
           ("I want to fly to Rome — show me flights", "Let's book a trip to Madrid").
           ONLY use when the user is ready to START PLANNING a specific trip, not just exploring.
        3. 'build_itinerary': ONLY when user EXPLICITLY asks for a day-by-day schedule.
           ("Build a 3-day itinerary for Rome", "plan my days in Paris", "replan for $700").
           "How should I split my time?" or "how many days per city?" → 'advisor', NOT here.
        4. 'update_travel_plan': Changing an existing confirmed plan's core parameters (budget, dates, destination, origin, or number of trip days/duration).
        5. 'update_itinerary': Editing the day-by-day schedule of an ALREADY-BUILT itinerary — swapping or
           removing activities/restaurants, adding a specific place, changing meal preferences
           ("I don't want the Eiffel Tower", "add PSG Museum", "remove breakfast from all days").
           Only use when an itinerary has already been built.
        6. 'out_of_scope': ONLY for queries with zero travel relevance (coding, math, recipes, sports).

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

        # Guardrail 0: Preserve build_itinerary intent across enrichment turns.
        if (state.get("intent") == "build_itinerary"
                and not state.get("enrichment_complete", False)):
            final_intent = "build_itinerary"

        # Guardrail 1: Both high-level planning and micro-planning require a destination.
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
            trigger_words = ["replan", "full plan", "schedule", "schedule", "plan", "  itinerary", "day-by-day", "day by day", "build my trip", "build this trip", "plan this trip", "sounds good let's do it", "let's go", "I want to go there"]
            if any(word in content_lower for word in trigger_words):
                final_intent = "build_itinerary"

        # Guardrail 4b: trip duration changes must go to update_travel_plan, not update_itinerary
        if final_intent == "update_itinerary":
            content_lower = last_msg.content.lower()
            day_change_phrases = [
                "add a day", "add one day", "one more day", "more day",
                "another day", "add day", "extra day",
                "fewer day", "remove a day", "one less day", "less day",
                "extend the trip", "shorten the trip", "longer trip", "shorter trip",
                "increase days", "decrease days", "more days", "less days",
            ]
            if any(phrase in content_lower for phrase in day_change_phrases):
                final_intent = "update_travel_plan"

        # Guardrail 5: out_of_scope cannot be overridden by trip context
        if final_intent == "out_of_scope":
            return {"intent": "out_of_scope"}

        # Guardrail 6: update_itinerary requires a built itinerary; otherwise build a fresh one
        if final_intent == "update_itinerary":
            has_itinerary = bool(
                isinstance(state.get("itinerary_plan"), dict)
                and state["itinerary_plan"].get("step_results")
            )
            if not has_itinerary:
                final_intent = "build_itinerary"

        return {"intent": final_intent}
