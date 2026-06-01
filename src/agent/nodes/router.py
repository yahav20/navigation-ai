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

        INTENT DEFINITIONS & DIFFERENCES (when in doubt, pick 'general_chat' for informational questions or 'advisor' for destination/planning questions):
        1. 'advisor': For destination exploration and planning research. Use for: "where should I go?",
           comparing destinations, activities in a city, city overviews, budget exploration,
           travel recommendations when the user is deciding WHERE to go.
           Do NOT use for currency, visa, safety, packing, or customs questions — those go to 'general_chat'.
        2. 'new_travel_plan': User commits to a SPECIFIC destination and wants flights/hotels/costs.
           ("I want to fly to Rome — show me flights", "Let's book a trip to Madrid").
           ONLY use when the user is ready to START PLANNING a specific trip, not just exploring.
        3. 'build_itinerary': ONLY when user EXPLICITLY asks for a day-by-day schedule.
           ("Build a 3-day itinerary for Rome", "plan my days in Paris", "replan for $700").
           "How should I split my time?" or "how many days per city?" → 'advisor', NOT here.
        4. 'update_travel_plan': Changing an existing confirmed plan's parameters (budget, dates, destination).
        5. 'general_chat': Conversational queries that do NOT require trip planning. Use for:
           - Greetings, thanks, "what can you do?"
           - Currency / exchange rate questions ("how much is $500 in euros?")
           - Visa requirement questions ("do I need a visa for Japan?")
           - Travel safety questions ("is Rio safe to visit?")
           - Packing list questions ("what should I pack for Tokyo?")
           - Local customs / etiquette / tipping questions ("what are tipping rules in Japan?")
           - Destination vibe / "where should I go for a romantic trip?"
           - How many days to spend in a city ("how many days in Amsterdam?")
           Do NOT use for questions that need flights, hotels, or day-by-day itinerary building.

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

        # Guardrail 5: Hard-redirect general-chat question types that the LLM
        # tends to misclassify as 'advisor' (currency, visa, safety, packing, customs).
        if final_intent == "advisor" and not has_active_trip:
            content_lower = last_msg.content.lower()
            _GENERAL_CHAT_SIGNALS = [
                "exchange rate", "currency", "convert", "how much is",
                "visa", "passport", "do i need a visa",
                "safe to visit", "is it safe", "safety", "travel advisory",
                "what to pack", "packing list", "what should i pack",
                "customs", "tipping", "etiquette", "local customs",
                "how many days", "how long to spend", "days in",
                "romantic", "budget-friendly", "best city for",
                "recommend a city", "where should i go for",
            ]
            if any(sig in content_lower for sig in _GENERAL_CHAT_SIGNALS):
                final_intent = "general_chat"

        return {"intent": final_intent}