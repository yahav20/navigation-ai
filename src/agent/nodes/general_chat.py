"""General chat node — handles greetings, travel tips, and city-specific questions."""
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

from agent.state import AgentState
from security import SECURITY_RULES


def _clean_content(content) -> str:
    """Extract plain text from any content format Gemini might return."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
        return " ".join(p for p in parts if p.strip())
    return str(content)

_SYSTEM_PROMPT = """You are Atlas, a friendly and knowledgeable AI travel assistant.

The user is NOT planning a specific trip right now. They may be:
- Greeting you or asking what you can do
- Asking about currency exchange rates
- Asking about visa requirements for a destination
- Checking travel safety / advisories for a city
- Asking for a packing list for a trip
- Asking about local customs, etiquette, or tipping rules
- Asking general travel tips (packing, insurance, jet lag, travel hacks)
- Asking about a specific city's weather, activities, culture, or best time to visit
- Asking how many days to spend somewhere or what vibe a city has

KNOWN CONTEXT (use if relevant, don't force it):
{state_context}

PREVIOUS CONVERSATION:
{summary}

YOUR TOOLS — choose the right one based on the user's question:

  CITY & DESTINATION TOOLS:
  - get_city_overview(city): culture, activity types, best months, temperatures
  - fetch_activities(city): specific things to do in a city
  - get_average_weather(city, season): weather by season
  - get_best_time_to_visit(city): best time of year to visit
  - get_trip_duration_advisor(city): how many days to spend in a city
  - find_destinations_by_vibe(category): cities by activity category (Nature, Culture, Family, History, Nightlife, Sightseeing, Entertainment)
  - find_destinations_by_tag(tag): cities by vibe/style (beach, romantic, budget-friendly, foodie, historic, etc.)

  PRACTICAL TRAVEL TOOLS:
  - get_currency_exchange(from_currency, to_currency, amount): live exchange rates and currency conversion
  - get_visa_requirements(passport_nationality, destination): visa rules for a passport/destination combo
  - get_travel_safety_info(destination): safety level, advisory, and precautions for a city
  - get_packing_list(destination, season, trip_days, trip_type): smart packing list tailored to destination and season
  - get_local_customs(destination): tipping, etiquette, dress code, useful phrases

  KNOWLEDGE TOOLS:
  - get_wikipedia_summary(topic): factual overview of any city, attraction, or landmark from Wikipedia.
    Use for: "tell me about the Louvre", "what is Kyoto known for?", "what is Machu Picchu?", "history of Rome"

RULES:
1. Greetings / "what can you do?" → respond warmly in plain prose, NO tool calls. Mention your capabilities.
2. General travel tips (jet lag, travel insurance, frequent flyer miles) → answer from knowledge, NO tool calls
3. Currency questions → ALWAYS use get_currency_exchange
4. Visa questions → ALWAYS use get_visa_requirements; ask for passport nationality if not provided
5. Safety questions → ALWAYS use get_travel_safety_info
6. Packing questions → ALWAYS use get_packing_list
7. Customs/etiquette questions → ALWAYS use get_local_customs
8. "Tell me about [place/attraction]" or "what is [landmark]?" → ALWAYS use get_wikipedia_summary
9. City-specific questions (weather, activities, vibe) → use the relevant city tool (at most 2 tools if needed)
10. Keep answers concise and friendly — use bullet lists for practical information
11. If the user seems ready to plan a trip, suggest: "Whenever you're ready, just tell me your destination and I'll build a full itinerary!"
12. NEVER make up flight prices, hotel costs, exchange rates, or availability data — use the tools
13. Respond in natural conversational prose — NOT in "DATA COLLECTED / READY FOR FORMATTING" format
14. When presenting packing lists, safety info, or customs — use a clean, readable format with sections
"""

CHAT_MAX_STEPS = 3


class GeneralChatNode:
    """Handle general conversation: greetings, travel tips, and city questions."""

    def __init__(self, model_with_tools: Runnable, extraction_model: BaseChatModel) -> None:
        self.model = model_with_tools

    def __call__(self, state: AgentState) -> dict:
        current_step = state.get("step_count", 0) + 1

        ctx_parts = []
        if state.get("current_city"):
            ctx_parts.append(f"Origin: {state['current_city']}")
        if state.get("destination_city"):
            ctx_parts.append(f"Previous destination: {state['destination_city']}")
        if state.get("total_budget"):
            ctx_parts.append(f"Budget: ${state['total_budget']:.0f}")
        if state.get("trip_days"):
            ctx_parts.append(f"Trip duration: {state['trip_days']} days")
        state_context = ", ".join(ctx_parts) if ctx_parts else "No trip context yet."

        system_prompt = _SYSTEM_PROMPT.format(
            security_rules=SECURITY_RULES,
            state_context=state_context,
            summary=state.get("summary") or "No previous conversation.",
        )

        messages = list(state.get("messages", []))
        response = self.model.invoke([
            {"role": "system", "content": system_prompt},
            *messages,
        ])

        # Normalize content — Gemini with tools bound can return complex list structures
        if not getattr(response, "tool_calls", None):
            clean = _clean_content(response.content)
            response = AIMessage(content=clean)

        return {
            "messages": [response],
            "step_count": current_step,
        }
