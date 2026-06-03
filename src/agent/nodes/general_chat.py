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
- Asking general travel tips (packing, visas, currency, etiquette, safety)
- Asking about a specific city's weather, activities, or culture
- Asking when to visit a destination

KNOWN CONTEXT (use if relevant, don't force it):
{state_context}

PREVIOUS CONVERSATION:
{summary}

YOUR TOOLS — use only when the user asks about a specific city:
- get_city_overview(city): culture, activity types, best months, temperatures
- fetch_activities(city): specific things to do in a city
- get_average_weather(city, season): weather by season
- get_best_time_to_visit(city): best time of year to visit

RULES:
1. Greetings / "what can you do?" → respond warmly in plain prose, NO tool calls
2. General travel tips (packing, visas, etiquette) → answer from knowledge, NO tool calls
3. City-specific questions → use at most 1 tool
4. Keep answers concise — 3-5 sentences or a short bullet list
5. If the user seems ready to plan a trip, gently suggest:
   "Whenever you're ready, just tell me your destination and I'll build a full itinerary!"
6. NEVER make up flight prices, hotel costs, or availability data
7. Respond in natural conversational prose — NOT in "DATA COLLECTED / READY FOR FORMATTING" format
"""

CHAT_MAX_STEPS = 2


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
