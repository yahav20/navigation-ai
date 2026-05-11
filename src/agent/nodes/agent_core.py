"""Core agent node that drives tool calling and the assistant reply."""
# src/agent/nodes/agent_core.py
from langchain_core.runnables import Runnable

from agent.state import AgentState


class AgentNode:
    """Drive the main agent step that decides whether to call tools or reply."""

    def __init__(self, model_with_tools: Runnable) -> None:
        """Store the tool-bound chat model used to generate responses."""
        self.model = model_with_tools

    def __call__(self, state: AgentState) -> dict:
        """Examine current state and decide whether to trigger a tool or provide an answer."""
        current_step = state.get("step_count", 0) + 1
        summary = state.get("summary", "")

        clean_history = [
            msg for msg in state.get("messages", [])
            if getattr(msg, "type", "") != "formatter_output"
        ]

        origin = state.get("current_city") or "NOT PROVIDED"
        dest = state.get("destination_city") or "NOT PROVIDED"
        trip_days = state.get("trip_days") or 3

        if state.get("total_budget"):
            budget = state["total_budget"]
        elif state.get("budget_optional"):
            budget = "No budget constraint (user opted to skip)"
        else:
            budget = "NOT PROVIDED"

        system_prompt = f"""You are Atlas, a strict and professional luxury travel assistant.
        CONTEXT FROM PREVIOUS EXCHANGES:
        {summary or "No previous context. This is a new conversation."}

        CURRENT TRIP STATUS:
        - User is currently in: {origin}
        - User wants to travel to: {dest}
        - User's budget: {budget}
        - Trip duration: {trip_days} days

        CRITICAL INSTRUCTIONS & GUARDRAILS:
        1. MISSING INFO: If ANY of the 'CURRENT TRIP STATUS' fields are 'NOT PROVIDED', ask the user politely for the missing information. Do not search until you have all three.
        2. EXPLICIT TOOL EXECUTION: You MUST gather real data by calling ALL of the following tools in order:
           a. `fetch_flights` — get available flights from origin to destination.
           b. `fetch_hotels` — get available hotels at the destination.
           c. `calculate_trip_cost` — call with the cheapest available flight price, cheapest available hotel price per night, and {trip_days} days. This gives the true total cost.
           d. `fetch_activities` — get available activities at the destination.
           e. `get_average_weather` — call with the destination city and the current or upcoming season (Spring/Summer/Autumn/Winter).
           f. `get_best_time_to_visit` — call with the destination city.
        3. BUDGET VALIDATION: After calling `calculate_trip_cost`, compare the total_estimate against the user's budget ({budget}).
           - If total_estimate <= budget: proceed normally.
           - If total_estimate > budget: still present the cheapest option but clearly flag it exceeds budget.
        4. NO HALLUCINATIONS: Check your conversation history. Have you received results from ALL six tools above? If NO, call the missing ones now.
        5. BOUNDARY ENFORCEMENT: Decline any non-travel questions and steer back to the trip.

        DO NOT output the final itinerary yourself. Just confirm you found the data. The system will handle formatting.
        DO NOT ask the user any questions.
        """

        messages_to_pass = [{"role": "system", "content": system_prompt}, *clean_history]

        response = self.model.invoke(messages_to_pass)

        return {
            "messages": [response],
            "step_count": current_step,
        }
