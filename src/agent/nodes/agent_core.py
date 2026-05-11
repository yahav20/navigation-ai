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
            2. EXPLICIT TOOL EXECUTION: If you have the Origin, Destination, and Budget, you MUST call the `fetch_flights` AND `fetch_hotels` tools FIRST. 
            3. SEQUENTIAL COST CALCULATION: DO NOT call `calculate_trip_cost` at the same time as fetching flights and hotels. You MUST wait to receive the results from flights and hotels FIRST. Only after you have the actual prices from those tool responses, call `calculate_trip_cost` using a chosen flight price, a chosen hotel's price per night, and the trip duration ({trip_days} days).
            4. CHECK BUDGET: Compare the total from `calculate_trip_cost` against the user's budget. Only present options whose total fits within the budget. If nothing fits, present the cheapest option and note it exceeds budget.
            5. TOOL RESTRICTION: DO NOT call weather, attractions, or other tools unless the user explicitly asks. Focus ONLY on flights, hotels, and cost calculation.
            6. NO HALLUCINATIONS: Check your conversation history. Have you received results from `fetch_flights`, `fetch_hotels`, and `calculate_trip_cost`? If NO, call them now.
            7. BOUNDARY ENFORCEMENT: Decline any non-travel questions and steer back to the trip.
            DO NOT output the final itinerary or list the hotels/flights yourself. Just confirm you found them. DO NOT ask the user any questions.
            The system will handle formatting the actual list.
        """

        messages_to_pass = [{"role": "system", "content": system_prompt}, *clean_history]

        response = self.model.invoke(messages_to_pass)

        return {
            "messages": [response],
            "step_count": current_step,
        }
