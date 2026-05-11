# src/agent/nodes/agent_core.py
from langchain_core.runnables import Runnable

from agent.state import AgentState


class AgentNode:
    """Drive the main agent step that decides whether to call tools or reply."""

    def __init__(self, model_with_tools: Runnable) -> None:
        self.model = model_with_tools

    def __call__(self, state: AgentState) -> dict:
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
            budget = "No budget constraint"
        else:
            budget = "NOT PROVIDED"

        system_prompt = f"""You are Atlas, a professional luxury travel assistant.
        CONTEXT FROM PREVIOUS EXCHANGES:
        {summary or "No previous context."}

        CURRENT TRIP STATUS:
        - Origin: {origin}
        - Destination: {dest}
        - Budget: {budget}
        - Duration: {trip_days} days

        CRITICAL INSTRUCTIONS (EXECUTE IN EXACT ORDER):
        1. If Origin, Destination, or Budget are 'NOT PROVIDED', ask the user for them.
        2. You MUST call `fetch_flights` to get flight options.
        3. You MUST call `fetch_hotels` to get hotel options.
        4. ONLY AFTER you have real flight prices and hotel prices, call `calculate_trip_cost`. 
        5. NEVER invent prices. NEVER send a flight price of 0.0 to the calculator.

        Once you have fetched flights, hotels, and cost, simply output a short confirmation message.
        """

        messages_to_pass = [{"role": "system", "content": system_prompt}, *clean_history]

        response = self.model.invoke(messages_to_pass)

        return {
            "messages": [response],
            "step_count": current_step,
        }