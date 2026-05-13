# src/agent/nodes/agent_core.py
from langchain_core.runnables import Runnable
from agent.state import AgentState

class AgentNode:
    def __init__(self, model_with_tools: Runnable) -> None:
        self.model = model_with_tools

    def __call__(self, state: AgentState) -> dict:
        current_step = state.get("step_count", 0) + 1
        messages = state.get("messages", [])
        summary = state.get("summary", [])
        
        clean_history = [m for m in messages if getattr(m, "type", "") != "formatter_output"]

        origin = state.get("current_city") or "NOT PROVIDED"
        dest = state.get("destination_city") or "NOT PROVIDED"
        trip_days = state.get("trip_days") or 3
        budget = state.get("total_budget") or "NOT PROVIDED"
        user_prefs = state.get("user_preferences") or {}
        prefs_string = "\n".join([f"- {k}: {v}" for k, v in user_prefs.items()]) if user_prefs else "None specified."
        print(f"\n[DEBUG] Current State Preferences: {user_prefs}\n")
        executed_tools = [getattr(m, "name", "") for m in messages if getattr(m, "type", "") == "tool"]

        if "fetch_flights" not in executed_tools:
            action = f"CRITICAL ACTION: You MUST call `fetch_flights` with origin='{origin}' and destination='{dest}'. Do nothing else."
        elif "fetch_hotels" not in executed_tools:
            action = f"CRITICAL ACTION: You MUST call `fetch_hotels` with city='{dest}'. Do nothing else."
        elif "calculate_trip_cost" not in executed_tools:
            action = "CRITICAL ACTION: You MUST call `calculate_trip_cost` using the flight and hotel prices you fetched."
        else:
            action = "You have all the data. Just say 'I have finished gathering the travel data.' DO NOT call any tools."

        system_prompt = f"""You are Atlas, a strict robotic travel agent.
        
        CONTEXT FROM PREVIOUS EXCHANGES:
        {summary or "No previous context. This is a new conversation."}

        TRIP DETAILS:
        - Origin: {origin}
        - Destination: {dest}
        - Budget: {budget}
        - Duration: {trip_days} days

        USER PREFERENCES (CRITICAL: You MUST respect these when passing arguments to tools):
        {prefs_string}

        CRITICAL INSTRUCTIONS & GUARDRAILS:
        1. MISSING INFO: If ANY of the 'CURRENT TRIP STATUS' fields are 'NOT PROVIDED', ask...
        2. EXPLICIT TOOL EXECUTION: You MUST gather real data by calling ALL of the following tools:
           a. `fetch_flights` (Consider airline preferences if provided).
           b. `fetch_hotels`
           c. `calculate_trip_cost`
           d. `fetch_activities` (Must align with dietary/lifestyle preferences like Vegan, Accessible, etc.).
           e. `get_average_weather`
           f. `get_best_time_to_visit`
        3. NO HALLUCINATIONS: Check your conversation history. Have you received results from ALL six tools above? If NO, call the missing ones now.
        5. BOUNDARY ENFORCEMENT: Decline any non-travel questions and steer back to the trip.

        DO NOT output the final itinerary yourself. Just confirm you found the data. The system will handle formatting.
        DO NOT ask the user any questions.
        """

        messages_to_pass = [{"role": "system", "content": system_prompt}, *clean_history]
        response = self.model.invoke(messages_to_pass)

        return {"messages": [response], "step_count": current_step}