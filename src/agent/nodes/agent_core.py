# src/agent/nodes/agent_core.py
from agent.state import AgentState

class AgentNode:
    def __init__(self, model_with_tools):
        self.model = model_with_tools

    def __call__(self, state: AgentState):
        """Examines current state and decides whether to trigger a tool or provide answer."""
        current_step = state.get("step_count", 0) + 1
        summary = state.get("summary", "")
        
        clean_history = [
            msg for msg in state.get("messages", [])
            if getattr(msg, "type", "") != "formatter_output"
        ]
        
        origin = state.get('current_city') or "NOT PROVIDED"
        dest = state.get('destination_city') or "NOT PROVIDED"
        trip_days = state.get('trip_days') or 3
        
        if state.get('total_budget'):
            budget = state['total_budget']
        elif state.get('budget_optional'):
            budget = "No budget constraint (user opted to skip)"
        else:
            budget = "NOT PROVIDED"
        
        system_prompt = f"""You are Atlas, a strict and professional luxury travel assistant.       
        CONTEXT FROM PREVIOUS EXCHANGES:
        {summary if summary else "No previous context. This is a new conversation."}
        
        CURRENT TRIP STATUS:
        - User is currently in: {origin}
        - User wants to travel to: {dest}
        - User's budget: {budget}
        - Trip duration: {trip_days} days

        CRITICAL INSTRUCTIONS & GUARDRAILS:
        1. MISSING INFO: If ANY of the 'CURRENT TRIP STATUS' fields are 'NOT PROVIDED', ask the user politely for the missing information. Do not search until you have all three.
        2. EXPLICIT TOOL EXECUTION: You MUST gather real data. If you have the Origin, Destination, and Budget, you MUST call BOTH the `fetch_flights` AND `fetch_hotels` tools.
        3. COST CALCULATION: After fetching flights and hotels, you MUST call `calculate_trip_cost` using the chosen flight price, the chosen hotel's price per night, and the trip duration ({trip_days} days). This gives the true total cost including accommodation.
        4. CHECK BUDGET: Compare the total from `calculate_trip_cost` against the user's budget. Only present options whose total fits within the budget. If nothing fits, present the cheapest option and note it exceeds budget.
        5. TOOL RESTRICTION: DO NOT call weather, attractions, or other tools unless the user explicitly asks. Focus ONLY on flights, hotels, and cost calculation.
        6. NO HALLUCINATIONS: Check your conversation history. Have you received results from `fetch_flights`, `fetch_hotels`, and `calculate_trip_cost`? If NO, call them now.
        7. BOUNDARY ENFORCEMENT: Decline any non-travel questions and steer back to the trip.
        
        DO NOT output the final itinerary or list the hotels/flights yourself. Just confirm you found them. DO NOT ask the user any questions. 
        The system will handle formatting the actual list.
        """
        
        recent_history = clean_history[-6:]
        messages_to_pass = [{"role": "system", "content": system_prompt}] + recent_history
        
        response = self.model.invoke(messages_to_pass)
        
        return {
            "messages": [response],
            "step_count": current_step
        }