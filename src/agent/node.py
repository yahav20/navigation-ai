import json
from typing import Optional
from pydantic import BaseModel, Field
from agent.state import AgentState
from agent.nodes.enrichment import EnrichmentNode
from tools.tools import tools
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Pydantic type definitions for structured output extraction
class TravelMetadata(BaseModel):
    current_city: Optional[str] = Field(default=None, description="The city the user is currently in / starting from")
    destination_city: Optional[str] = Field(default=None, description="The city the user wants to travel to")
    budget: Optional[float] = Field(default=None, description="The user's travel budget as a number, if mentioned")
    trip_days: Optional[int] = Field(default=None, description="Number of days the user wants to spend on the trip, if mentioned")

# --- Factory function for model selection ---
def get_models(provider: str = "google"):
    """
    Returns a tuple of (model_with_tools, extraction_model) based on the chosen provider.
    """
    if provider.lower() == "groq":
        print(" Initializing Groq (Llama 3)...")
        model = ChatGroq(model="llama-3.1-8b-instant", temperature=0).bind_tools(tools)
        extraction_model = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    else:
        print(" Initializing Google (Gemini 2.5 Flash)...")
        model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0).bind_tools(tools)
        extraction_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        
    return model, extraction_model

def extract_travel_data(state: AgentState):
    # Use dictionaries to prevent duplicates (key is flight number or hotel name)
    flights_dict = {}
    hotels_dict = {}

    for msg in state.get("messages", []):
        if msg.type == "tool":
            tool_name = getattr(msg, "name", "")
            try:
                data = json.loads(msg.content)
                if isinstance(data, dict):
                    data = [data]
                    
                if tool_name == "fetch_flights":
                    for f in data:
                        # Use flight number as unique key
                        flight_num = f.get("flight_number", "unknown")
                        flights_dict[flight_num] = f
                elif tool_name == "fetch_hotels":
                    for h in data:
                        # Use hotel name as unique key
                        hotel_name = h.get("name", "unknown")
                        hotels_dict[hotel_name] = h
            except json.JSONDecodeError:
                continue
            
    exclude_keys = {"messages", "step_count"}
    travel_data = {
        key: value 
        for key, value in state.items() 
        if key not in exclude_keys
    }

    # Convert back to a list clean of duplicates
    travel_data["flights"] = list(flights_dict.values())
    travel_data["hotels"] = list(hotels_dict.values())

    return travel_data

# --- Nodes now need to receive the model as a parameter, or we define them as dynamic ---
# To avoid breaking langgraph (which expects functions receiving only state),
# we create a function that "produces" the nodes with the injected models (Closure)

def create_nodes(provider: str):
    """Factory that initializes models and returns the node functions for the LangGraph."""
    model, extraction_model = get_models(provider)
    check_enrichment = EnrichmentNode(extraction_model)

    def extract_metadata(state: AgentState):
        """Extract travel metadata from the conversation and update state."""
        updates = {"step_count": state.get("step_count", 0)}

        if not state.get("messages"):
            return updates

        messages = state.get("messages", [])
        recent_messages = messages[-6:] if messages else []
        
        # Use the chosen extraction_model
        extractor = extraction_model.with_structured_output(TravelMetadata)
        
        metadata: TravelMetadata = extractor.invoke([
            {
                "role": "system",
                "content": """
                Extract travel metadata from the conversation.
                Only fill a field if it is explicitly mentioned or very clear.
                Do not guess. If a field is missing, return null.
                Extract: current_city, destination_city, budget.
                """
            },
            *recent_messages,
        ])

        if metadata.current_city is not None:
            updates["current_city"] = metadata.current_city
        if metadata.destination_city is not None:
            updates["destination_city"] = metadata.destination_city
        if metadata.budget is not None:
            updates["total_budget"] = metadata.budget
        if metadata.trip_days is not None:
            updates["trip_days"] = metadata.trip_days

        # Return updates to be merged into the global state
        return updates
    
    def summary_node(state: AgentState):
        """Summarizes the current state for debugging purposes."""
        
        messages = state.get("messages", [])
        existing_summary = state.get("summary", "")
       
        if len(messages) < 3:
            return {}
        
        recent_messages = messages[-5:]
        
        summary_prompt = f"""
            You are a memory management module for a travel agent.
            Your task is to maintain a concise "World State" summary.
            
            EXISTING MEMORY:
            {existing_summary if existing_summary else "No previous memory."}
            
            NEW CONVERSATION SEGMENT:
            Analyze the recent messages and update the memory. 
            Ensure you keep track of:
            1. Origin city
            2. Destination city
            3. Total budget (and currency)
            4. Any specific preferences or constraints mentioned.
            
            Return ONLY the updated summary text.
            """

        response = extraction_model.invoke([
            {"role": "system", "content": summary_prompt},
            *recent_messages
        ])
    
        # Update the summary field in the state
        return {"summary": response.content}
        

    def call_model(state: AgentState):
        """Examines current state and decides whether to trigger a tool or provide answer."""
        current_step = state.get("step_count", 0) + 1
        summary = state.get("summary", "")
        
        clean_history = [
            msg for msg in state.get("messages", [])
            if getattr(msg, "type", "") != "formatter_output"
        ]
        
        origin = state.get('current_city') or "NOT PROVIDED"
        dest = state.get('destination_city') or "NOT PROVIDED"
        trip_days = state.get('trip_days') or "NOT PROVIDED"
        if state.get('total_budget'):
            budget = state['total_budget']
        elif state.get('budget_optional'):
            budget = "No budget constraint (user opted to skip)"
        else:
            budget = "NOT PROVIDED"
        
        # TODO: Refine Tool Restriction Logic
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
        3. TOOL RESTRICTION: DO NOT call weather, attractions, or cost calculator tools unless the user explicitly asks for them. Focus ONLY on flights and hotels.
        4. CHECK BUDGET: You must use the budget information to filter your search results. Only return options that fit within the user's stated budget. If no options are found within budget, you may expand the search but must clearly indicate when presenting results.
        4. NO HALLUCINATIONS: Check your conversation history. Have you received the JSON response from `fetch_flights` and `fetch_hotels`? If NO, you must call them right now. 
        5. BOUNDARY ENFORCEMENT: Decline any non-travel questions and steer back to the trip.
        
        DO NOT output the final itinerary or list the hotels/flights yourself. Just confirm you found them. DO NOT ask the user any questions. 
        The system will handle formatting the actual list.
        """
        
        recent_history = clean_history[-6:]
        messages_to_pass = [{"role": "system", "content": system_prompt}] + recent_history
        
        # Use the model with tools
        response = model.invoke(messages_to_pass)
        
        return {
            "messages": [response],
            "step_count": current_step
        }
        
    def formatter(state: AgentState):
        """Final node to format the gathered travel data into a pretty Markdown response."""
        if not state.get("messages"):
            return {}

        last_agent_message = state["messages"][-1].content if state.get("messages") else ""
        
        if "let me know" in last_agent_message.lower():
            return {}
        
        has_origin = bool(state.get('current_city'))
        has_dest = bool(state.get('destination_city'))
        has_budget = bool(state.get('total_budget')) or bool(state.get('budget_optional'))

        if not (has_origin and has_dest and has_budget):
            return {}

        travel_data = extract_travel_data(state)
        # TODO: Adding Number of Nights based on the budget.
        system_prompt = """
        You are a strict data formatter. Your ONLY job is to output the provided <data> into the EXACT Markdown template below.
        
        CRITICAL RULES:
        1. DO NOT add conversational filler.
        2. FORCE CURRENCY: You MUST use the '$' symbol for ALL prices and budgets. DO NOT use '€' or '£'.
        3. DO NOT ask the user any questions.
        4. Use the exact currency provided in the budget.
        5. If any section has no data, use the appropriate fallback text provided in the template.
        6. Do not invent flights or hotels. Use ONLY what is in the <data>.
        3. STRICT CONDITIONAL LOGIC: Follow the IF/ELSE logic in the template perfectly. Do not output the fallback text if data exists.
        7. YOU MUST USE THIS EXACT TEMPLATE:

        [IF NO FLIGHTS ARE FOUND, USE THIS EXACT TEXT AND DO NOT ADD ANYTHING]
        Based on our search, we unfortunately could not find any available flights from your origin to [Destination City] at this time.

        ---
        [IF FLIGHTS ARE FOUND IN THE DATA, USE THIS FORMAT:]
        [Greeting tailored to the destination]
        ### ✨ **Your [Destination City] Escape** ✨

        **Destination:** [City Name, Country]
        **Total Budget:** [Budget with correct currency symbol]
        
        **Total Price :** [Total price of the flight + hotel with correct currency symbol]

        ---

        ### ✈️ **Your Flight Details**
        Based on our search, we have found the following flight option:
        * **Airline:** [Airline Name]
        * **Flight Number:** [Flight Number]
        * **Price:** [Price with correct currency symbol]
        * **Status:** Available

        ---

        ### 🏨 **Accommodation Options in [Destination City]**

        [IF HOTELS ARE FOUND IN THE DATA, USE THIS FORMAT:]
        Based on our search, we've found excellent options to suit different preferences:

        **1. [Hotel Name]**
            * [Star Emojis corresponding to rating, e.g., ⭐ ⭐ ⭐] ([Number] Stars)
            * **Price Per Night:** [Price with correct currency symbol]
            * **Highlights:** [Brief, engaging sentence summarizing amenities]

        [Repeat numbered list for additional hotels]
        [Appropriate closing sign-off]
        """

        messages_to_pass = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"<data>\n{travel_data}\n</data>"}
        ]

        response = extraction_model.invoke(messages_to_pass)
        response.name = "formatter_output" # Set name to avoid API validation errors for certain providers

        return {"messages": [response]}
        
    # Return both functions ready for graph execution
    return extract_metadata, check_enrichment, call_model, formatter, summary_node
