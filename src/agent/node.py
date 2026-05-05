import json
from typing import Optional
from pydantic import BaseModel, Field
from agent.state import AgentState
from tools.tools import tools
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Pydantic type definitions remain the same
class TravelMetadata(BaseModel):
    current_city: Optional[str] = Field(default=None, description="The city the user is currently in / starting from")
    destination_city: Optional[str] = Field(default=None, description="The city the user wants to travel to")
    budget: Optional[float] = Field(default=None, description="The user's travel budget as a number, if mentioned")

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
    flights = []
    hotels = []

    # Iterate through message history to find tool outputs
    for msg in state.get("messages", []):
        if msg.type == "tool":
            tool_name = getattr(msg, "name", "")
            try:
                # Parse the tool output content from JSON string
                data = json.loads(msg.content)
                if isinstance(data, dict):
                    data = [data]
                    
                if tool_name == "fetch_flights":
                    flights.extend(data)
                elif tool_name == "fetch_hotels":
                    hotels.extend(data)
            except json.JSONDecodeError:
                continue
            
    # Keys to ignore when copying state variables
    exclude_keys = {"messages", "step_count"}
    
    # Build a dictionary of current state variables (origin, destination, budget)
    travel_data = {
        key: value 
        for key, value in state.items() 
        if key not in exclude_keys
    }

    # Append the collected tool results
    travel_data["flights"] = flights
    travel_data["hotels"] = hotels

    return travel_data


# --- Nodes now need to receive the model as a parameter, or we define them as dynamic ---
# To avoid breaking langgraph (which expects functions receiving only state),
# we create a function that "produces" the nodes with the injected models (Closure)

def create_nodes(provider: str):
    model, extraction_model = get_models(provider)
    
    def extract_metadata(state: AgentState):
        """Extract travel metadata from the conversation and update state."""
        updates = {"step_count": state.get("step_count", 0)}

        if not state.get("messages"):
            return updates

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
            *state["messages"],
        ])

        if metadata.current_city is not None:
            updates["current_city"] = metadata.current_city
        if metadata.destination_city is not None:
            updates["destination_city"] = metadata.destination_city
        if metadata.budget is not None:
            updates["total_budget"] = metadata.budget

        return updates
    
    def summary_node(state: AgentState):
        """Summarizes the current state for debugging purposes."""
        
        messages = state.get("messages", [])
        existing_summary = state.get("summary", "")
       
        if len(messages) < 3:
            return {}
        
        summary_prompt = f"""
            You are a memory management module for a travel agent.
            Your task is to maintain a concise "World State" summary.
            
            EXISTING MEMORY:
            {existing_summary if existing_summary else "No previous memory."}
            
            CRITICAL INSTRUCTIONS:
            1. Use the 'CONTEXT' above to avoid asking the user questions they have already answered.
            2. If the user mentions a new preference that conflicts with the 'CONTEXT', prioritize the new information.
            3. Only call tools if you have a clear Origin, Destination, and Budget.
            
            NEW CONVERSATION SEGMENT:
            Analyze the recent messages and update the memory. 
            Ensure you keep track of:
            1. Origin city
            2. Destination city
            3. Total budget (and currency)
            4. Any specific preferences or constraints mentioned.
            
            Return only the updated summary text.
            """

        response = extraction_model.invoke([
            {"role": "system", "content": summary_prompt},
            *messages
        ])
    
        return {"summary": response.content}
        

    def call_model(state: AgentState):
        """Examines current state and decides whether to trigger a tool or provide answer."""
        current_step = state.get("step_count", 0) + 1
        summary = state.get("summary", "")
        
        clean_history = [
            msg for msg in state.get("messages", [])
            if getattr(msg, "type", "") != "formatted_response"
        ]
        
        origin = state.get('current_city') or "NOT PROVIDED"
        dest = state.get('destination_city') or "NOT PROVIDED"
        budget = state.get('total_budget') or "NOT PROVIDED"
        
        system_prompt = f"""You are AtlasAI, a strict and professional luxury travel assistant.       
        CONTEXT FROM PREVIOUS EXCHANGES:
        {summary if summary else "No previous context. This is a new conversation."}
        
        CURRENT TRIP STATUS:
        - User is currently in: {origin}
        - User wants to travel to: {dest}
        - User's budget: {budget}

        CRITICAL INSTRUCTIONS:
        1. Address the user's specific prompt. 
        2. Use tools ONLY if you need missing information.
        3. If you have all the information needed to answer the user, provide a final conversational answer and DO NOT call any more tools.
        
        CRITICAL INSTRUCTIONS & GUARDRAILS:
        1. BOUNDARY ENFORCEMENT: You are a travel agent ONLY. If the user asks about math (e.g., 5+5), coding, politics, or ANY non-travel topic, you MUST politely decline to answer and immediately steer the conversation back to their travel plans. Do NOT provide the answer to off-topic questions.
        2. MISSING INFO: If ANY of the 'CURRENT TRIP STATUS' fields are 'NOT PROVIDED', ask the user politely for the missing information. Do not search until you have all three.
        3. AVOID DUPLICATION: Do not repeat questions if the user has already provided the answer in the context.
        """

        messages_to_pass = [{"role": "system", "content": system_prompt}] + clean_history
        
        # Use the model with tools
        response = model.invoke(messages_to_pass)
        
        return {
            "messages": [response],
            "step_count": current_step
        }
        
    def formatter(state: AgentState):
        if not state.get("messages"):
            return {}

        has_origin = bool(state.get('current_city'))
        has_dest = bool(state.get('destination_city'))
        has_budget = bool(state.get('total_budget'))

        if not (has_origin and has_dest and has_budget):
            return {}

        travel_data = extract_travel_data(state)
        
        system_prompt = """
        You are a strict luxury travel concierge. Your ONLY task is to format the raw data into the beautiful Markdown template below.

        CRITICAL RULES:
        1. DO NOT add conversational filler.
        2. DO NOT ask the user any questions.
        3. Use the exact currency provided in the budget.
        4. If any section has no data, use the appropriate fallback text provided in the template.
        5. YOU MUST USE THIS EXACT TEMPLATE:

        [Greeting tailored to the destination]
        ---
        ### ✨ **Your [Destination City] Escape** ✨
        **Destination:** [City Name, Country]
        **Total Budget:** [Budget]
        ---
        ### ✈️ **Your Flight Details**
        [List flights if found, otherwise: "Based on our search, we unfortunately could not find any available flights..."]
        ---
        ### 🏨 **Accommodation Options**
        [List hotels if found, otherwise: "Based on our search, we unfortunately could not find any available accommodations..."]
        ---
        [Appropriate closing sign-off]
        """

        messages_to_pass = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"<data>\n{travel_data}\n</data>"}
        ]

        response = extraction_model.invoke(messages_to_pass)
        response.name = "formatter_output" # שומר על התיקון מהשלב הקודם כדי לא להקריס את ה-API

        return {"messages": [response]}
        
    # Return both functions ready for graph execution
    return extract_metadata, call_model, formatter,summary_node