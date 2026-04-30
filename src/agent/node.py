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

    def call_model(state: AgentState):
        """Examines current state and decides whether to trigger a tool or provide answer."""
        current_step = state.get("step_count", 0) + 1
        
        system_prompt = f"""You are a helpful travel assistant.
        Current State Information:
        - User is currently in: {state.get('current_city', 'Unknown')}
        - User wants to travel to: {state.get('destination_city', 'Unknown')}
        - User's budget: {state.get('total_budget', 'Unknown')}

        CRITICAL INSTRUCTIONS:
        1. Address the user's specific prompt. 
        2. Use tools ONLY if you need missing information.
        3. If you have all the information needed to answer the user, provide a final conversational answer and DO NOT call any more tools.
        """

        messages_to_pass = [{"role": "system", "content": system_prompt}] + state["messages"]
        
        # Use the model with tools
        response = model.invoke(messages_to_pass)
        
        return {
            "messages": [response],
            "step_count": current_step
        }
        
    # Return both functions ready for graph execution
    return extract_metadata, call_model