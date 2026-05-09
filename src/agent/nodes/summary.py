from langchain_core.messages import RemoveMessage
from agent.state import AgentState

class SummaryNode:
    def __init__(self, extraction_model):
        self.extraction_model = extraction_model

    def __call__(self, state: AgentState):
        """Summarizes the current state and clears the message history."""
        messages = state.get("messages", [])
        existing_summary = state.get("summary", "")
       
        # We only want to summarize if there has been actual progress
        if len(messages) < 3:
            return {}
        
        recent_messages = messages[-6:]
        
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
            5. CRITICAL: If the system found specific flights or hotels (e.g., from tool responses), 
               summarize their details (airline, flight numbers, hotel names, prices) so the agent 
               remembers them for the next turn.
            
            Return ONLY the updated summary text.
            """

        response = self.extraction_model.invoke([
            {"role": "system", "content": summary_prompt},
            *recent_messages
        ])
    
        new_summary = response.content
        
        messages_to_keep = 2
        messages_to_delete = messages[:-messages_to_keep] 
        
        # Deleting all messages to manage context window size
        delete_commands = [RemoveMessage(id=m.id) for m in messages_to_delete if m.id is not None]
        
        return {
            "summary": new_summary,
            "messages": delete_commands
        }