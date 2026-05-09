# src/agent/nodes/metadata.py
from agent.state import AgentState
from agent.models import TravelMetadata 

class MetadataNode:
    def __init__(self, extraction_model):
        self.extraction_model = extraction_model

    def __call__(self, state: AgentState):
        updates = {"step_count": state.get("step_count", 0)}

        if not state.get("messages"):
            return updates

        messages = state.get("messages", [])
        recent_messages = [
            msg for msg in messages[-10:]
            if getattr(msg, "type", "") in ("human", "ai")
            and not getattr(msg, "tool_calls", None)
        ][-6:]

        extractor = self.extraction_model.with_structured_output(TravelMetadata)
        
        metadata: TravelMetadata = extractor.invoke([
            {
                "role": "system",
                "content": """
                Extract travel metadata from the conversation.
                Only fill a field if it is explicitly mentioned or very clear.
                Do not guess. If a field is missing, return null.
                Extract: current_city, destination_city, budget, trip_days.
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

        return updates