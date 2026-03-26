from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """
    The state of the agent, tracked across the graph execution.
    'add_messages' ensures history is preserved.
    """
    messages: Annotated[list, add_messages]
    current_city: str
    total_budget: float