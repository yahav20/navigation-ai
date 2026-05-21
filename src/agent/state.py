"""Typed state shared across nodes of the travel-agent graph."""
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Track the state of the agent across the graph execution.

    'add_messages' ensures history is preserved.
    """

    messages: Annotated[list, add_messages]
    current_city: str
    destination_city: str
    total_budget: float
    step_count: int
    enrichment_complete: bool
    user_preferences: dict
    is_adjustment: bool            # True when the user just modified their trip parameters
    enrichment_asked_fields: list  # field keys already requested from the user
    budget_optional: bool          # True when user explicitly declined to provide a budget
    trip_days: int                 # number of trip days; defaults to 3 if user skips after being asked
    summary: str                   # rolling conversation summary maintained by summary_node
    alternative_destinations: list # populated when fetch_flights returns no results for the route
    plan : list[str]               # list of steps the agent plans to take, updated by the agent_core node
    intent: str                    # intent classification
    advisor_plan: list                 # list of {tool_name, args} dicts produced by advisor_planner
    advisor_last_tool_results: list    # [{tool_name, args, result}] from the most recent executor run
    advisor_shown_cities: list         # accumulates every city name presented to the user across advisor turns

