import json
from agent.state import AgentState
from langgraph.graph import END

MAX_STEPS = 10


def _fetch_flights_returned_empty(state: AgentState) -> bool:
    """True iff fetch_flights ran at least once and none of its results
    contained a real flight (i.e. only message-only payloads)."""
    saw_fetch_flights = False
    for msg in state.get("messages", []):
        if msg.type != "tool":
            continue
        if getattr(msg, "name", "") != "fetch_flights":
            continue
        saw_fetch_flights = True
        try:
            data = json.loads(msg.content)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            data = [data]
        if any(isinstance(item, dict) and "flight_number" in item for item in data):
            return False
    return saw_fetch_flights


def should_continue(state: AgentState):
    """
    Determines the next path in the graph based on the model's output.
    """
    last_message = state["messages"][-1]
    step_count = state.get("step_count", 0)

    if step_count >= MAX_STEPS:
        print("--- Agent stopped due to maximum step count. Possible infinite loop. ---")
        return "formatter"

    if last_message.tool_calls:
        return "tools"

    if (
        _fetch_flights_returned_empty(state)
        and state.get("current_city")
        and state.get("destination_city")
    ):
        return "alternative_destination"

    return "formatter"
