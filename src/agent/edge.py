import json

from agent.state import AgentState


def _no_real_flights_in_history(state: AgentState) -> bool:
    """True when the message history contains no usable flight records.

    Covers two cases for the alt branch: (a) the agent called fetch_flights and
    got back a 'no results' message, and (b) the agent skipped fetch_flights
    entirely (the small Llama variants do this sometimes even when the prompt
    insists). In both cases there is no flight to show, so the alt branch is
    the right destination.
    """
    for msg in state.get("messages", []):
        if getattr(msg, "type", "") != "tool":
            continue
        if getattr(msg, "name", "") != "fetch_flights":
            continue
        try:
            data = json.loads(msg.content)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            continue
        for entry in data:
            if isinstance(entry, dict) and "flight_number" in entry:
                return False
    return True


def should_continue(state: AgentState):
    """
    Determines the next path in the graph based on the model's output.
    Returns 'tools' if the model wants to call a function, 'alternative_destination'
    when fetch_flights came back empty for a known origin/destination, otherwise 'formatter'.
    """
    last_message = state["messages"][-1]
    step_count = state.get("step_count", 0)
    origin = state.get("current_city")
    dest = state.get("destination_city")

    # Safety Check: Stop after 5 tool invocations in a single turn
    MAX_STEPS = 5
    if step_count >= MAX_STEPS:
        print("--- Agent stopped due to maximum step count. Possible infinite loop. ---")
        if origin and dest and _no_real_flights_in_history(state):
            return "alternative_destination"
        return "formatter"

    if getattr(last_message, "tool_calls", None):
        return "tools"

    if origin and dest and _no_real_flights_in_history(state):
        return "alternative_destination"

    return "formatter"
