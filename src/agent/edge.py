from agent.state import AgentState

def should_continue(state: AgentState):
    """
    Determines the next path in the graph based on the model's output.
    Returns 'tools' if the model wants to call a function, otherwise 'END'.
    """
    last_message = state["messages"][-1]
    step_count = state.get("step_count", 0)
    
    # Safety Check: Stop after 5 tool invocations in a single turn
    MAX_STEPS = 10
    if step_count >= MAX_STEPS:
        print("--- Agent stopped due to maximum step count. Possible infinite loop. ---")
        return "formatter"
        
    if last_message.tool_calls:
        return "tools"
        
    return "formatter"