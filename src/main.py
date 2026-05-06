import sys
from langgraph.checkpoint import memory
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from agent.state import AgentState
from agent.edge import should_continue
from tools.tools import tools

# Note: We are now importing the function that creates the nodes, not the nodes themselves
from agent.node import create_nodes 

# --- Choose which model provider to run here ---
# Change to "groq" if Google quota is exceeded, or "google" if you have a valid key
CHOSEN_PROVIDER = "groq"  # or "groq"

def build_graph(provider: str = "google"):
    """
    Builds the graph using the specified model provider ('google' or 'groq').
    """
    # 1. Create the nodes with the chosen model provider
    extract_metadata_node, call_model_node, formatter, summary_node = create_nodes(provider)
    
    # 2. Build the standard graph
    builder = StateGraph(AgentState)

    builder.add_node("extract_metadata", extract_metadata_node)
    builder.add_node("agent", call_model_node)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("formatter", formatter)
    builder.add_node("summary", summary_node)

    # 3. Define the workflow edges
    builder.add_edge(START, "extract_metadata")
    builder.add_edge("extract_metadata", "agent")
    builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "formatter": "formatter"})
    builder.add_edge("tools", "agent")
    builder.add_edge("formatter", "summary")
    # The summary node marks the end of the processing cycle for the current turn
    builder.add_edge("summary", END)
    # Adding a checkpointer to save the agent's state across turns
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)

def run_agent():    
    graph = build_graph(provider=CHOSEN_PROVIDER)
    config = {"configurable": {"thread_id": "student_session_01"}}
    
    print(f"--- Autonomous Travel Agent Started ({CHOSEN_PROVIDER.upper()}) ---")
    print("Type 'exit' or 'quit' to end the session.")
    print("-" * 50)
    print("             Atlas AI Travel Assistant             ")
    print("-" * 50)
    print("Agent: Hello! I'm your travel assistant. Where are you starting from and where would you like to go?")
    print("-" * 50)


    while True:
        user_input = input("\nUser: ")
        if user_input.strip().lower() in ["exit", "quit"]: 
            print("Goodbye!")
            break
            
        initial_state = {
            "messages": [("user", user_input)],
            "step_count": 0
        }
        
        last_printed_content = ""
        last_printed_state = ()

        for event in graph.stream(initial_state, config, stream_mode="values"):
            last_msg = event["messages"][-1]
            msg_type = last_msg.__class__.__name__
            
            content = str(last_msg.content) if hasattr(last_msg, 'content') else "No content"
            
            current = event.get('current_city', 'None')
            dest = event.get('destination_city', 'None')
            budget = event.get('total_budget', 'None')
            current_state_tuple = (current, dest, budget)
            
            if content != last_printed_content or current_state_tuple != last_printed_state:
                
                if content != last_printed_content:
                    msg_type = last_msg.__class__.__name__
                    print(f"[{msg_type}] Content: {content}")
                    last_printed_content = content
                
                print(f"State -> Origin: {current} | Dest: {dest} | Budget: {budget}")
                print("-" * 20)
                last_printed_state = current_state_tuple

if __name__ == "__main__":
    run_agent()