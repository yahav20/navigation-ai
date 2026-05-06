import sys
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from agent.state import AgentState
from agent.edge import should_continue
from tools.tools import tools

# Note: We are now importing the function that creates the nodes, not the nodes themselves
from agent.node import create_nodes
from src.agent.nodes.node_alternative import create_alternative_nodes

# --- Choose which model provider to run here ---
# Change to "groq" if Google quota is exceeded, or "google" if you have a valid key
CHOSEN_PROVIDER = "groq"  # or "groq"

def build_graph(provider: str = "google"):
    """
    Builds the graph using the specified model provider ('google' or 'groq').
    """
    # 1. Create the nodes with the chosen model provider
    extract_metadata_node, call_model_node, formatter = create_nodes(provider)
    alternative_destination_node, formatter_alternative = create_alternative_nodes(provider)

    # 2. Build the standard graph
    builder = StateGraph(AgentState)

    builder.add_node("extract_metadata", extract_metadata_node)
    builder.add_node("agent", call_model_node)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("alternative_destination", alternative_destination_node)
    builder.add_node("formatter", formatter)
    builder.add_node("formatter_alternative", formatter_alternative)

    builder.add_edge(START, "extract_metadata")
    builder.add_edge("extract_metadata", "agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "formatter": "formatter",
            "alternative_destination": "alternative_destination",
        },
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("alternative_destination", "formatter_alternative")
    builder.add_edge("formatter", END)
    builder.add_edge("formatter_alternative", END)

    return builder.compile()

def run_agent():    
    graph = build_graph(provider=CHOSEN_PROVIDER)
    config = {"configurable": {"thread_id": "student_session_01"}}
    
    print(f"--- Autonomous Travel Agent Started ({CHOSEN_PROVIDER.upper()}) ---")

    while True:
        user_input = input("\nUser: ")
        if user_input.strip().lower() in ["exit", "quit"]: 
            print("Goodbye!")
            break
            
        initial_state = {
            "messages": [("user", user_input)],
            "step_count": 0
        }

        for event in graph.stream(initial_state, config, stream_mode="values"):
            last_msg = event["messages"][-1]
            msg_type = last_msg.__class__.__name__
            
            content = str(last_msg.content) if hasattr(last_msg, 'content') else "No content"
            print(f"[{msg_type}] Content: {content}")

            current = event.get('current_city', 'None')
            dest = event.get('destination_city', 'None')
            budget = event.get('total_budget', 'None')
            print(f"State -> Origin: {current} | Dest: {dest} | Budget: {budget}")
            print("-" * 20)

if __name__ == "__main__":
    run_agent()