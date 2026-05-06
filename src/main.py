import sys
from langgraph.checkpoint import memory
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from agent.state import AgentState
from agent.edge import should_continue
from agent.nodes.enrichment import after_enrichment
from tools.tools import tools

# Note: We are now importing the function that creates the nodes, not the nodes themselves
from agent.node import create_nodes
from agent.nodes.node_alternative import create_alternative_nodes

# --- Choose which model provider to run here ---
# Change to "groq" if Google quota is exceeded, or "google" if you have a valid key
CHOSEN_PROVIDER = "google"  # or "groq"

def build_graph(provider: str = "google"):
    """
    Builds the graph using the specified model provider ('google' or 'groq').
    """
    # 1. Create the nodes with the chosen model provider
    extract_metadata_node, enrichment_node, call_model_node, formatter, summary_node = create_nodes(provider)
    alternative_destination_node, formatter_alternative = create_alternative_nodes(provider)

    # 2. Build the standard graph
    builder = StateGraph(AgentState)

    builder.add_node("extract_metadata", extract_metadata_node)
    builder.add_node("enrichment", enrichment_node)
    builder.add_node("agent", call_model_node)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("formatter", formatter)
    builder.add_node("alternative_destination", alternative_destination_node)
    builder.add_node("formatter_alternative", formatter_alternative)
    builder.add_node("summary", summary_node)

    # 3. Define the workflow edges
    builder.add_edge(START, "extract_metadata")
    builder.add_edge("extract_metadata", "enrichment")
    builder.add_conditional_edges("enrichment", after_enrichment, {"agent": "agent", END: END})
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
    builder.add_edge("formatter_alternative", "summary")
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
        current_node = "unknown"

        for mode, data in graph.stream(initial_state, config, stream_mode=["values", "updates"]):
            if mode == "updates":
                current_node = next(iter(data))
                print(f"\n{'='*10} Node: {current_node} {'='*10}")
                continue

            # mode == "values" — full state snapshot
            last_msg = data["messages"][-1]
            msg_type = last_msg.__class__.__name__
            content = str(last_msg.content) if hasattr(last_msg, 'content') else "No content"

            current = data.get('current_city', 'None')
            dest = data.get('destination_city', 'None')
            budget = data.get('total_budget', 'None')
            current_state_tuple = (current, dest, budget)

            if content != last_printed_content or current_state_tuple != last_printed_state:
                if content != last_printed_content:
                    print(f"[{msg_type}] Content: {content}")
                    last_printed_content = content
                print(f"State -> Origin: {current} | Dest: {dest} | Budget: {budget}")
                print("-" * 20)
                last_printed_state = current_state_tuple

if __name__ == "__main__":
    run_agent()