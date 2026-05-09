from agent.graph import build_graph
from config.setting import CHOSEN_PROVIDER
import uuid

def run_agent():    
    graph = build_graph(provider=CHOSEN_PROVIDER)
    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}
    
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

        try:
            for mode, data in graph.stream(initial_state, config, stream_mode=["values", "updates"]):
                if mode == "updates":
                    current_node = next(iter(data))
                    print(f"\n{'='*10} Node: {current_node} {'='*10}")
                    continue

                messages = data.get("messages", [])
                if not messages:
                    continue
                            
                # mode == "values" — full state snapshot
                last_msg = messages[-1]
                msg_type = last_msg.__class__.__name__
                content = str(last_msg.content) if hasattr(last_msg, 'content') else "No content"

                current = data.get('current_city', 'None')
                dest = data.get('destination_city', 'None')
                budget = data.get('total_budget', 'None')
                trip_days = data.get('trip_days', 'None')
                current_state_tuple = (current, dest, budget, trip_days)

                if content != last_printed_content or current_state_tuple != last_printed_state:
                    if content != last_printed_content:
                        print(f"[{msg_type}] Content: {content}")
                        last_printed_content = content
                    print(f"State -> Origin: {current} | Dest: {dest} | Budget: {budget} | Trip Days: {trip_days}")
                    print("-" * 20)
                    last_printed_state = current_state_tuple
        except Exception as e:
            print(f"\n[Error] Connection failed. Please check your internet connection and API key. Details: {e}")
            print("Shutting down gracefully...")
            break

if __name__ == "__main__":
    run_agent()