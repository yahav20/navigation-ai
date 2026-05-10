"""CLI entry point for the navigation AI agent."""
import uuid

from agent.graph import build_graph
from config.setting import CHOSEN_PROVIDER


def run_agent() -> None:
    """Run the interactive agent loop until the user exits."""
    graph = build_graph(provider=CHOSEN_PROVIDER)
    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}



    while True:
        user_input = input("\nUser: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            break

        initial_state = {
            "messages": [("user", user_input)],
            "step_count": 0,
        }

        last_printed_content = ""
        last_printed_state = ()

        try:
            for mode, data in graph.stream(initial_state, config, stream_mode=["values", "updates"]):
                if mode == "updates":
                    next(iter(data))
                    continue

                messages = data.get("messages", [])
                if not messages:
                    continue

                # mode == "values" — full state snapshot
                last_msg = messages[-1]
                content = str(last_msg.content) if hasattr(last_msg, "content") else "No content"

                current = data.get("current_city", "None")
                dest = data.get("destination_city", "None")
                budget = data.get("total_budget", "None")
                trip_days = data.get("trip_days", "None")
                current_state_tuple = (current, dest, budget, trip_days)

                if content != last_printed_content or current_state_tuple != last_printed_state:
                    if content != last_printed_content:
                        last_printed_content = content
                    last_printed_state = current_state_tuple
        except Exception:  # noqa: BLE001  # top-level handler: report any unhandled error to user
            break

if __name__ == "__main__":
    run_agent()
