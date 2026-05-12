"""CLI entry point for the navigation AI agent."""
import argparse
import sqlite3
import warnings
from pathlib import Path

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer  # noqa: E402
from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402

from agent.graph import build_graph  # noqa: E402
from config.setting import CHOSEN_PROVIDER  # noqa: E402
from config.session_name import generate_session_name  # noqa: E402

BANNER = r"""
       _   _____ _      _    ____
      / \ |_   _| |    / \  / ___|
     / _ \  | | | |   / _ \ \___ \
    / ___ \ | | | |__/ ___ \ ___) |
   /_/   \_\|_| |_____/_/   \_\____/
"""

CHECKPOINT_DB = Path(__file__).resolve().parent.parent / "data" / "checkpoints.db"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous Travel Agent")
    parser.add_argument(
        "--session",
        default=None,
        help="Session name (thread_id) to resume or create. Defaults to a random name like 'happy-traveler'.",
    )
    return parser.parse_args()


def run_agent(session_id: str = "default") -> None:
    """Run the interactive agent loop until the user exits."""
    CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
    try:
        _interactive_loop(conn, session_id)
    finally:
        conn.close()


def _interactive_loop(conn: sqlite3.Connection, session_id: str) -> None:
    checkpointer = SqliteSaver(conn=conn, serde=JsonPlusSerializer())
    graph = build_graph(provider=CHOSEN_PROVIDER, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": session_id}}

    resuming = bool(graph.get_state(config).values)

    print(BANNER)
    print(f"--- Autonomous Travel Agent Started ({CHOSEN_PROVIDER.upper()}) ---")
    print(f"Session: {session_id} (state persisted to {CHECKPOINT_DB})")
    print("Type 'exit' or 'quit' to end the session.")
    print("-" * 50)
    if resuming:
        print(f"Agent: Welcome back! Resuming session '{session_id}'. How can I help you continue?")
    else:
        print("Agent: Hello! I'm your travel assistant. Where are you starting from and where would you like to go?")
    print("-" * 50)

    while True:
        user_input = input("\nUser: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        initial_state = {
            "messages": [("user", user_input)],
            "step_count": 0,
        }

        last_printed_content = ""
        last_printed_state = ()
        current_node = "unknown"

        try:
            for mode, data in graph.stream(initial_state, config, stream_mode=["values", "updates"]):
                if mode == "updates":
                    current_node = next(iter(data))
                    print(f"\n{'=' * 10} Node: {current_node} {'=' * 10}")
                    continue

                messages = data.get("messages", [])
                if not messages:
                    continue

                # mode == "values" — full state snapshot
                last_msg = messages[-1]
                msg_type = last_msg.__class__.__name__
                content = str(last_msg.content) if hasattr(last_msg, "content") else "No content"

                current = data.get("current_city", "None")
                dest = data.get("destination_city", "None")
                budget = data.get("total_budget", "None")
                trip_days = data.get("trip_days", "None")
                current_state_tuple = (current, dest, budget, trip_days)

                if content != last_printed_content or current_state_tuple != last_printed_state:
                    if content != last_printed_content:
                        print(f"[{msg_type}] Content: {content}")
                        last_printed_content = content
                    print(f"State -> Origin: {current} | Dest: {dest} | Budget: {budget} | Trip Days: {trip_days}")
                    print("-" * 20)
                    last_printed_state = current_state_tuple
        except Exception as e:  # noqa: BLE001  # top-level handler: report connection/runtime errors to user
            print(f"\n[Error] Connection failed. Please check your internet connection and API key. Details: {e}")
            print("Shutting down gracefully...")
            break


if __name__ == "__main__":
    args = _parse_args()
    run_agent(session_id=args.session or generate_session_name())
