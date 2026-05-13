"""CLI entry point for the navigation AI agent."""
import argparse
import sqlite3
import warnings
from pathlib import Path

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer  # noqa: E402
from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402

import ui  # noqa: E402
from agent.graph import build_graph  # noqa: E402
from config.session_name import generate_session_name  # noqa: E402
from config.setting import CHOSEN_PROVIDER  # noqa: E402

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
    ui.render_banner(CHOSEN_PROVIDER, session_id, CHECKPOINT_DB, resuming)

    prompt_session = ui.make_prompt_session()

    while True:
        user_input = ui.ask_user(prompt_session)
        if user_input is None:
            ui.render_goodbye(newline=True)
            break
        if user_input.strip().lower() in {"exit", "quit"}:
            ui.render_goodbye()
            break
        if not user_input.strip():
            continue

        try:
            _run_turn(graph, config, user_input)
        except Exception as e:  # noqa: BLE001  # top-level handler: report connection/runtime errors to user
            ui.render_error(e)
            break


def _run_turn(graph, config: dict, user_input: str) -> None:
    initial_state = {"messages": [("user", user_input)], "step_count": 0}
    last_content = ""
    last_state: tuple = ()

    with ui.thinking():
        for mode, data in graph.stream(initial_state, config, stream_mode=["values", "updates"]):
            if mode == "updates":
                ui.render_node(next(iter(data)))
                continue

            messages = data.get("messages", [])
            if not messages:
                continue

            last_msg = messages[-1]
            content = str(last_msg.content) if hasattr(last_msg, "content") else "No content"
            state_tuple = (
                data.get("current_city", "None"),
                data.get("destination_city", "None"),
                data.get("total_budget", "None"),
                data.get("trip_days", "None"),
            )

            if content == last_content and state_tuple == last_state:
                continue
            if content != last_content:
                ui.render_agent_message(last_msg.__class__.__name__, content)
                last_content = content
            ui.render_state(*state_tuple)
            last_state = state_tuple


if __name__ == "__main__":
    args = _parse_args()
    run_agent(session_id=args.session or generate_session_name())
