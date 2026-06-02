"""CLI entry point for the navigation AI agent."""
import argparse
import sqlite3
import time
import warnings
from pathlib import Path

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer  # noqa: E402
from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

import ui  # noqa: E402
from agent.graph import build_graph  # noqa: E402
from config.session_name import generate_session_name  # noqa: E402
from config.setting import CHOSEN_PROVIDER  # noqa: E402
from security import MAX_TURNS_PER_SESSION, generate_session_id, log_turn, scan_output, validate_input  # noqa: E402

CHECKPOINT_DB = Path(__file__).resolve().parent.parent / "data" / "checkpoints.db"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous Travel Agent")
    parser.add_argument(
        "--session",
        default=None,
        help="Session name (thread_id) to resume or create. Defaults to a random name like 'happy-traveler'.",
    )
    return parser.parse_args()


def _restrict_db_permissions(path: Path) -> None:
    """Restrict DB file to current user only (Windows via icacls, Unix via chmod)."""
    import os, platform
    if path.exists():
        if platform.system() == "Windows":
            os.system(f'icacls "{path}" /inheritance:r /grant:r "%USERNAME%":(F) >nul 2>&1')
        else:
            os.chmod(path, 0o600)


def run_agent(session_id: str = "default") -> None:
    """Run the interactive agent loop until the user exits."""
    CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
    _restrict_db_permissions(CHECKPOINT_DB)
    conn = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
    try:
        _interactive_loop(conn, session_id)
    finally:
        conn.close()


def _interactive_loop(conn: sqlite3.Connection, session_id: str) -> None:
    checkpointer = SqliteSaver(conn=conn, serde=JsonPlusSerializer())
    graph = build_graph(provider=CHOSEN_PROVIDER, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": session_id}}

    saved = graph.get_state(config).values
    resuming = bool(saved)
    ui.render_banner(CHOSEN_PROVIDER, session_id, CHECKPOINT_DB, resuming)

    prompt_session = ui.make_prompt_session()
    current_state: tuple[str, str, str, str, str] = (
        saved.get("current_city", "None") if saved else "None",
        saved.get("destination_city", "None") if saved else "None",
        saved.get("total_budget", "None") if saved else "None",
        saved.get("trip_days", "None") if saved else "None",
        saved.get("trip_start", "None") if saved else "None",
    )
    turn_count = 0

    while True:
        if turn_count >= MAX_TURNS_PER_SESSION:
            ui.render_error(Exception("Session limit reached. Please restart the agent."))
            break

        user_input = ui.ask_user(prompt_session, state=current_state)
        if user_input is None:
            ui.render_goodbye(newline=True)
            break
        if user_input.strip().lower() in {"exit", "quit"}:
            ui.render_goodbye()
            break
        if not user_input.strip():
            continue

        try:
            validate_input(user_input, session_id=session_id)
        except ValueError as e:
            ui.render_error(e)
            continue

        log_turn(session_id, user_input, turn=turn_count + 1)
        turn_count += 1

        try:
            current_state = _run_turn(graph, config, user_input, current_state, session_id, prompt_session)
        except Exception as e:  # noqa: BLE001  # top-level handler: report connection/runtime errors to user
            ui.render_error(e)
            break


def _run_turn(
    graph,
    config: dict,
    user_input: str,
    current_state: tuple[str, str, str, str, str],
    session_id: str = "unknown",
    prompt_session=None,
) -> tuple[str, str, str, str, str]:
    """Run one user turn, including any HITL interrupt-resume cycles."""
    _SELF_REPORTING_NODES = {"advisor_planner", "advisor_executor", "advisor_replanner"}

    # First stream input is the user's message; on interrupt-resume it becomes a Command.
    stream_input = {"messages": [("user", user_input)], "step_count": 0}
    last_content = ""

    while True:
        last_node_start = time.perf_counter()
        pending_interrupt: str | dict | None = None

        with ui.thinking(current_state) as display:
            for mode, data in graph.stream(stream_input, config, stream_mode=["values", "updates"]):
                if mode == "updates":
                    node_name = next(iter(data))
                    elapsed_ms = (time.perf_counter() - last_node_start) * 1000
                    last_node_start = time.perf_counter()

                    # LangGraph surfaces interrupt() calls as a special update key.
                    if node_name == "__interrupt__":
                        interrupts = data["__interrupt__"]
                        # Preserve dict payloads (choice widgets) intact; stringify plain text.
                        pending_interrupt = interrupts[0].value if interrupts else ""
                        continue

                    if node_name not in _SELF_REPORTING_NODES:
                        ui.render_node(node_name, elapsed_ms=elapsed_ms)
                    continue

                messages = data.get("messages", [])
                if not messages:
                    continue

                last_msg = messages[-1]
                content = str(last_msg.content) if hasattr(last_msg, "content") else "No content"
                content = scan_output(content, session_id=session_id)
                current_state = (
                    data.get("current_city", "None"),
                    data.get("destination_city", "None"),
                    data.get("total_budget", "None"),
                    data.get("trip_days", "None"),
                    data.get("trip_start", "None"),
                )
                display.update(current_state)

                if content != last_content:
                    ui.render_agent_message(last_msg.__class__.__name__, content)
                    last_content = content

        # No interrupt — turn is complete.
        if pending_interrupt is None:
            break

        # ── HITL: show the question and collect the user's choice ──────
        if isinstance(pending_interrupt, dict) and "options" in pending_interrupt:
            # Structured interrupt: render the question then show arrow-key picker.
            question = pending_interrupt.get("question", "")
            if question:
                ui.render_agent_message("Atlas", question)
            user_response = ui.ask_choice(
                options=pending_interrupt["options"],
                state=current_state,
            )
        else:
            # Plain-text interrupt: free-form input.
            ui.render_agent_message("Atlas", str(pending_interrupt))
            user_response = ui.ask_user(prompt_session, state=current_state)

        if user_response is None or user_response.strip().lower() in ("exit", "quit"):
            break

        # Resume the paused graph with the user's answer.
        stream_input = Command(resume=user_response.strip())
        last_content = ""  # reset so the redirect AIMessage is shown

    return current_state


if __name__ == "__main__":
    args = _parse_args()
    run_agent(session_id=args.session or f"{generate_session_name()}-{generate_session_id()[:8]}")
