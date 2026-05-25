"""Automated scenario runner for the travel agent.

Usage:
    python tests/run_scenario.py <prompts_file.txt> [--session <session_name>]

The prompts file should contain one user message per line.
Lines starting with '#' or empty lines are skipped.

Output is written to <prompts_file>_outputLog.txt in the same directory.
"""
import argparse
import sqlite3
import sys
import warnings
from datetime import datetime
from pathlib import Path

# ── Project imports ──────────────────────────────────────────────────────────
# Add src/ to the path so we can import project modules
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from agent.graph import build_graph
from config.session_name import generate_session_name
from config.setting import CHOSEN_PROVIDER
from security import generate_session_id, scan_output

CHECKPOINT_DB = SRC_DIR.parent / "data" / "checkpoints.db"

# ── ANSI colors for terminal output ──────────────────────────────────────────
class C:
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    DIM     = "\033[2m"
    BOLD    = "\033[1m"
    RESET   = "\033[0m"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a scripted scenario against the travel agent")
    parser.add_argument("prompts_file", type=str, help="Path to text file with one user prompt per line")
    parser.add_argument("--session", default=None, help="Session name to use (default: auto-generated)")
    return parser.parse_args()


def _load_prompts(path: Path) -> list[str]:
    """Load user prompts from a text file (skip comments and blanks)."""
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                prompts.append(stripped)
    return prompts


def _build_output_path(input_path: Path) -> Path:
    """Build the output log path: same dir, same stem + _outputLog.txt"""
    return input_path.parent / f"{input_path.stem}_outputLog.txt"


def _format_state(data: dict) -> str:
    """Format current state as a readable one-liner."""
    origin = data.get("current_city", "—") or "—"
    dest   = data.get("destination_city", "—") or "—"
    budget = data.get("total_budget", "—") or "—"
    days   = data.get("trip_days", "—") or "—"
    return f"origin={origin} | destination={dest} | budget={budget} | days={days}"


def _run_scenario(prompts: list[str], session_id: str, output_path: Path) -> None:
    """Run all prompts through the agent graph and write the conversation log."""
    CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)

    try:
        checkpointer = SqliteSaver(conn=conn, serde=JsonPlusSerializer())
        graph = build_graph(provider=CHOSEN_PROVIDER, checkpointer=checkpointer)
        config = {"configurable": {"thread_id": session_id}}

        log_lines: list[str] = []

        # ── Header ───────────────────────────────────────────────────────
        header = (
            f"{'='*70}\n"
            f"  SCENARIO RUN LOG\n"
            f"  Session:  {session_id}\n"
            f"  Provider: {CHOSEN_PROVIDER}\n"
            f"  Started:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"  Prompts:  {len(prompts)}\n"
            f"{'='*70}\n"
        )
        log_lines.append(header)
        print(f"\n{C.MAGENTA}{C.BOLD}{header}{C.RESET}")

        # ── Run each prompt ──────────────────────────────────────────────
        for i, prompt in enumerate(prompts, 1):
            separator = f"\n{'─'*70}\n"
            user_header = f"[TURN {i}/{len(prompts)}] USER:\n{prompt}\n"

            log_lines.append(separator)
            log_lines.append(user_header)

            print(f"{C.YELLOW}{separator}{C.RESET}", end="")
            print(f"{C.CYAN}{C.BOLD}❯ [{i}/{len(prompts)}] USER:{C.RESET} {prompt}")
            print(f"{C.DIM}  Processing...{C.RESET}")

            initial_state = {"messages": [("user", prompt)], "step_count": 0}
            last_content = ""
            current_state_str = ""

            try:
                for mode, data in graph.stream(initial_state, config, stream_mode=["values", "updates"]):
                    if mode == "updates":
                        node_name = next(iter(data))
                        node_line = f"  ── {node_name} ──"
                        log_lines.append(node_line + "\n")
                        print(f"{C.MAGENTA}  ── {node_name} ──{C.RESET}")
                        continue

                    messages = data.get("messages", [])
                    if not messages:
                        continue

                    last_msg = messages[-1]
                    content = str(last_msg.content) if hasattr(last_msg, "content") else "No content"
                    content = scan_output(content, session_id=session_id)
                    current_state_str = _format_state(data)

                    if content == last_content:
                        continue

                    msg_type = last_msg.__class__.__name__
                    agent_line = f"\n[{msg_type}]:\n{content}\n"
                    log_lines.append(agent_line)
                    print(f"{C.GREEN}  [{msg_type}]{C.RESET} {content[:200]}{'...' if len(content) > 200 else ''}")
                    last_content = content

                # Log state after turn
                state_line = f"\n[STATE]: {current_state_str}\n"
                log_lines.append(state_line)
                print(f"{C.DIM}  [STATE] {current_state_str}{C.RESET}")

            except Exception as e:
                error_line = f"\n[ERROR]: {e}\n"
                log_lines.append(error_line)
                print(f"{C.RED}  [ERROR] {e}{C.RESET}")

        # ── Footer ───────────────────────────────────────────────────────
        footer = (
            f"\n{'='*70}\n"
            f"  SCENARIO COMPLETE\n"
            f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"  Turns:    {len(prompts)}\n"
            f"{'='*70}\n"
        )
        log_lines.append(footer)
        print(f"\n{C.MAGENTA}{C.BOLD}{footer}{C.RESET}")

        # ── Write output file ────────────────────────────────────────────
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("".join(log_lines))

        print(f"{C.GREEN}{C.BOLD}✅ Log saved to: {output_path}{C.RESET}\n")

    finally:
        conn.close()


def main():
    args = _parse_args()
    prompts_path = Path(args.prompts_file).resolve()

    if not prompts_path.exists():
        print(f"{C.RED}Error: File not found: {prompts_path}{C.RESET}")
        sys.exit(1)

    prompts = _load_prompts(prompts_path)
    if not prompts:
        print(f"{C.RED}Error: No prompts found in {prompts_path}{C.RESET}")
        sys.exit(1)

    session_id = args.session or f"{generate_session_name()}-{generate_session_id()[:8]}"
    output_path = _build_output_path(prompts_path)

    print(f"\n{C.CYAN}📂 Input:   {prompts_path}{C.RESET}")
    print(f"{C.CYAN}📝 Output:  {output_path}{C.RESET}")
    print(f"{C.CYAN}🔑 Session: {session_id}{C.RESET}")
    print(f"{C.CYAN}📊 Prompts: {len(prompts)}{C.RESET}")

    _run_scenario(prompts, session_id, output_path)


if __name__ == "__main__":
    main()
