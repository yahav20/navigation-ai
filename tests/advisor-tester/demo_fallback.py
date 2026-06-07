"""
Demo: watch the replanner fallback substitution fire in isolation.

Uses the tag 'eco-friendly' which exists in the planner's available tag list
but has NO cities in the DB. This naturally causes find_destinations_by_tag to
return empty without any DB modification.

The replanner should:
  1. See find_destinations_by_tag(tag='eco-friendly') → empty (tag not in DB)
  2. Detect the empty discovery result and call the LLM for substitution
  3. Insert find_destinations_by_vibe(category='Nature') as the substitute step
  4. Execute the substitute → real destinations returned
  5. Format a useful response

Run from the project root:
    python tests/advisor-tester/demo_fallback.py
"""

import os
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from agent.core.graph import build_graph
from config.config import CHOSEN_PROVIDER
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

# 'eco-friendly' is in the planner's tag list but has NO rows in city_tags.
# The planner will generate find_destinations_by_tag(tag='eco-friendly'),
# the DB will return empty, and the replanner will substitute with find_destinations_by_vibe(category='Nature').
QUERY = "I want to travel somewhere eco-friendly and sustainable, where should I go?"


def run_demo():
    print(f"\n{BOLD}=== Replanner Fallback Demo ==={RESET}")
    print(f"Query : {CYAN}\"{QUERY}\"{RESET}")
    print(f"\n{YELLOW}Why this triggers the fallback:{RESET}")
    print(f"  The planner will choose find_destinations_by_tag(tag='eco-friendly').")
    print(f"  'eco-friendly' exists in the planner's tag list but has NO cities in the DB.")
    print(f"  Expected: replanner detects the empty result and substitutes")
    print(f"            find_destinations_by_vibe(category='Nature').")
    print(f"  No DB modification needed.\n")

    checkpointer = MemorySaver(serde=JsonPlusSerializer())
    graph = build_graph(provider=CHOSEN_PROVIDER, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    initial_state = {"messages": [("user", QUERY)], "step_count": 0}

    print(f"{YELLOW}Running advisor agent...{RESET}\n")
    for update in graph.stream(initial_state, config, stream_mode="updates"):
        pass  # Planner / Executor / Replanner print their own logs

    final_state = graph.get_state(config).values
    tool_results = final_state.get("advisor_last_tool_results") or []

    print(f"\n{YELLOW}Results:{RESET}")
    print(f"  Tools that ran: {len(tool_results)}")
    for r in tool_results:
        result = r.get("result")
        is_empty = (
            not result
            or (isinstance(result, list) and len(result) == 1 and "message" in result[0])
            or (isinstance(result, dict) and "message" in result)
        )
        badge    = f"{RED}empty{RESET}" if is_empty else f"{GREEN}data{RESET}"
        args_str = ", ".join(f"{k}={v!r}" for k, v in r.get("args", {}).items())
        print(f"    [{badge}] {r['tool_name']}({args_str})")

    first_tool  = tool_results[0]["tool_name"] if tool_results else ""
    second_tool = tool_results[1]["tool_name"] if len(tool_results) > 1 else ""

    if len(tool_results) == 2 and first_tool in ("find_destinations_by_tag", "find_destinations_by_vibe"):
        print(f"\n  {GREEN}Fallback fired correctly!{RESET} "
              f"Replanner substituted '{second_tool}' after '{first_tool}' returned empty.")
    elif len(tool_results) == 1:
        print(f"\n  {RED}Fallback did NOT fire.{RESET} "
              f"Only '{first_tool}' ran — replanner finished without substituting.")
    else:
        print(f"\n  {YELLOW}Unexpected tool sequence.{RESET} Check node logs above.")

    messages = final_state.get("messages", [])
    response = next(
        (str(m.content) for m in reversed(messages)
         if getattr(m, "type", "") == "ai" and hasattr(m, "content")),
        ""
    )
    print(f"\n{YELLOW}Agent response preview:{RESET}")
    print(f"  {DIM}{response[:400]}{'...' if len(response) > 400 else ''}{RESET}")


if __name__ == "__main__":
    run_demo()
