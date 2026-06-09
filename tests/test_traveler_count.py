#!/usr/bin/env python3
"""Automated tests for traveler count (adults / children) + hotel rooms in state.

Two layers:

1. Deterministic UNIT asserts on the shared room rules
   (`compute_default_rooms`, `apply_traveler_updates`) — no LLM, instant.
2. Multi-turn INTEGRATION cases that drive the real compiled graph and assert
   the resulting `num_adults` / `num_children` / `num_rooms` state, including
   the isolation rules (a room-only edit must not touch destination / budget /
   people; a people change must auto-recompute rooms).

==========================================================================
Room policy under test: a room holds 2 adults + 1 child (capacity 3).
  compute_default_rooms = max(1, ceil(adults/2), ceil((adults+children)/3))
Examples: (2,1)->1  (4,0)->2  (6,0)->3  (2,4)->2
==========================================================================

Usage:
    python tests/test_traveler_count.py                 # unit + integration
    python tests/test_traveler_count.py --unit-only     # fast, no LLM/graph
    python tests/test_traveler_count.py --test T3       # one integration case
    python tests/test_traveler_count.py --no-response-checks
"""

import argparse
import os
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.core.graph import build_graph
from agent.shared.travelers import apply_traveler_updates, compute_default_rooms
from config.config import CHOSEN_PROVIDER
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Command

# ---------------------------------------------------------------------------
# ANSI colours
# ---------------------------------------------------------------------------
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
MAGENTA = "\033[95m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

TICK  = f"{GREEN}✔{RESET}"
CROSS = f"{RED}✘{RESET}"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Check:
    description: str
    passed: bool
    detail: str = ""


@dataclass
class TurnSpec:
    message: str
    hitl_response: str = "no"          # answer to any plan_check HITL interrupt

    # --- Traveler / room state checks (None = skip) ---
    expected_num_adults: int | None = None
    expected_num_children: int | None = None
    expected_num_rooms: int | None = None

    # --- Isolation: state keys that must keep a specific value after the turn ---
    expect_unchanged: dict = field(default_factory=dict)

    # --- Response text ---
    response_must_contain: list[str] = field(default_factory=list)

    note: str = ""


@dataclass
class TestCase:
    id: str
    description: str
    turns: list[TurnSpec]


# ---------------------------------------------------------------------------
# Integration test definitions (multi-turn, real graph)
# ---------------------------------------------------------------------------

TESTS: list[TestCase] = [

    TestCase("T1", "Initial request splits people into adults + children", [
        TurnSpec(
            "Plan a trip from Tel Aviv to Paris for 2 adults and 1 child. "
            "Budget $3000, 3 days.",
            expected_num_adults=2,
            expected_num_children=1,
            expected_num_rooms=1,   # max(1, ceil(2/2), ceil(3/3)) = 1
            note="Rooms auto-derived: 2 adults + 1 child fit in a single room.",
        ),
    ]),

    TestCase("T2", "People joining auto-recomputes rooms", [
        TurnSpec(
            "Plan a trip from Tel Aviv to Paris for 2 adults. Budget $3000, 3 days.",
            expected_num_adults=2,
            expected_num_children=0,
            expected_num_rooms=1,
        ),
        TurnSpec(
            "4 more people are joining the trip.",
            expected_num_adults=6,
            expected_num_rooms=3,           # ceil(6/2) = 3
            expect_unchanged={"destination_city": "Paris", "total_budget": 3000,
                              "num_children": 0},
            note="Group grows 2->6 adults; rooms recompute 1->3, nothing else moves.",
        ),
    ]),

    TestCase("T3", "Explicit room change is isolated from everything else", [
        TurnSpec(
            "Plan a trip from Tel Aviv to Paris for 2 adults and 1 child. "
            "Budget $3000, 3 days.",
            expected_num_rooms=1,
        ),
        TurnSpec(
            "Actually, make it 2 rooms.",
            expected_num_rooms=2,
            expect_unchanged={"destination_city": "Paris", "total_budget": 3000,
                              "num_adults": 2, "num_children": 1},
            note="Room-only edit: rooms 1->2; destination/budget/people untouched.",
        ),
    ]),

    TestCase("T4", "Add-a-room delta vs. absolute count", [
        TurnSpec(
            "Plan a trip from Tel Aviv to Paris for 4 adults. Budget $5000, 3 days.",
            expected_num_adults=4,
            expected_num_rooms=2,           # ceil(4/2) = 2
        ),
        TurnSpec(
            "Add another room.",
            expected_num_rooms=3,           # delta +1 over current 2
            expect_unchanged={"num_adults": 4, "num_children": 0},
            note="'Add a room' is a +1 delta, not a reset to a default.",
        ),
    ]),

    TestCase("T5", "People change + explicit rooms in one message: explicit wins", [
        TurnSpec(
            "Plan a trip from Tel Aviv to Paris for 2 adults. Budget $3000, 3 days.",
            expected_num_adults=2,
            expected_num_rooms=1,
        ),
        TurnSpec(
            "2 more adults are joining and please book 3 rooms.",
            expected_num_adults=4,
            expected_num_rooms=3,           # explicit 3 wins over auto-recompute (which would be 2)
            note="Explicit room total overrides the people-driven recompute.",
        ),
    ]),

]


# ---------------------------------------------------------------------------
# Pure unit checks (no LLM)
# ---------------------------------------------------------------------------

def run_unit_checks() -> list[Check]:
    checks: list[Check] = []

    def eq(desc: str, got, want):
        checks.append(Check(desc, got == want, f"got {got!r}, want {want!r}"))

    # compute_default_rooms — 2 adults + 1 child per room (capacity 3)
    eq("rooms(2,1) == 1", compute_default_rooms(2, 1), 1)
    eq("rooms(2,0) == 1", compute_default_rooms(2, 0), 1)
    eq("rooms(4,0) == 2", compute_default_rooms(4, 0), 2)
    eq("rooms(6,0) == 3", compute_default_rooms(6, 0), 3)
    eq("rooms(3,0) == 2", compute_default_rooms(3, 0), 2)   # 3 adults need 2 rooms
    eq("rooms(2,4) == 2", compute_default_rooms(2, 4), 2)   # ceil(6/3)
    eq("rooms(0,0) == 1", compute_default_rooms(0, 0), 1)   # floor of 1 room

    # apply_traveler_updates — people change auto-recomputes rooms
    st = {"num_adults": 2, "num_children": 0, "num_rooms": 1}
    eq("people 2->6 recomputes rooms to 3",
       apply_traveler_updates(st, new_adults=6, new_children=None,
                              new_rooms_abs=None, rooms_delta=None),
       {"num_adults": 6, "num_rooms": 3})

    # room-only absolute change touches nothing else
    st = {"num_adults": 2, "num_children": 1, "num_rooms": 1}
    eq("explicit rooms=2 is isolated",
       apply_traveler_updates(st, new_adults=None, new_children=None,
                              new_rooms_abs=2, rooms_delta=None),
       {"num_rooms": 2})

    # add-a-room delta
    eq("rooms_delta +1 from 1 -> 2",
       apply_traveler_updates({"num_rooms": 1}, new_adults=None, new_children=None,
                              new_rooms_abs=None, rooms_delta=1),
       {"num_rooms": 2})

    # explicit room total wins over people recompute in the same call
    st = {"num_adults": 2, "num_children": 0, "num_rooms": 1}
    eq("explicit rooms wins over recompute",
       apply_traveler_updates(st, new_adults=4, new_children=None,
                              new_rooms_abs=3, rooms_delta=None),
       {"num_adults": 4, "num_rooms": 3})

    # delta floors at 1 room
    eq("rooms_delta floors at 1",
       apply_traveler_updates({"num_rooms": 1}, new_adults=None, new_children=None,
                              new_rooms_abs=None, rooms_delta=-5),
       {"num_rooms": 1})

    # no-op when the value is unchanged
    eq("no change yields empty dict",
       apply_traveler_updates({"num_adults": 2}, new_adults=2, new_children=None,
                              new_rooms_abs=None, rooms_delta=None),
       {})

    # children default to 0 the first time adults are set without a children count
    eq("first adult set seeds children=0",
       apply_traveler_updates({}, new_adults=2, new_children=None,
                              new_rooms_abs=None, rooms_delta=None),
       {"num_adults": 2, "num_children": 0, "num_rooms": 1})

    # children NOT re-defaulted when already set (even to 0)
    eq("children not re-seeded when already 0",
       apply_traveler_updates({"num_adults": 2, "num_children": 0, "num_rooms": 1},
                              new_adults=4, new_children=None,
                              new_rooms_abs=None, rooms_delta=None),
       {"num_adults": 4, "num_rooms": 2})

    return checks


# ---------------------------------------------------------------------------
# Graph factory & turn runner
# ---------------------------------------------------------------------------

def make_graph():
    checkpointer = MemorySaver(serde=JsonPlusSerializer())
    return build_graph(provider=CHOSEN_PROVIDER, checkpointer=checkpointer)


def run_turn(graph, config: dict, message: str, hitl_response: str = "no") -> tuple[dict, str]:
    """Stream one turn to completion, resuming any HITL interrupt."""
    initial_state = {"messages": [("user", message)], "step_count": 0}

    for _ in graph.stream(initial_state, config, stream_mode="values"):
        pass

    while graph.get_state(config).next:
        for _ in graph.stream(Command(resume=hitl_response), config, stream_mode="values"):
            pass

    final_state = graph.get_state(config).values
    response = ""
    for m in reversed(final_state.get("messages", [])):
        if getattr(m, "type", "") == "ai" and hasattr(m, "content"):
            response = str(m.content)
            break
    return final_state, response


def _values_equal(got, want) -> bool:
    if isinstance(got, str) and isinstance(want, str):
        return got.lower() == want.lower()
    try:
        return got == want
    except Exception:
        return False


def check_turn(spec: TurnSpec, state: dict, response: str, skip_response_checks: bool) -> list[Check]:
    checks: list[Check] = []

    if spec.expected_num_adults is not None:
        checks.append(Check(f"num_adults == {spec.expected_num_adults}",
                            state.get("num_adults") == spec.expected_num_adults,
                            f"Actual: {state.get('num_adults')!r}"))

    if spec.expected_num_children is not None:
        checks.append(Check(f"num_children == {spec.expected_num_children}",
                            state.get("num_children") == spec.expected_num_children,
                            f"Actual: {state.get('num_children')!r}"))

    if spec.expected_num_rooms is not None:
        checks.append(Check(f"num_rooms == {spec.expected_num_rooms}",
                            state.get("num_rooms") == spec.expected_num_rooms,
                            f"Actual: {state.get('num_rooms')!r}"))

    for key, want in spec.expect_unchanged.items():
        checks.append(Check(f"{key} unchanged ({want!r})",
                            _values_equal(state.get(key), want),
                            f"Actual: {state.get(key)!r}"))

    if not skip_response_checks:
        response_lower = response.lower()
        for phrase in spec.response_must_contain:
            checks.append(Check(f"Response contains '{phrase}'",
                                phrase.lower() in response_lower,
                                f"(response length: {len(response)} chars)"))

    return checks


def run_test(test: TestCase, skip_response_checks: bool):
    graph = make_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    results = []
    elapsed = 0.0

    for i, turn_spec in enumerate(test.turns):
        t0 = time.time()
        try:
            state, response = run_turn(graph, config, turn_spec.message, turn_spec.hitl_response)
            checks = check_turn(turn_spec, state, response, skip_response_checks)
        except Exception as exc:
            print(f"\n{DIM}{traceback.format_exc()}{RESET}")
            checks = [Check(f"Turn {i + 1} execution crashed", False, str(exc))]
        elapsed += time.time() - t0
        results.append((i + 1, turn_spec, checks))

    all_passed = all(c.passed for _, _, checks in results for c in checks)
    return all_passed, results, elapsed


# ---------------------------------------------------------------------------
# Runner / output
# ---------------------------------------------------------------------------

def _print_checks(turn_results) -> None:
    for turn_idx, turn_spec, checks in turn_results:
        if not any(not c.passed for c in checks):
            continue
        preview = turn_spec.message[:80] + ("…" if len(turn_spec.message) > 80 else "")
        print(f'    Turn {turn_idx}: "{preview}"')
        if turn_spec.note:
            print(f"      {DIM}\U0001f4dd {turn_spec.note}{RESET}")
        for c in checks:
            icon = TICK if c.passed else CROSS
            print(f"      {icon} {c.description}")
            if not c.passed and c.detail:
                print(f"        {DIM}↳ {c.detail}{RESET}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Traveler-count state tests")
    parser.add_argument("--test", nargs="+", help="Run specific integration case IDs (e.g. T2 T3)")
    parser.add_argument("--unit-only", action="store_true", help="Run only the no-LLM unit checks")
    parser.add_argument("--no-response-checks", action="store_true", help="Skip LLM response-text checks")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print(f"\n{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}  \U0001f9f3  Traveler-count + hotel-rooms state tests{RESET}")
    print(f"  Provider:  {CYAN}{CHOSEN_PROVIDER}{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}\n")

    # --- Unit layer (always) ---
    unit_checks = run_unit_checks()
    unit_passed = all(c.passed for c in unit_checks)
    status = f"{GREEN}PASS{RESET}" if unit_passed else f"{RED}FAIL{RESET}"
    print(f"{MAGENTA}[UNIT]{RESET} room rules ({len(unit_checks)} checks)  {status}")
    if not unit_passed or args.verbose:
        for c in unit_checks:
            icon = TICK if c.passed else CROSS
            print(f"    {icon} {c.description}")
            if not c.passed and c.detail:
                print(f"      {DIM}↳ {c.detail}{RESET}")

    if args.unit_only:
        sys.exit(0 if unit_passed else 1)

    # --- Integration layer ---
    tests_to_run = TESTS
    if args.test:
        wanted = {t.upper() for t in args.test}
        tests_to_run = [t for t in tests_to_run if t.id in wanted]

    print()
    results = [unit_passed]
    for i, test in enumerate(tests_to_run, 1):
        print(f"{DIM}[{i}/{len(tests_to_run)}]{RESET} {CYAN}{test.id}{RESET} {test.description}",
              end="  ", flush=True)
        passed, turn_results, elapsed = run_test(test, args.no_response_checks)
        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"{status}  {DIM}({elapsed:.1f}s){RESET}")
        if not passed or args.verbose:
            _print_checks(turn_results)
        results.append(passed)

    passed_count = sum(results)
    total = len(results)
    print(f"\n{BOLD}{'=' * 70}{RESET}")
    color = GREEN if passed_count == total else RED
    print(f"  {BOLD}{color}Total: {passed_count}/{total}{RESET}  {DIM}(incl. unit layer){RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}\n")
    sys.exit(0 if passed_count == total else 1)


if __name__ == "__main__":
    main()
