#!/usr/bin/env python3
"""
Automated test runner for the Plan-and-Execute advisor agent.

Each test drives the agent through one or more turns, then checks:
  - Which tools the planner chose (rec_plan state field)
  - Which cities were tracked (rec_shown_cities state field)
  - Response text (contains / does not contain)

Usage:
    python test_rec_agent.py                       # run all tests
    python test_rec_agent.py --section A C         # run sections A and C only
    python test_rec_agent.py --id A1 B2 C4         # run specific tests
    python test_rec_agent.py --no-response-checks  # skip LLM response text checks (faster)
"""

import argparse
import os
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from agent.graph import build_graph
from config.setting import CHOSEN_PROVIDER
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

# ---------------------------------------------------------------------------
# ANSI colours
# ---------------------------------------------------------------------------
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

TICK = f"{GREEN}v{RESET}"
CROSS = f"{RED}x{RESET}"


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
    # --- Tool selection ---
    must_have_tools: list[str]     = field(default_factory=list)  # ALL must appear in plan
    must_have_one_of: list[str]    = field(default_factory=list)  # AT LEAST ONE must appear
    must_not_have_tools: list[str] = field(default_factory=list)  # NONE must appear
    must_have_tool_args: list[dict]= field(default_factory=list)  # [{tool_name, arg_key, arg_value}]
    max_tools: int | None = None
    min_tools: int | None = None
    # --- State tracking ---
    must_show_cities: list[str]    = field(default_factory=list)  # must be in rec_shown_cities
    # --- Response text ---
    response_must_contain: list[str]     = field(default_factory=list)
    response_must_not_contain: list[str] = field(default_factory=list)
    # --- Annotation ---
    note: str = ""


@dataclass
class TestCase:
    id: str
    section: str
    description: str
    turns: list[TurnSpec]


# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------

TESTS: list[TestCase] = [

    # =========================================================
    # SECTION A — Single-turn, single-tool (planner precision)
    # =========================================================

    TestCase("A1", "A", "Best time to visit Tokyo", [
        TurnSpec(
            "When is the best time to visit Tokyo?",
            must_have_one_of=["get_best_time_to_visit", "get_city_overview"],
            must_not_have_tools=["find_destinations_by_tag", "find_destinations_by_vibe",
                                 "fetch_activities", "get_reachable_destinations"],
            max_tools=1,
            must_show_cities=["Tokyo"],
        ),
    ]),

    TestCase("A2", "A", "Trip duration for London", [
        TurnSpec(
            "How many days should I spend in London?",
            must_have_tools=["get_trip_duration_advisor"],
            must_have_tool_args=[{"tool_name": "get_trip_duration_advisor",
                                  "arg_key": "city", "arg_value": "London"}],
            must_not_have_tools=["find_destinations_by_tag", "find_destinations_by_vibe",
                                 "get_reachable_destinations"],
            max_tools=1,
            must_show_cities=["London"],
        ),
    ]),

    TestCase("A3", "A", "Weather in Paris in summer", [
        TurnSpec(
            "What's the weather like in Paris in summer?",
            must_have_one_of=["get_average_weather", "get_city_overview"],
            must_not_have_tools=["find_destinations_by_tag", "find_destinations_by_vibe",
                                 "get_reachable_destinations"],
            max_tools=1,
            must_show_cities=["Paris"],
        ),
    ]),

    TestCase("A4", "A", "Romantic destination — no origin given", [
        TurnSpec(
            "I want a romantic destination — where should I go?",
            must_have_tools=["find_destinations_by_tag"],
            must_have_tool_args=[{"tool_name": "find_destinations_by_tag",
                                  "arg_key": "tag", "arg_value": "romantic"}],
            must_not_have_tools=["get_reachable_destinations"],
            max_tools=1,
        ),
    ]),

    TestCase("A5", "A", "Nature trip — no origin given", [
        TurnSpec(
            "Where should I go for a nature trip?",
            must_have_tools=["find_destinations_by_vibe"],
            must_have_tool_args=[{"tool_name": "find_destinations_by_vibe",
                                  "arg_key": "category", "arg_value": "Nature"}],
            must_not_have_tools=["get_reachable_destinations"],
            max_tools=1,
        ),
    ]),

    TestCase("A6", "A", "Activities for a named city the user is visiting", [
        TurnSpec(
            "I'm going to Tokyo — what activities are there?",
            must_have_tools=["fetch_activities"],
            must_have_tool_args=[{"tool_name": "fetch_activities",
                                  "arg_key": "city", "arg_value": "Tokyo"}],
            must_not_have_tools=["find_destinations_by_tag", "find_destinations_by_vibe",
                                 "get_reachable_destinations"],
            max_tools=1,
            must_show_cities=["Tokyo"],
        ),
    ]),

    TestCase("A7", "A", "Budget + origin, no trip duration stated", [
        TurnSpec(
            "I'm flying from Tel Aviv and my total budget is $800. Where can I go?",
            must_have_tools=["find_destinations_within_budget_auto"],
            must_have_tool_args=[
                {"tool_name": "find_destinations_within_budget_auto",
                 "arg_key": "origin", "arg_value": "Tel Aviv"},
                {"tool_name": "find_destinations_within_budget_auto",
                 "arg_key": "total_budget", "arg_value": 800},
            ],
            must_not_have_tools=["find_destinations_within_budget", "get_reachable_destinations"],
            max_tools=1,
        ),
    ]),

    TestCase("A8", "A", "Budget + origin + explicit 7-day duration", [
        TurnSpec(
            "I'm in Tel Aviv with a budget of $800 for exactly 7 days. Where can I realistically fly?",
            must_have_tools=["find_destinations_within_budget"],
            must_have_tool_args=[
                {"tool_name": "find_destinations_within_budget",
                 "arg_key": "origin", "arg_value": "Tel Aviv"},
                {"tool_name": "find_destinations_within_budget",
                 "arg_key": "trip_days", "arg_value": 7},
            ],
            must_not_have_tools=["find_destinations_within_budget_auto"],
            max_tools=1,
        ),
    ]),

    TestCase("A9", "A", "Short-haul from Tel Aviv — no budget", [
        TurnSpec(
            "I'm flying from Tel Aviv. I hate long flights — where should I go?",
            must_have_tools=["get_reachable_destinations"],
            must_have_tool_args=[{"tool_name": "get_reachable_destinations",
                                  "arg_key": "origin", "arg_value": "Tel Aviv"}],
            must_not_have_tools=["find_destinations_within_budget",
                                 "find_destinations_within_budget_auto"],
            max_tools=1,
        ),
    ]),

    # =========================================================
    # SECTION B — Single-turn, multi-tool (planner strategy)
    # =========================================================

    TestCase("B1", "B", "Full city profile: Berlin", [
        TurnSpec(
            "Tell me everything about Berlin — what kind of city is it, "
            "what can I do there, and when should I visit?",
            must_have_tools=["get_city_overview"],
            must_have_tool_args=[{"tool_name": "get_city_overview",
                                  "arg_key": "city", "arg_value": "Berlin"}],
            must_not_have_tools=["find_destinations_by_tag", "find_destinations_by_vibe",
                                 "get_reachable_destinations"],
            max_tools=2,
            must_show_cities=["Berlin"],
        ),
    ]),

    TestCase("B2", "B", "Origin + vibe, no budget — reachable + tag combined", [
        TurnSpec(
            "I'm flying from Tel Aviv and I want a foodie destination. Where should I go?",
            must_have_one_of=["get_reachable_destinations", "find_destinations_by_tag"],
            must_not_have_tools=["find_destinations_within_budget",
                                 "find_destinations_within_budget_auto"],
            min_tools=2,
            max_tools=2,
        ),
    ]),

    TestCase("B3", "B", "Nightlife — tag and vibe both called", [
        TurnSpec(
            "I want a city with great nightlife — any recommendations?",
            must_have_one_of=["find_destinations_by_tag", "find_destinations_by_vibe"],
            max_tools=2,
        ),
    ]),

    TestCase("B4", "B", "Multi-city duration: Rome and Amsterdam", [
        TurnSpec(
            "I want to visit both Rome and Amsterdam. How should I split my time?",
            must_have_tools=["get_trip_duration_advisor"],
            min_tools=2,
            max_tools=2,
            must_show_cities=["Rome", "Amsterdam"],
        ),
    ]),

    TestCase("B5", "B", "Specific city: weather + best time", [
        TurnSpec(
            "I'm thinking of going to Tokyo. What's the weather in summer and when's the best time to visit?",
            must_have_one_of=["get_city_overview", "get_average_weather", "get_best_time_to_visit"],
            must_not_have_tools=["find_destinations_by_tag", "find_destinations_by_vibe",
                                 "get_reachable_destinations"],
            max_tools=2,
            must_show_cities=["Tokyo"],
        ),
    ]),

    TestCase("B6", "B", "Family trip — vibe tool required, no fetch_activities", [
        TurnSpec(
            "I'm planning a family trip with kids aged 7 and 10. Where should we go?",
            must_have_tools=["find_destinations_by_vibe"],
            must_have_tool_args=[{"tool_name": "find_destinations_by_vibe",
                                  "arg_key": "category", "arg_value": "Family"}],
            must_not_have_tools=["fetch_activities"],
            max_tools=2,
        ),
    ]),

    TestCase("B7", "B", "Budget + vibe preference — only budget tool, no tag tool", [
        TurnSpec(
            "I'm from Tel Aviv, I have $1000, and I want a romantic city. Where can I go?",
            must_have_one_of=["find_destinations_within_budget_auto", "find_destinations_within_budget"],
            must_not_have_tools=["find_destinations_by_tag", "find_destinations_by_vibe"],
            max_tools=1,
            note="Budget tool already filters by reachability — adding a tag tool introduces out-of-budget cities",
        ),
    ]),

    # =========================================================
    # SECTION C — Multi-turn (follow-up detection + state)
    # =========================================================

    TestCase("C1", "C", "Follow-up: detail on a city shown in previous turn", [
        TurnSpec(
            "I want a romantic destination — where should I go?",
            must_have_tools=["find_destinations_by_tag"],
            note="Turn 1: discover romantic destinations",
        ),
        TurnSpec(
            "Tell me more about Paris — when should I visit and how many days?",
            must_have_one_of=["get_city_overview", "get_best_time_to_visit",
                              "get_trip_duration_advisor"],
            must_not_have_tools=["find_destinations_by_tag"],
            must_show_cities=["Paris"],
            note="Turn 2: follow-up — must not re-run the discovery tool",
        ),
    ]),

    TestCase("C2", "C", "Fresh question — no context bleed from previous turn", [
        TurnSpec(
            "What's the weather like in Tokyo in summer?",
            must_have_one_of=["get_average_weather", "get_city_overview"],
            note="Turn 1: Tokyo weather",
        ),
        TurnSpec(
            "I want a family-friendly destination — where should we go?",
            must_have_tools=["find_destinations_by_vibe"],
            must_have_tool_args=[{"tool_name": "find_destinations_by_vibe",
                                  "arg_key": "category", "arg_value": "Family"}],
            must_not_have_tools=["get_average_weather"],
            response_must_not_contain=["Tokyo"],
            note="Turn 2: fresh question — Tokyo must not bleed into plan or response",
        ),
    ]),

    TestCase("C3", "C", "Budget refinement in active rec session", [
        TurnSpec(
            "I'm from Tel Aviv. I want a romantic city — where should I go?",
            must_have_one_of=["get_reachable_destinations", "find_destinations_by_tag"],
            note="Turn 1: establish origin + vibe",
        ),
        TurnSpec(
            "My budget is $900. Which of those could I afford?",
            must_have_one_of=["find_destinations_within_budget_auto", "find_destinations_within_budget"],
            must_not_have_tools=["find_destinations_by_tag"],
            note="Turn 2: budget refinement — origin 'Tel Aviv' must be picked up from state context",
        ),
    ]),

    TestCase("C4", "C", "Avoid re-fetching same city+season on follow-up", [
        TurnSpec(
            "What's the weather like in Amsterdam in summer?",
            must_have_tools=["get_average_weather"],
            must_have_tool_args=[{"tool_name": "get_average_weather",
                                  "arg_key": "season", "arg_value": "Summer"}],
            max_tools=1,
            note="Turn 1: fetch summer weather",
        ),
        TurnSpec(
            "And what about Amsterdam in winter?",
            must_have_tools=["get_average_weather"],
            must_have_tool_args=[{"tool_name": "get_average_weather",
                                  "arg_key": "season", "arg_value": "Winter"}],
            max_tools=1,
            note="Turn 2: fetch winter — must NOT re-fetch summer (max_tools=1 enforces this)",
        ),
    ]),

    # =========================================================
    # SECTION D — Edge cases
    # =========================================================

    TestCase("D2", "D", "Country resolution: 'Israel' -> 'Tel Aviv' in tool args", [
        TurnSpec(
            "I'm from Israel and want short flights only. Where can I go?",
            must_have_tools=["get_reachable_destinations"],
            must_have_tool_args=[{"tool_name": "get_reachable_destinations",
                                  "arg_key": "origin", "arg_value": "Tel Aviv"}],
            note="User says 'Israel' — tool arg must be 'Tel Aviv'",
        ),
    ]),

    TestCase("D3", "D", "No reachability tool when no origin is given", [
        TurnSpec(
            "I want a beach holiday — where should I fly?",
            must_have_one_of=["find_destinations_by_tag"],
            must_not_have_tools=["get_reachable_destinations"],
            max_tools=1,
            response_must_not_contain=["no flights", "unavailable"],
        ),
    ]),

    TestCase("D4", "D", "No-hallucination: only reachability tool for NYC options", [
        TurnSpec(
            "I'm flying from New York. What are my options?",
            must_have_tools=["get_reachable_destinations"],
            must_have_tool_args=[{"tool_name": "get_reachable_destinations",
                                  "arg_key": "origin", "arg_value": "New York"}],
            must_not_have_tools=["find_destinations_by_tag", "find_destinations_by_vibe"],
            max_tools=1,
        ),
    ]),

    TestCase("D5", "D", "Budget stated but no origin — use tag, not budget tool", [
        TurnSpec(
            "I have a $500 budget — where can I go?",
            must_have_one_of=["find_destinations_by_tag", "find_destinations_by_vibe"],
            must_not_have_tools=["find_destinations_within_budget_auto",
                                 "find_destinations_within_budget"],
            note="Budget alone without origin cannot drive a budget tool call",
        ),
    ]),

    TestCase("D6", "D", "rec_shown_cities accumulates across turns", [
        TurnSpec(
            "I want a romantic destination — where should I go?",
            note="Turn 1: cities shown are tracked",
        ),
        TurnSpec(
            "Where should I go for a nature trip?",
            note="Turn 2: cities from both turns must be in rec_shown_cities",
        ),
    ]),

    # =========================================================
    # SECTION E — Replanner decision-making
    # These tests verify the replanner's loop, pruning, and
    # graceful-failure behaviour (the defining feature of the
    # Plan-and-Execute architecture).
    # =========================================================

    TestCase("E1", "E", "Three-step plan — replanner iterates twice before finishing", [
        TurnSpec(
            "I'm visiting Lisbon next summer. What activities are there, "
            "how many days should I stay, and what's the weather in summer?",
            must_have_tools=["fetch_activities", "get_trip_duration_advisor", "get_average_weather"],
            must_have_tool_args=[
                {"tool_name": "fetch_activities",
                 "arg_key": "city", "arg_value": "Lisbon"},
                {"tool_name": "get_trip_duration_advisor",
                 "arg_key": "city", "arg_value": "Lisbon"},
                {"tool_name": "get_average_weather",
                 "arg_key": "season", "arg_value": "Summer"},
            ],
            min_tools=3,
            max_tools=3,
            must_show_cities=["Lisbon"],
            note="Three distinct asks → 3 steps → replanner loops twice with 'Continuing' before finishing",
        ),
    ]),

    TestCase("E2", "E", "Replanner prunes get_best_time when get_city_overview already covers it", [
        TurnSpec(
            "Give me a complete overview of Amsterdam — what it's like, "
            "what to do there, and specifically the best months to visit.",
            must_have_tools=["get_city_overview"],
            must_not_have_tools=["get_best_time_to_visit", "find_destinations_by_tag",
                                 "find_destinations_by_vibe", "get_reachable_destinations"],
            max_tools=2,
            must_show_cities=["Amsterdam"],
            note="city_overview already includes best-visit months — either the planner or replanner "
                 "must suppress a redundant get_best_time_to_visit call",
        ),
    ]),

    TestCase("E3", "E", "Graceful handling when a city does not exist in the database", [
        TurnSpec(
            "I'm planning a trip to Eldorado — what activities are there?",
            must_have_tools=["fetch_activities"],
            must_have_tool_args=[{"tool_name": "fetch_activities",
                                  "arg_key": "city", "arg_value": "Eldorado"}],
            max_tools=2,
            response_must_not_contain=["Error", "Exception", "traceback"],
            note="Non-existent city → empty result → replanner finishes gracefully, "
                 "formatter provides a polite 'no data' response without crashing",
        ),
    ]),

    TestCase("E4", "E", "Replanner continues across two different-city steps and covers both in response", [
        TurnSpec(
            "I want to compare Rome and Barcelona. "
            "Tell me how many days I should spend in each city.",
            must_have_tools=["get_trip_duration_advisor"],
            min_tools=2,
            max_tools=2,
            must_show_cities=["Rome", "Barcelona"],
            note="Two get_trip_duration_advisor calls for different cities — replanner must continue "
                 "after the first and recognise the second is not redundant",
        ),
    ]),

]


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

def make_graph():
    checkpointer = MemorySaver(serde=JsonPlusSerializer())
    return build_graph(provider=CHOSEN_PROVIDER, checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Turn runner
# ---------------------------------------------------------------------------

def run_turn(graph, config: dict, message: str) -> tuple[dict, str]:
    """Stream one turn to completion; return (final_state, formatter_response)."""
    initial_state = {"messages": [("user", message)], "step_count": 0}

    for _ in graph.stream(initial_state, config, stream_mode="values"):
        pass  # consume stream until graph finishes

    final_state = graph.get_state(config).values

    # Formatter response is the last AI message after summary pruning
    messages = final_state.get("messages", [])
    response = ""
    for m in reversed(messages):
        if getattr(m, "type", "") == "ai" and hasattr(m, "content"):
            response = str(m.content)
            break

    return final_state, response


# ---------------------------------------------------------------------------
# Assertion checks
# ---------------------------------------------------------------------------

def _arg_matches(actual, expected) -> bool:
    """Compare tool arg values flexibly (handles numeric strings)."""
    if actual == expected:
        return True
    try:
        return float(actual) == float(expected)
    except (TypeError, ValueError):
        return str(actual).strip().lower() == str(expected).strip().lower()


def check_turn(
    spec: TurnSpec,
    state: dict,
    response: str,
    skip_response_checks: bool = False,
) -> list[Check]:
    checks: list[Check] = []
    # advisor_plan is cleared to [] after execution — check what was actually run instead
    plan = state.get("advisor_last_tool_results") or []
    tools_in_plan = [step.get("tool_name", "") for step in plan]
    shown_cities = [c.lower() for c in (state.get("advisor_shown_cities") or [])]

    # --- must_have_tools: every tool in list must appear ---
    for tool in spec.must_have_tools:
        checks.append(Check(
            f"'{tool}' in plan",
            tool in tools_in_plan,
            f"Plan: {tools_in_plan}",
        ))

    # --- must_have_one_of: at least one must appear ---
    if spec.must_have_one_of:
        found = any(t in tools_in_plan for t in spec.must_have_one_of)
        checks.append(Check(
            f"At least one of {spec.must_have_one_of} in plan",
            found,
            f"Plan: {tools_in_plan}",
        ))

    # --- must_not_have_tools ---
    for tool in spec.must_not_have_tools:
        checks.append(Check(
            f"'{tool}' NOT in plan",
            tool not in tools_in_plan,
            f"Plan: {tools_in_plan}",
        ))

    # --- must_have_tool_args ---
    for arg_check in spec.must_have_tool_args:
        tool_name = arg_check["tool_name"]
        arg_key   = arg_check["arg_key"]
        expected  = arg_check["arg_value"]
        found = False
        detail = f"No '{tool_name}' step in plan"
        for step in plan:
            if step.get("tool_name") == tool_name:
                actual = step.get("args", {}).get(arg_key)
                if _arg_matches(actual, expected):
                    found = True
                    detail = ""
                else:
                    detail = f"{tool_name}.{arg_key} = {actual!r}  (expected {expected!r})"
                break
        checks.append(Check(f"{tool_name}.{arg_key} = {expected!r}", found, detail))

    # --- max_tools ---
    if spec.max_tools is not None:
        n = len(tools_in_plan)
        checks.append(Check(
            f"Tool count <= {spec.max_tools}",
            n <= spec.max_tools,
            f"Actual: {n}  ({tools_in_plan})",
        ))

    # --- min_tools ---
    if spec.min_tools is not None:
        n = len(tools_in_plan)
        checks.append(Check(
            f"Tool count >= {spec.min_tools}",
            n >= spec.min_tools,
            f"Actual: {n}  ({tools_in_plan})",
        ))

    # --- must_show_cities ---
    for city in spec.must_show_cities:
        in_state = city.lower() in shown_cities
        checks.append(Check(
            f"'{city}' in advisor_shown_cities",
            in_state,
            f"advisor_shown_cities: {state.get('advisor_shown_cities', [])}",
        ))

    if skip_response_checks:
        return checks

    # --- response_must_contain ---
    response_lower = response.lower()
    for phrase in spec.response_must_contain:
        checks.append(Check(
            f"Response contains '{phrase}'",
            phrase.lower() in response_lower,
            "Phrase not found",
        ))

    # --- response_must_not_contain ---
    for phrase in spec.response_must_not_contain:
        checks.append(Check(
            f"Response does NOT contain '{phrase}'",
            phrase.lower() not in response_lower,
            f"Unwanted phrase found in response",
        ))

    return checks


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_test(
    test: TestCase,
    skip_response_checks: bool = False,
) -> tuple[bool, list[tuple[int, str, list[Check]]], float]:
    """
    Run all turns for one test case.
    Returns (all_passed, [(turn_idx, turn_message, checks)], elapsed_seconds).
    """
    graph = make_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    all_turn_results: list[tuple[int, str, list[Check]]] = []
    elapsed = 0.0

    for i, turn_spec in enumerate(test.turns):
        t0 = time.time()
        try:
            state, response = run_turn(graph, config, turn_spec.message)
            checks = check_turn(turn_spec, state, response, skip_response_checks)
        except Exception as exc:
            tb = traceback.format_exc()
            checks = [Check(f"Turn {i + 1} execution", False, str(exc))]
            print(f"\n{DIM}{tb}{RESET}")
        elapsed += time.time() - t0
        all_turn_results.append((i + 1, turn_spec.message, checks))

    all_passed = all(c.passed for _, _, checks in all_turn_results for c in checks)
    return all_passed, all_turn_results, elapsed


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_test_header(test: TestCase, index: int, total: int) -> None:
    print(f"\n{DIM}[{index}/{total}]{RESET} {CYAN}{test.id}{RESET} {test.description}", end="  ", flush=True)


def print_test_result(
    test: TestCase,
    passed: bool,
    turn_results: list[tuple[int, str, list[Check]]],
    elapsed: float,
) -> None:
    status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
    print(f"{status}  {DIM}({elapsed:.1f}s){RESET}")

    if not passed:
        for turn_idx, message, checks in turn_results:
            failed = [c for c in checks if not c.passed]
            if not failed:
                continue
            label = f"  Turn {turn_idx}: \"{message[:60]}{'…' if len(message) > 60 else ''}\""
            print(f"{label}")
            for c in failed:
                print(f"    {CROSS} {c.description}")
                if c.detail:
                    print(f"       {DIM}{c.detail}{RESET}")


def print_summary(
    results: list[tuple[TestCase, bool, list, float]],
) -> None:
    total   = len(results)
    passed  = sum(1 for _, p, _, _ in results if p)
    failed  = total - passed
    elapsed = sum(e for _, _, _, e in results)

    print(f"\n{'=' * 62}")
    print(f"{BOLD}SUMMARY{RESET}")
    print(f"{'=' * 62}")

    # Per-section breakdown
    sections: dict[str, dict] = {}
    for test, p, _, _ in results:
        s = sections.setdefault(test.section, {"passed": 0, "total": 0, "failed_ids": []})
        s["total"] += 1
        if p:
            s["passed"] += 1
        else:
            s["failed_ids"].append(test.id)

    for sec in sorted(sections):
        s = sections[sec]
        color = GREEN if s["passed"] == s["total"] else (YELLOW if s["passed"] > 0 else RED)
        bar = f"{color}{s['passed']}/{s['total']}{RESET}"
        fails = f"  {DIM}(failed: {', '.join(s['failed_ids'])}){RESET}" if s["failed_ids"] else ""
        print(f"  Section {BOLD}{sec}{RESET}: {bar}{fails}")

    # Failed tests list
    if failed:
        print(f"\n{RED}Failed tests:{RESET}")
        for test, p, _, _ in results:
            if not p:
                print(f"  {RED}{test.id}{RESET}  {test.description}")

    # Score
    pct = int(passed / total * 100) if total else 0
    color = GREEN if pct >= 80 else (YELLOW if pct >= 50 else RED)
    print(f"\n{BOLD}Score: {color}{passed}/{total}  ({pct}%){RESET}   {DIM}total time: {elapsed:.1f}s{RESET}")
    print(f"{'=' * 62}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run advisor agent tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--section", nargs="+", metavar="S",
        help="Run only these sections (e.g. --section A C D)",
    )
    parser.add_argument(
        "--id", nargs="+", metavar="ID",
        help="Run only these test IDs (e.g. --id A1 B2 C4)",
    )
    parser.add_argument(
        "--no-response-checks", action="store_true",
        help="Skip response-text assertions (faster; still checks plan and state)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Filter test list
    tests_to_run = TESTS
    if args.section:
        sections = {s.upper() for s in args.section}
        tests_to_run = [t for t in tests_to_run if t.section in sections]
    if args.id:
        ids = {i.upper() for i in args.id}
        tests_to_run = [t for t in tests_to_run if t.id.upper() in ids]

    if not tests_to_run:
        print("No tests matched the given filters.")
        sys.exit(1)

    skip_resp = args.no_response_checks
    total = len(tests_to_run)

    print(f"\n{BOLD}Advisor Agent Test Runner{RESET}")
    print(f"Provider: {CYAN}{CHOSEN_PROVIDER}{RESET}  |  Tests: {total}"
          + (f"  |  {YELLOW}response checks skipped{RESET}" if skip_resp else ""))

    all_results: list[tuple[TestCase, bool, list, float]] = []

    for i, test in enumerate(tests_to_run, 1):
        print_test_header(test, i, total)
        passed, turn_results, elapsed = run_test(test, skip_response_checks=skip_resp)
        print_test_result(test, passed, turn_results, elapsed)
        all_results.append((test, passed, turn_results, elapsed))

    print_summary(all_results)


if __name__ == "__main__":
    main()
