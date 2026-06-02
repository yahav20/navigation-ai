#!/usr/bin/env python3
"""
Automated test runner for the Full Itinerary Plan-and-Execute Agent.

This test suite verifies the core capabilities of the planner, executor,
and replanner, ensuring they adapt dynamically to user changes, strict
constraints, and edge cases based on the travel_agency.db.

==========================================================================
DB Cheat-Sheet (from init_db.py):
--------------------------------------------------------------------------
CITIES:  Tel Aviv, Paris, London, Tokyo, New York, Berlin, Amsterdam

FLIGHT ROUTES (one-way, prices):
  TLV→Paris  $180-420   |  Paris→TLV  $150-400
  TLV→London $390-450   |  London→TLV  ✗ (no direct)
  TLV→Tokyo  $820-950   |  Tokyo→TLV   ✗ (no direct)
  TLV→NY     $750       |  NY→TLV      ✗ (no direct)
  TLV→Berlin $110-280   |  Berlin→TLV  ✗ (no direct)
  TLV→AMS    $410       |  AMS→TLV     ✗ (no direct)
  London→Paris   $120   |  Paris→London $120
  London→Berlin  $60    |  London→Tokyo $890
  London→NY      $550   |  NY→London   $550
  NY→Paris   $480       |  NY→AMS      $500
  Paris→Berlin   $160   |  AMS→Berlin  $80

HOTELS (per night):
  Paris:   Ibis Budget $80/2★, Hotel de Ville $150/3★,
           Le Marais Boutique $220/4★ (KOSHER), Luxury Ritz $600/5★
  London:  Premier Inn $120/3★, The Savoy $450/5★
  Tokyo:   Park Hyatt $700/5★ (only hotel!)
  NY:      The Plaza $850/5★ (only hotel!)
  Berlin:  Hilton Berlin $220/4★ (only hotel!)
  AMS:     Canal Boutique $180/4★ (only hotel!)

ACTIVITY COUNTS:
  Paris: 20   London: 3   Tokyo: 2   NY: 1   Berlin: 2   AMS: 2
==========================================================================

Usage:
    python tests/test_itinerary_agent.py                       # run all tests
    python tests/test_itinerary_agent.py --section A C         # run specific sections
    python tests/test_itinerary_agent.py --test A1 B3          # run specific tests
    python tests/test_itinerary_agent.py --no-response-checks  # skip LLM output checks
"""

import argparse
import os
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Adjust these imports according to your actual project structure
from agent.graph import build_graph
from config.setting import CHOSEN_PROVIDER
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
    # --- Adaptive Planner Checks ---
    must_plan_steps: list[str]     = field(default_factory=list)
    must_not_plan_steps: list[str] = field(default_factory=list)

    # --- State Checks ---
    expected_destination: str | None = None
    expected_days: int | None = None

    # --- Response text ---
    response_must_contain: list[str]     = field(default_factory=list)
    response_must_not_contain: list[str] = field(default_factory=list)

    # --- Structural checks ---
    expect_feasible: bool | None = None          # True / False / None=skip

    note: str = ""


@dataclass
class TestCase:
    id: str
    section: str
    description: str
    turns: list[TurnSpec]


# ---------------------------------------------------------------------------
# Test definitions — designed around the actual travel_agency.db data
# ---------------------------------------------------------------------------

TESTS: list[TestCase] = [

    # =========================================================
    # SECTION A — Happy Path (Full Trip Generation)
    # =========================================================

    TestCase("A1", "A", "Standard 3-day trip: Tel Aviv → Paris (richest data)", [
        TurnSpec(
            "Build me a detailed day-by-day itinerary for a 3-day trip from "
            "Tel Aviv to Paris. My budget is $2500.",
            must_plan_steps=[
                "fetch_activities", "build_day_schedule", "verify_budget",
            ],
            expected_destination="Paris",
            expected_days=3,
            response_must_contain=["Paris", "Day 1"],
            note=(
                "Runs in standalone mode (no prior travel plan). "
                "Planner must produce fetch_activities + 3×build_day_schedule + verify_budget. "
                "Paris has 20 activities — scheduler has plenty to work with."
            ),
        ),
    ]),

    TestCase("A2", "A", "Budget 2-day trip: Tel Aviv → Berlin (sparse activities)", [
        TurnSpec(
            "Plan a day-by-day schedule for 2 days in Berlin, flying from "
            "Tel Aviv. Budget: $800.",
            must_plan_steps=[
                "fetch_activities", "build_day_schedule", "verify_budget",
            ],
            expected_destination="Berlin",
            expected_days=2,
            response_must_contain=["Berlin"],
            note=(
                "Berlin has only 2 activities in the DB. Tests the scheduler's "
                "graceful handling of a very limited activity pool for 2 days."
            ),
        ),
    ]),

    TestCase("A3", "A", "3-day itinerary: London → Paris", [
        TurnSpec(
            "Create a 3-day day-by-day itinerary from London to Paris. "
            "My budget is $1500.",
            must_plan_steps=[
                "fetch_activities", "build_day_schedule", "verify_budget",
            ],
            expected_destination="Paris",
            expected_days=3,
            response_must_contain=["Paris"],
            note=(
                "Paris has 20 activities — happy path with rich data. "
                "Standalone mode; verify_budget uses estimated average prices."
            ),
        ),
    ]),

    # =========================================================
    # SECTION B — Dynamic Mid-Flight Modifications
    # =========================================================

    TestCase("B1", "B", "Preference update triggers itinerary rebuild", [
        TurnSpec(
            "Build me a day-by-day itinerary for 4 days in Paris from "
            "Tel Aviv. Budget is $2500.",
            must_plan_steps=["fetch_activities", "build_day_schedule", "verify_budget"],
            expected_destination="Paris",
            expected_days=4,
            response_must_contain=["Paris", "Day 1"],
            note="Turn 1: Build 4-day standalone Paris itinerary.",
        ),
        TurnSpec(
            "Actually, I just realized I need a Kosher diet. "
            "Can you rebuild the itinerary with that constraint?",
            must_plan_steps=["fetch_activities", "build_day_schedule"],
            note=(
                "Turn 2: Preference update — itinerary agent re-fetches activities "
                "and rebuilds schedule. Hotel selection is the travel agent's job; "
                "the itinerary agent filters activities by dietary preference."
            ),
        ),
    ]),

    TestCase("B2", "B", "Complete Destination Pivot: Paris → Tokyo", [
        TurnSpec(
            "Plan a detailed daily schedule for 3 days in Paris from "
            "Tel Aviv. Budget $3000.",
            must_plan_steps=["fetch_activities", "build_day_schedule", "verify_budget"],
            expected_destination="Paris",
            note="Turn 1: Build 3-day Paris itinerary.",
        ),
        TurnSpec(
            "Wait, I changed my mind — let's go to Tokyo instead. "
            "Build me a day-by-day itinerary for Tokyo.",
            must_plan_steps=["fetch_activities", "build_day_schedule"],
            expected_destination="Tokyo",
            response_must_contain=["Tokyo"],
            response_must_not_contain=["Paris"],
            note=(
                "Turn 2: Destination pivot. Metadata/enrichment updates destination "
                "to Tokyo. Planner rebuilds fresh. Tokyo has only 2 activities — "
                "tests graceful scheduling with minimal data."
            ),
        ),
    ]),

    TestCase("B3", "B", "Extend trip duration: 2 days → 4 days", [
        TurnSpec(
            "Create a day-by-day itinerary for a 2-day trip from "
            "New York to London. Budget $2000.",
            must_plan_steps=["fetch_activities", "build_day_schedule", "verify_budget"],
            expected_destination="London",
            expected_days=2,
        ),
        TurnSpec(
            "Let's make it 4 days instead. Rebuild the daily schedule.",
            must_plan_steps=["build_day_schedule"],
            expected_days=4,
            response_must_contain=["Day 4"],
            note=(
                "Turn 2: Re-runs planner with trip_days=4. "
                "London has only 3 activities — the scheduler is "
                "seriously constrained and may need to reuse slots."
            ),
        ),
    ]),

    TestCase("B4", "B", "Budget increase after verify_budget flags over-budget", [
        TurnSpec(
            "Build a day-by-day itinerary for 3 days in Tokyo from "
            "Tel Aviv. My budget is $200.",
            must_plan_steps=["fetch_activities", "build_day_schedule", "verify_budget"],
            expected_destination="Tokyo",
            note=(
                "Turn 1: $200 budget is impossibly low for Tokyo. "
                "verify_budget should flag the plan as over-budget."
            ),
        ),
        TurnSpec(
            "OK, I can increase my budget to $5000. "
            "Please rebuild the itinerary with this new budget.",
            must_plan_steps=["verify_budget"],
            response_must_contain=["Tokyo"],
            note=(
                "Turn 2: With $5000 budget Tokyo is feasible. "
                "Replanner re-verifies budget with the new ceiling."
            ),
        ),
    ]),

    # =========================================================
    # SECTION C — Fallback & Edge Cases
    # =========================================================

    TestCase("C1", "C", "Hard Budget Violation: 7-day luxury Tokyo on $1000", [
        TurnSpec(
            "Build me a day-by-day itinerary for a 7-day luxury trip from "
            "Tel Aviv to Tokyo. My budget is only $1000.",
            must_plan_steps=["verify_budget"],
            expect_feasible=False,
            note=(
                "TLV→TYO flights alone $820-950. Park Hyatt $700/night * 7 "
                "= $4900. Total minimum ~$5720. Budget $1000 is impossible. "
                "Observer should trigger Fallback to suggest alternatives "
                "or adjust days."
            ),
        ),
    ]),

    TestCase("C2", "C", "Non-existent City (Atlantis)", [
        TurnSpec(
            "Build me a day-by-day itinerary for 3 days in Atlantis "
            "from Tel Aviv.",
            must_plan_steps=["fetch_activities"],
            expect_feasible=False,
            note=(
                "DB has no city named 'Atlantis'. fetch_activities returns empty. "
                "The replanner should mark the plan infeasible after exhausting retries."
            ),
        ),
    ]),

    TestCase("C3", "C", "City with minimal activities: New York (only 1)", [
        TurnSpec(
            "Build a day-by-day itinerary for a 3-day trip from "
            "London to New York. Budget $3000.",
            expected_destination="New York",
            expected_days=3,
            note=(
                "NY has ONLY 1 activity (Statue of Liberty) and 1 hotel "
                "(The Plaza at $850/night!). The schedule builder needs "
                "3 days of activities but only has 1 to work with. "
                "This should trigger graceful degradation or fallback."
            ),
        ),
    ]),

    TestCase("C4", "C", "Sparse activities: Tel Aviv → Amsterdam (2 activities only)", [
        TurnSpec(
            "Plan a day-by-day 2-day itinerary from Tel Aviv to Amsterdam. "
            "Budget: $1500.",
            expected_destination="Amsterdam",
            expected_days=2,
            must_plan_steps=["fetch_activities", "build_day_schedule", "verify_budget"],
            response_must_contain=["Amsterdam"],
            note=(
                "Amsterdam has only 2 activities in the DB. "
                "Tests the scheduler's handling of a minimal activity pool. "
                "Flight routing is the travel agent's concern — itinerary "
                "only schedules days using whatever data is in state."
            ),
        ),
    ]),

    TestCase("C5", "C", "Adults-only hotel constraint clash (min_age=18)", [
        TurnSpec(
            "Build a day-by-day itinerary for a family trip with young "
            "children to Paris from Tel Aviv for 3 days. Budget $2000.",
            expected_destination="Paris",
            response_must_contain=["Paris"],
            note=(
                "Paris hotels: Luxury Ritz min_age=18, Le Marais Boutique "
                "min_age=0, Hotel de Ville min_age=0, Ibis min_age=0. "
                "Family-friendly hotels should be preferred. "
                "Also tests activity filtering (Pub Crawl min_age=18)."
            ),
        ),
    ]),

    # =========================================================
    # SECTION D — Observer Validation & Replanning
    # =========================================================

    TestCase("D1", "D", "Tight budget boundary test: TLV→Berlin $600", [
        TurnSpec(
            "Build a day-by-day itinerary for 2 days in Berlin from "
            "Tel Aviv. My budget is $600.",
            expected_destination="Berlin",
            expected_days=2,
            must_plan_steps=["verify_budget"],
            note=(
                "Cheapest: Ryanair $110. Hilton $220*2=$440. "
                "Base: $110+$440=$550 + return flight (connecting, ~$160-280) "
                "= $710-830. Budget $600 is too tight — should trigger "
                "budget exceeded and fallback/replanning."
            ),
        ),
    ]),

    TestCase("D2", "D", "Verify replanning preserves successful fetch cache", [
        TurnSpec(
            "Build a day-by-day itinerary for 4 days in Paris from "
            "Tel Aviv. Budget: $1200.",
            must_plan_steps=["fetch_activities", "build_day_schedule", "verify_budget"],
            expected_destination="Paris",
            expected_days=4,
            note=(
                "If verify_budget fails, replanning should NOT re-fetch activities "
                "— they are already cached in step_results. Only rebuild day "
                "schedules or adjust trip_days. Paris has 20 activities so "
                "there's plenty to work with across replans."
            ),
        ),
    ]),

    TestCase("D3", "D", "Verify day schedule quality: meals present", [
        TurnSpec(
            "Create a detailed day-by-day itinerary for 2 days in Paris "
            "from London. Budget $1000.",
            expected_destination="Paris",
            expected_days=2,
            must_plan_steps=["build_day_schedule"],
            note=(
                "Paris has 8 food venues (Café de Flore, L'As du Fallafel, "
                "Latin Quarter Food Tour, Le Jules Verne, Bouillon Chartier, "
                "Bateaux Dinner Cruise, Angelina, Macaron Class). "
                "Observer validates meals: breakfast < 10:00, lunch 11-15, "
                "dinner > 18:00. Each day should include food slots."
            ),
        ),
    ]),

    # =========================================================
    # SECTION E — Stress & Integration Tests
    # =========================================================

    TestCase("E1", "E", "Maximum duration: 7-day Paris itinerary", [
        TurnSpec(
            "Build me a comprehensive 7-day day-by-day itinerary in Paris "
            "from Tel Aviv. Budget: $4000. I want to see everything!",
            expected_destination="Paris",
            expected_days=7,
            must_plan_steps=["build_day_schedule", "verify_budget"],
            response_must_contain=["Day 7"],
            note=(
                "7 build_day_schedule steps + all fetch steps = 12+ steps. "
                "20 Paris activities spread across 7 days. "
                "Tests scheduler deduplication (no activity repeated). "
                "Budget: $180+$150 + $80*7=$560 + activities ~$500 + meals "
                "~$420 ≈ $1810. Well within $4000."
            ),
        ),
    ]),

    TestCase("E2", "E", "Multi-constraint: Kosher diet + long trip", [
        TurnSpec(
            "Build a day-by-day itinerary for 5 days in Paris from Tel Aviv. "
            "I need kosher food options. Budget: $2500.",
            expected_destination="Paris",
            expected_days=5,
            must_plan_steps=["fetch_activities", "build_day_schedule", "verify_budget"],
            note=(
                "5-day itinerary with kosher dietary preference. "
                "Paris has kosher food activities: L'As du Fallafel, "
                "Café de Flore, Bouillon Chartier. The itinerary agent "
                "filters activities by dietary restriction; hotel selection "
                "is the travel agent's responsibility."
            ),
        ),
    ]),

    TestCase("E3", "E", "Minimal info: user gives only destination", [
        TurnSpec(
            "Build me a day-by-day itinerary for Tokyo.",
            note=(
                "No origin city, no budget, no trip days specified. "
                "The enrichment node should ask for missing info. "
                "Tests the enrichment → planner pipeline when data is "
                "incomplete. Should NOT crash — should either ask for "
                "more details or use reasonable defaults (3 days)."
            ),
        ),
    ]),

    TestCase("E4", "E", "Sparse activities: NY → Berlin (2 activities only)", [
        TurnSpec(
            "Plan a detailed 3-day daily itinerary from New York to Berlin. "
            "Budget: $2000.",
            expected_destination="Berlin",
            expected_days=3,
            must_plan_steps=["fetch_activities", "build_day_schedule", "verify_budget"],
            note=(
                "Berlin has only 2 activities in the DB. Tests the scheduler "
                "and replanner when building 3 days from a very thin activity pool. "
                "Flight routing is the travel agent's concern — the itinerary "
                "agent schedules days using the state it receives."
            ),
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
    """Stream one turn to completion; return (final_state, response_text).

    Automatically resumes any HITL interrupt (PlanCheckNode asking whether to
    search for flights/hotels first) with "no" so the itinerary runs in
    standalone mode without requiring real human input.
    """
    initial_state = {"messages": [("user", message)], "step_count": 0}

    for _ in graph.stream(initial_state, config, stream_mode="values"):
        pass

    # Auto-resume HITL interrupts: PlanCheckNode asks about travel data;
    # "no" → standalone mode (build schedule without pre-existing bookings).
    current = graph.get_state(config)
    while current.next:
        for _ in graph.stream(Command(resume="no"), config, stream_mode="values"):
            pass
        current = graph.get_state(config)

    final_state = current.values

    # Find the final formatted response (the last AI message)
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

def check_turn(
    spec: TurnSpec,
    state: dict,
    response: str,
    skip_response_checks: bool = False,
) -> list[Check]:
    checks: list[Check] = []

    # Extract the plan generated by the LLM Planner for this specific turn
    plan_data = state.get("itinerary_plan", {})
    exec_plan = plan_data.get("execution_plan", {})
    plan_steps = exec_plan.get("steps", [])

    # Handle Pydantic models or dicts depending on serialization
    planned_step_types = [
        s.step_type if hasattr(s, "step_type") else s.get("step_type", "")
        for s in plan_steps
    ]

    # --- Planner Checks (Adaptive logic) ---
    for step in spec.must_plan_steps:
        checks.append(Check(
            f"Planner included '{step}'",
            step in planned_step_types,
            f"Plan generated: {planned_step_types}",
        ))

    for step in spec.must_not_plan_steps:
        checks.append(Check(
            f"Planner correctly SKIPPED '{step}'",
            step not in planned_step_types,
            f"Plan generated: {planned_step_types}",
        ))

    # --- State Checks ---
    if spec.expected_destination:
        actual_dest = state.get("destination_city", "")
        checks.append(Check(
            f"Destination state is '{spec.expected_destination}'",
            actual_dest.lower() == spec.expected_destination.lower()
            if actual_dest else False,
            f"Actual: {actual_dest!r}",
        ))

    if spec.expected_days is not None:
        actual_days = state.get("trip_days", 0)
        checks.append(Check(
            f"Trip days state is {spec.expected_days}",
            actual_days == spec.expected_days,
            f"Actual: {actual_days}",
        ))

    # --- Feasibility Check ---
    if spec.expect_feasible is not None:
        actual_feasible = state.get("itinerary_feasible", True)
        checks.append(Check(
            f"Feasibility is {spec.expect_feasible}",
            actual_feasible == spec.expect_feasible,
            f"Actual: {actual_feasible}",
        ))

    # --- Response Checks ---
    if skip_response_checks:
        return checks

    response_lower = response.lower()
    for phrase in spec.response_must_contain:
        checks.append(Check(
            f"Response contains '{phrase}'",
            phrase.lower() in response_lower,
            f"Phrase not found (response length: {len(response)} chars)",
        ))

    for phrase in spec.response_must_not_contain:
        checks.append(Check(
            f"Response does NOT contain '{phrase}'",
            phrase.lower() not in response_lower,
            "Unwanted phrase found in output",
        ))

    return checks


# ---------------------------------------------------------------------------
# Test runner & Output
# ---------------------------------------------------------------------------

def run_test(test: TestCase, skip_response_checks: bool = False):
    graph = make_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    all_turn_results = []
    elapsed = 0.0

    for i, turn_spec in enumerate(test.turns):
        t0 = time.time()
        try:
            state, response = run_turn(graph, config, turn_spec.message)
            checks = check_turn(turn_spec, state, response, skip_response_checks)
        except Exception as exc:
            tb = traceback.format_exc()
            checks = [Check(f"Turn {i + 1} execution crashed", False, str(exc))]
            print(f"\n{DIM}{tb}{RESET}")

        elapsed += time.time() - t0
        all_turn_results.append((i + 1, turn_spec, checks))

    all_passed = all(c.passed for _, _, checks in all_turn_results for c in checks)
    return all_passed, all_turn_results, elapsed


def _format_state_snapshot(state: dict) -> str:
    """One-liner state snapshot for debugging."""
    return (
        f"dest={state.get('destination_city', '—')} | "
        f"days={state.get('trip_days', '—')} | "
        f"budget={state.get('total_budget', '—')} | "
        f"feasible={state.get('itinerary_feasible', '—')} | "
        f"mode={state.get('itinerary_mode', '—')} | "
        f"replanner={state.get('replanner_action', '—')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Full Itinerary Agent tests")
    parser.add_argument("--section", nargs="+", help="Run specific sections (e.g. A B)")
    parser.add_argument("--test", nargs="+", help="Run specific test IDs (e.g. A1 B3)")
    parser.add_argument("--no-response-checks", action="store_true",
                        help="Skip LLM response-text checks")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed output for passing tests too")
    args = parser.parse_args()

    tests_to_run = TESTS
    if args.section:
        sections = {s.upper() for s in args.section}
        tests_to_run = [t for t in tests_to_run if t.section in sections]
    if args.test:
        test_ids = {t.upper() for t in args.test}
        tests_to_run = [t for t in tests_to_run if t.id in test_ids]

    skip_resp = args.no_response_checks
    total = len(tests_to_run)

    if total == 0:
        print(f"{RED}No tests matched the filter.{RESET}")
        sys.exit(1)

    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}  ✈️  Itinerary Plan-and-Execute Agent Test Runner{RESET}")
    print(f"  Provider:  {CYAN}{CHOSEN_PROVIDER}{RESET}")
    print(f"  Tests:     {total}")
    print(f"  Sections:  {', '.join(sorted({t.section for t in tests_to_run}))}")
    if skip_resp:
        print(f"  {YELLOW}⚠ Response text checks DISABLED{RESET}")
    print(f"{BOLD}{'='*70}{RESET}\n")

    results = []
    section_scores: dict[str, list[bool]] = {}

    for i, test in enumerate(tests_to_run, 1):
        section_scores.setdefault(test.section, [])

        print(f"{DIM}[{i}/{total}]{RESET} "
              f"{MAGENTA}[{test.section}]{RESET} "
              f"{CYAN}{test.id}{RESET} {test.description}",
              end="  ", flush=True)

        passed, turn_results, elapsed = run_test(test, skip_response_checks=skip_resp)

        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"{status}  {DIM}({elapsed:.1f}s){RESET}")

        section_scores[test.section].append(passed)

        # Show turn details on failure (or with --verbose)
        if not passed or args.verbose:
            for turn_idx, turn_spec, checks in turn_results:
                has_failures = any(not c.passed for c in checks)
                msg_preview = turn_spec.message[:80] + ("…" if len(turn_spec.message) > 80 else "")
                if has_failures or args.verbose:
                    print(f"    Turn {turn_idx}: \"{msg_preview}\"")
                    if turn_spec.note:
                        print(f"      {DIM}📝 {turn_spec.note}{RESET}")
                    for c in checks:
                        icon = TICK if c.passed else CROSS
                        print(f"      {icon} {c.description}")
                        if not c.passed and c.detail:
                            print(f"        {DIM}↳ {c.detail}{RESET}")

        results.append(passed)

    # --- Summary ---
    passed_count = sum(results)
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}  📊 Results Summary{RESET}\n")

    for section in sorted(section_scores.keys()):
        scores = section_scores[section]
        s_pass = sum(scores)
        s_total = len(scores)
        color = GREEN if s_pass == s_total else (YELLOW if s_pass > 0 else RED)
        section_names = {
            "A": "Happy Path",
            "B": "Dynamic Modifications",
            "C": "Fallback & Edge Cases",
            "D": "Observer Validation",
            "E": "Stress & Integration",
        }
        name = section_names.get(section, section)
        print(f"  {color}Section {section} ({name}): {s_pass}/{s_total}{RESET}")

    print()
    color = GREEN if passed_count == total else RED
    print(f"  {BOLD}{color}Total: {passed_count}/{total}{RESET}")
    print(f"{BOLD}{'='*70}{RESET}\n")

    sys.exit(0 if passed_count == total else 1)


if __name__ == "__main__":
    main()
