#!/usr/bin/env python3
"""
Router intent classification test suite.

Tests that the RouterNode correctly classifies user intent and routes to
the correct sub-agent. Stops immediately after the router node fires —
no sub-agents are run, so each test makes exactly ONE cheap LLM call.

Usage:
    python tests/test_router.py                    # run all tests
    python tests/test_router.py --section A B      # specific sections
    python tests/test_router.py --test A1 F3       # specific test IDs
    python tests/test_router.py --verbose          # show details for passing tests too

Intent → Sub-agent mapping:
    new_travel_plan  → TravelAgentNode  (flights/hotels, then ask about itinerary)
    update_travel_plan → AdjustmentsNode (same path, changes existing plan)
    build_itinerary  → ItineraryPlannerNode (full day-by-day schedule)
    advisor          → AdvisorPlannerNode (destination discovery & travel info)
    general_chat     → GeneralChatNode (greetings, visas, etiquette, meta questions)
    out_of_scope     → static rejection (no LLM sub-agent called)
"""

import argparse
import os
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.graph import build_graph
from config.setting import CHOSEN_PROVIDER
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

# ---------------------------------------------------------------------------
# ANSI colours
# ---------------------------------------------------------------------------
GREEN   = "\033[92m"
RED     = "\033[91m"
YELLOW  = "\033[93m"
CYAN    = "\033[96m"
MAGENTA = "\033[95m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RESET   = "\033[0m"
TICK    = f"{GREEN}✔{RESET}"
CROSS   = f"{RED}✘{RESET}"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TurnSpec:
    message: str
    expected_intent: str
    acceptable_intents: list[str] = field(default_factory=list)
    setup_state: dict = field(default_factory=dict)
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
    # SECTION A — new_travel_plan
    # User commits to a specific destination; wants flights / hotels / costs.
    # =========================================================

    TestCase("A1", "A", "Explicit fly-to with destination → new_travel_plan", [
        TurnSpec(
            "I want to fly to Rome from Tel Aviv. Show me my options.",
            expected_intent="new_travel_plan",
            note="Clear commit: specific destination + 'show me options'",
        ),
    ]),

    TestCase("A2", "A", "Book a specific trip → new_travel_plan", [
        TurnSpec(
            "Let's book a trip to Paris for 4 days. Budget $1500, flying from London.",
            expected_intent="new_travel_plan",
            note="'Book' + specific destination + budget + origin = commitment",
        ),
    ]),

    TestCase("A3", "A", "Advisor flow → user commits (TRANSITION RULE) → new_travel_plan", [
        TurnSpec(
            "I like that idea, let's go to Amsterdam. Show me flights and hotels.",
            expected_intent="new_travel_plan",
            setup_state={
                "current_city": "Tel Aviv",
                "intent": "advisor",
            },
            note="TRANSITION RULE: mid-advisor-flow + 'show me flights' = commit",
        ),
    ]),

    # =========================================================
    # SECTION B — build_itinerary
    # User explicitly requests a day-by-day schedule.
    # =========================================================

    TestCase("B1", "B", "Day-by-day itinerary request → build_itinerary", [
        TurnSpec(
            "Build me a day-by-day itinerary for 3 days in Paris from Tel Aviv. Budget $2000.",
            expected_intent="build_itinerary",
            note="'Day-by-day itinerary' is the canonical schedule trigger",
        ),
    ]),

    TestCase("B2", "B", "Plan my daily schedule → build_itinerary", [
        TurnSpec(
            "Plan my daily schedule for Tokyo, 5 days, budget $3000.",
            expected_intent="build_itinerary",
            note="'Daily schedule' = explicit schedule request",
        ),
    ]),

    TestCase("B3", "B", "Build my schedule (active trip context) → build_itinerary", [
        TurnSpec(
            "Build my daily schedule for the trip.",
            expected_intent="build_itinerary",
            setup_state={
                "destination_city": "Paris",
                "current_city": "Tel Aviv",
                "trip_days": 3,
                "total_budget": 2000.0,
            },
            note="'Build my daily schedule' with destination already set → itinerary",
        ),
    ]),

    TestCase("B4", "B", "Replan (trigger word) → build_itinerary", [
        TurnSpec(
            "Replan my itinerary with a budget of $700.",
            expected_intent="build_itinerary",
            setup_state={
                "destination_city": "Paris",
                "current_city": "Tel Aviv",
            },
            note="'Replan' is an explicit guardrail trigger word for build_itinerary",
        ),
    ]),

    # =========================================================
    # SECTION C — advisor
    # Exploring destinations or asking for travel information.
    # =========================================================

    TestCase("C1", "C", "Destination discovery → advisor", [
        TurnSpec(
            "Where should I travel this summer? I like beaches.",
            expected_intent="advisor",
            note="Destination discovery question — always advisor",
        ),
    ]),

    TestCase("C2", "C", "Budget + origin, no committed destination → advisor", [
        TurnSpec(
            "I have $2000 and I'm flying from Tel Aviv. What destinations can I reach?",
            expected_intent="advisor",
            note="Budget + origin without committed destination → advisor (budget tool path)",
        ),
    ]),

    TestCase("C3", "C", "How many days for a city → advisor (NOT build_itinerary)", [
        TurnSpec(
            "How many days should I spend in Amsterdam?",
            expected_intent="advisor",
            note="Duration question = travel info, not a schedule → advisor",
        ),
    ]),

    TestCase("C4", "C", "Best time to visit → advisor", [
        TurnSpec(
            "When is the best time to visit Tokyo?",
            expected_intent="advisor",
            note="Travel info question → advisor",
        ),
    ]),

    TestCase("C5", "C", "City overview → advisor (NOT new_travel_plan)", [
        TurnSpec(
            "Tell me about Paris. What's it like to visit?",
            expected_intent="advisor",
            note="City overview without booking intent → advisor",
        ),
    ]),

    TestCase("C6", "C", "Split time between cities → advisor (NOT build_itinerary)", [
        TurnSpec(
            "How should I split my time between Rome and Florence?",
            expected_intent="advisor",
            note="Duration split = travel advice, not a schedule → advisor per router rule",
        ),
    ]),

    # =========================================================
    # SECTION D — general_chat
    # Greetings, travel logistics/etiquette, meta questions.
    # =========================================================

    TestCase("D1", "D", "Greeting → general_chat", [
        TurnSpec(
            "Hello! What can you do?",
            expected_intent="general_chat",
            note="Pure greeting / capability question",
        ),
    ]),

    TestCase("D2", "D", "Visa logistics → general_chat (NOT advisor)", [
        TurnSpec(
            "Do I need a visa to travel from Israel to France?",
            expected_intent="general_chat",
            note="General logistics not tied to destination discovery → general_chat",
        ),
    ]),

    TestCase("D3", "D", "Tipping etiquette → general_chat", [
        TurnSpec(
            "Is it polite to tip in Japan? What's the custom there?",
            expected_intent="general_chat",
            note="Cultural etiquette = general travel knowledge → general_chat",
        ),
    ]),

    TestCase("D4", "D", "Thank you → general_chat", [
        TurnSpec(
            "Thanks, that was really helpful!",
            expected_intent="general_chat",
            note="Pure conversational response",
        ),
    ]),

    TestCase("D5", "D", "Meta/memory question → general_chat", [
        TurnSpec(
            "What do you remember about me from our conversation?",
            expected_intent="general_chat",
            note="Bot-meta question → general_chat",
        ),
    ]),

    # =========================================================
    # SECTION E — out_of_scope
    # Completely non-travel queries that should be cheaply rejected.
    # =========================================================

    TestCase("E1", "E", "Recipe request → out_of_scope", [
        TurnSpec(
            "Give me a recipe for tahini.",
            expected_intent="out_of_scope",
            note="Food recipe = zero travel relevance",
        ),
    ]),

    TestCase("E2", "E", "Math question → out_of_scope", [
        TurnSpec(
            "What is 2 + 2?",
            expected_intent="out_of_scope",
            note="Math = zero travel relevance",
        ),
    ]),

    TestCase("E3", "E", "Coding request → out_of_scope", [
        TurnSpec(
            "Write me a Python function to sort a list.",
            expected_intent="out_of_scope",
            note="Programming = zero travel relevance",
        ),
    ]),

    TestCase("E4", "E", "General trivia → out_of_scope", [
        TurnSpec(
            "Who is the current president of the United States?",
            expected_intent="out_of_scope",
            note="Political trivia = zero travel relevance",
        ),
    ]),

    # =========================================================
    # SECTION F — Confusable Edge Cases
    # Real misclassifications observed in production runs.
    # =========================================================

    # F-a: advisor vs general_chat — city-specific questions must stay in advisor
    TestCase("F1", "F", "City weather → advisor (NOT general_chat)", [
        TurnSpec(
            "What's the weather like in Paris in December?",
            expected_intent="advisor",
            note="City + season = destination research → advisor tools, NOT general_chat",
        ),
    ]),

    TestCase("F2", "F", "City activities → advisor (NOT general_chat)", [
        TurnSpec(
            "What are the top activities in Amsterdam?",
            expected_intent="advisor",
            note="Specific city activities = advisor tools, NOT general_chat",
        ),
    ]),

    TestCase("F3", "F", "City safety → advisor (NOT general_chat)", [
        TurnSpec(
            "Is Barcelona safe to visit as a solo traveler?",
            expected_intent="advisor",
            note="Safety-in-context-of-visiting = destination info → advisor",
        ),
    ]),

    # F-c: new_travel_plan vs build_itinerary — "plan a trip" ≠ "plan my days"
    TestCase("F6", "F", "'Plan a trip' with specifics → new_travel_plan (NOT build_itinerary)", [
        TurnSpec(
            "Plan a trip to Madrid from Tel Aviv, budget $1000.",
            expected_intent="new_travel_plan",
            note="'Plan a trip' = wants flights/hotels, NOT a day-by-day schedule",
        ),
    ]),

    TestCase("F7", "F", "Vague 'plan my trip' → NOT build_itinerary", [
        TurnSpec(
            "Can you help me plan my trip to Tokyo?",
            expected_intent="new_travel_plan",
            acceptable_intents=["advisor"],
            note="Vague planning intent without 'day-by-day' → new_travel_plan or advisor, NEVER build_itinerary",
        ),
    ]),

    # F-d: build_itinerary vs advisor — "what should I do" = advisor, not itinerary
    TestCase("F8", "F", "'What should I do for N days' → advisor (NOT build_itinerary)", [
        TurnSpec(
            "What should I do in Paris for 3 days?",
            expected_intent="advisor",
            note="Asking for IDEAS, not a schedule → advisor",
        ),
    ]),

    TestCase("F9", "F", "'Give me ideas for N-day trip' → advisor (NOT build_itinerary)", [
        TurnSpec(
            "Give me ideas for a 5-day trip to Rome.",
            expected_intent="advisor",
            note="'Ideas' ≠ 'schedule' → advisor",
        ),
    ]),

    # F-e: advisor vs new_travel_plan — expressing interest ≠ committing
    TestCase("F10", "F", "'I want to go to X' without booking intent → advisor", [
        TurnSpec(
            "I want to go to Tokyo.",
            expected_intent="advisor",
            acceptable_intents=["new_travel_plan"],
            note="Interest without explicit 'show me flights/book' → advisor (or new_travel_plan)",
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
# Intent detector — stops immediately after the router fires
# ---------------------------------------------------------------------------

def detect_intent(graph, message: str, setup_state: dict | None = None) -> str:
    """
    Stream the graph and capture the router's intent classification.
    Breaks as soon as the router update arrives — sub-agents are never run.
    Returns 'other' when the security gate blocks the message before routing.
    """
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    initial_state: dict = {"messages": [("user", message)], "step_count": 0}
    if setup_state:
        initial_state.update(setup_state)

    for update in graph.stream(initial_state, config, stream_mode="updates"):
        if "router" in update:
            return (update["router"] or {}).get("intent", "other")

    return "other"  # security gate blocked before router could fire


# ---------------------------------------------------------------------------
# Check & run
# ---------------------------------------------------------------------------

@dataclass
class Check:
    description: str
    passed: bool
    detail: str = ""


def check_turn(spec: TurnSpec, detected_intent: str) -> list[Check]:
    checks: list[Check] = []
    passed = (
        detected_intent == spec.expected_intent
        or detected_intent in spec.acceptable_intents
    )
    expected_display = spec.expected_intent
    if spec.acceptable_intents:
        expected_display += f" (also ok: {', '.join(spec.acceptable_intents)})"
    checks.append(Check(
        f"Intent is '{expected_display}'",
        passed,
        f"Got '{detected_intent}'",
    ))
    return checks


def run_test(test: TestCase) -> tuple[bool, list[tuple[int, TurnSpec, list[Check], str]], float]:
    graph = make_graph()
    all_turn_results = []
    elapsed = 0.0

    for i, spec in enumerate(test.turns):
        t0 = time.time()
        try:
            detected = detect_intent(graph, spec.message, spec.setup_state or None)
            checks = check_turn(spec, detected)
        except Exception as exc:
            traceback.print_exc()
            detected = "ERROR"
            checks = [Check(f"Turn {i + 1} crashed", False, str(exc))]
        elapsed += time.time() - t0
        all_turn_results.append((i + 1, spec, checks, detected))

    all_passed = all(c.passed for _, _, checks, _ in all_turn_results for c in checks)
    return all_passed, all_turn_results, elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run router intent classification tests")
    parser.add_argument("--section", nargs="+", help="Sections to run (e.g. A B F)")
    parser.add_argument("--test", nargs="+", help="Test IDs to run (e.g. A1 F3)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show pass details too, not just failures")
    args = parser.parse_args()

    tests_to_run = TESTS
    if args.section:
        sections = {s.upper() for s in args.section}
        tests_to_run = [t for t in tests_to_run if t.section in sections]
    if args.test:
        ids = {t.upper() for t in args.test}
        tests_to_run = [t for t in tests_to_run if t.id in ids]

    if not tests_to_run:
        print(f"{RED}No tests matched the filter.{RESET}")
        sys.exit(1)

    total = len(tests_to_run)
    section_names = {
        "A": "new_travel_plan",
        "B": "build_itinerary",
        "C": "advisor",
        "D": "general_chat",
        "E": "out_of_scope",
        "F": "Edge Cases",
    }

    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}  🧭  Router Intent Classification Tests{RESET}")
    print(f"  Provider:  {CYAN}{CHOSEN_PROVIDER}{RESET}")
    print(f"  Tests:     {total}")
    print(f"  Sections:  {', '.join(sorted({t.section for t in tests_to_run}))}")
    print(f"  Mode:      stops after router fires (sub-agents NOT run)")
    print(f"{BOLD}{'='*70}{RESET}\n")

    results = []
    section_scores: dict[str, list[bool]] = {}

    for i, test in enumerate(tests_to_run, 1):
        section_scores.setdefault(test.section, [])
        print(
            f"{DIM}[{i}/{total}]{RESET} "
            f"{MAGENTA}[{test.section}]{RESET} "
            f"{CYAN}{test.id}{RESET} {test.description}",
            end="  ", flush=True,
        )

        passed, turn_results, elapsed = run_test(test)

        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"{status}  {DIM}({elapsed:.1f}s){RESET}")

        section_scores[test.section].append(passed)
        results.append(passed)

        if not passed or args.verbose:
            for turn_idx, spec, checks, detected in turn_results:
                has_failures = any(not c.passed for c in checks)
                if has_failures or args.verbose:
                    msg_preview = spec.message[:80] + ("…" if len(spec.message) > 80 else "")
                    print(f"    Turn {turn_idx}: \"{msg_preview}\"")
                    if spec.note:
                        print(f"      {DIM}📝 {spec.note}{RESET}")
                    if spec.setup_state:
                        print(f"      {DIM}⚙  setup_state: {spec.setup_state}{RESET}")
                    for c in checks:
                        icon = TICK if c.passed else CROSS
                        print(f"      {icon} {c.description}")
                        if not c.passed and c.detail:
                            print(f"        {DIM}↳ {c.detail}{RESET}")

    # --- Summary ---
    passed_count = sum(results)
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}  📊 Results Summary{RESET}\n")

    for section in sorted(section_scores.keys()):
        scores = section_scores[section]
        s_pass = sum(scores)
        s_total = len(scores)
        color = GREEN if s_pass == s_total else (YELLOW if s_pass > 0 else RED)
        name = section_names.get(section, section)
        print(f"  {color}Section {section} ({name}): {s_pass}/{s_total}{RESET}")

    print()
    color = GREEN if passed_count == total else RED
    print(f"  {BOLD}{color}Total: {passed_count}/{total}{RESET}")
    print(f"{BOLD}{'='*70}{RESET}\n")

    sys.exit(0 if passed_count == total else 1)


if __name__ == "__main__":
    main()
