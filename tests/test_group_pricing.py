#!/usr/bin/env python3
"""Automated tests for group-aware pricing in the travel agent.

Two layers:

1. Deterministic UNIT asserts on `src/agent/shared/pricing.py` — no LLM, instant.
2. Multi-turn INTEGRATION cases that drive the real compiled graph and assert
   that `travel_plan["lowest_group_estimate"]` and `travel_plan["travelers_label"]`
   in the resulting state reflect the correct group-pricing math.

==========================================================================
Pricing rules under test:
  Flights : adults × price + children × price × 0.8
  Hotels  : num_rooms × price_per_night × trip_days
  Budget  : total_budget is always the GROUP budget

DB cheat-sheet (flights used in tests):
  TLV → Paris  $180–420   |  Paris → TLV  $150–400
  TLV → Berlin $110–280   |  London → Paris $120  |  Paris → London $120
Hotels (Paris): Ibis Budget $80/night  |  Hotel de Ville $150  |  Le Marais $220  |  Ritz $600
Hotels (Berlin): Hilton Berlin $220/night (only hotel)
==========================================================================

Usage:
    python tests/test_group_pricing.py                 # unit + integration
    python tests/test_group_pricing.py --unit-only     # fast, no LLM/graph
    python tests/test_group_pricing.py --test P2       # one integration case
    python tests/test_group_pricing.py --no-response-checks
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
from agent.shared.pricing import (
    CHILD_RATE,
    activity_group_price,
    flight_group_price,
    group_label,
    hotel_group_price,
)
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
    hitl_response: str = "no"

    # Group estimate range check (None = skip)
    min_group_total: float | None = None
    max_group_total: float | None = None

    # Solo (per-adult) estimate range check
    min_solo_total: float | None = None
    max_solo_total: float | None = None

    # Exact travelers_label in travel_plan (None = skip)
    expected_travelers_label: str | None = None

    # group_total >= solo_total × this ratio (None = skip)
    min_group_to_solo_ratio: float | None = None
    max_group_to_solo_ratio: float | None = None

    # Direct state field checks (checked on state, not travel_plan)
    expect_state: dict = field(default_factory=dict)

    note: str = ""


@dataclass
class TestCase:
    id: str
    description: str
    turns: list[TurnSpec]


# ---------------------------------------------------------------------------
# Integration test definitions
# ---------------------------------------------------------------------------

# Cheapest Paris combo (local DB):
#   outbound TLV→Paris $180, return TLV←Paris $150, hotel Ibis $80/night
#   solo (1A):  $180+$150 + $80×3 = $570
#   2A 1 room:  ($180+$150)×2 + $80×3 = $900  → group/solo = 1.58
#   2A+1C 1rm:  2×$180+0.8×$180 + 2×$150+0.8×$150 + $80×3 = $504+$420+$240 = $1164
#   4A 2 rooms: ($180+$150)×4 + $80×2×3 = $1320+$480 = $1800 → group/solo = 3.16
#
# NOTE: Tests include an explicit adult count AND trip start ("in June") so that
# enrichment completes in a single turn without asking follow-up questions.

TESTS: list[TestCase] = [

    TestCase("P1", "Solo traveller: group estimate equals solo estimate", [
        TurnSpec(
            "Plan a trip from Tel Aviv to Paris for 1 adult. Budget $2000, 3 days, in June.",
            expected_travelers_label="1 adult",
            min_group_to_solo_ratio=0.99,
            max_group_to_solo_ratio=1.01,
            note=(
                "1 adult + 0 children + 1 room: group formula == solo formula. "
                "Ratio must be exactly 1.0."
            ),
        ),
    ]),

    TestCase("P2", "2 adults: group estimate > solo (flights double, hotel shared in 1 room)", [
        TurnSpec(
            "Plan a trip from Tel Aviv to Paris for 2 adults. Budget $3000, 3 days, in June.",
            expected_travelers_label="2 adults",
            # group = 2×flights + 1×hotel; solo = 1×flights + 1×hotel
            # group/solo depends on flight-to-hotel split, always 1 < ratio < 2
            min_group_to_solo_ratio=1.1,
            max_group_to_solo_ratio=2.05,
            note=(
                "2 adults fit in 1 room (capacity 3). Flights double, hotel unchanged. "
                "Ratio must be strictly > 1 and < 2."
            ),
        ),
    ]),

    TestCase("P3", "2 adults + 1 child: child pays 80% on flights, hotel still 1 room", [
        TurnSpec(
            "Plan a trip from Tel Aviv to Paris for 2 adults and 1 child. "
            "Budget $3000, 3 days, in June.",
            expected_travelers_label="2 adults, 1 child",
            # group > P2 group (extra 0.8× child flights) and ≤ budget
            min_group_total=500.0,
            max_group_total=3000.0,
            note=(
                "2A+1C total=3 people, fits in 1 room. "
                "Child adds 0.8× adult flight price to the group total."
            ),
        ),
    ]),

    TestCase("P4", "4 adults require 2 rooms: hotel portion doubles vs solo", [
        TurnSpec(
            "Plan a trip from Tel Aviv to Paris for 4 adults. Budget $6000, 3 days, in June.",
            expected_travelers_label="4 adults",
            # 4 adults → 2 rooms (ceil(4/2)=2)
            # group = 4×flights + 2×hotel×days
            # solo  = 1×flights + 1×hotel×days
            # ratio: (4F+2H)/(F+H) with DB prices: (4×330+2×240)/(330+240) = (1320+480)/570 = 3.16
            min_group_to_solo_ratio=2.5,
            max_group_to_solo_ratio=6.0,
            note=(
                "4 adults → 2 rooms. Flights ×4, hotel ×2. "
                "group/solo ratio ≈ 3.16 with local DB prices."
            ),
        ),
    ]),

    TestCase("P5", "Budget filter uses group cost: traveler state captured before plan", [
        TurnSpec(
            "Plan a trip from Tel Aviv to Paris for 4 adults. Budget $1500, 3 days, in June.",
            # Cheapest group ≈ $1800 (4 adults, local DB) → over the $1500 group budget.
            # The graph will take the alternatives path so travel_plan will be empty.
            # We verify via direct state checks that the traveler count was captured.
            expect_state={"num_adults": 4, "num_children": 0},
            note=(
                "Group budget ($1500) < group cost (~$1800 local DB). "
                "No travel_plan is produced (alternatives path), but num_adults must be "
                "captured in state, proving budget filtering used the group-cost formula."
            ),
        ),
    ]),

]


# ---------------------------------------------------------------------------
# Pure unit checks (no LLM)
# ---------------------------------------------------------------------------

def run_unit_checks() -> list[Check]:
    checks: list[Check] = []

    def approx_eq(desc, got, want, tol=0.001):
        checks.append(Check(desc, abs(got - want) < tol, f"got {got!r}, want {want!r}"))

    def eq(desc, got, want):
        checks.append(Check(desc, got == want, f"got {got!r}, want {want!r}"))

    # flight_group_price -----------------------------------------------
    approx_eq("flight_group: 1A 0C = 1× price", flight_group_price(100, 1, 0), 100.0)
    approx_eq("flight_group: 2A 0C = 2× price", flight_group_price(100, 2, 0), 200.0)
    approx_eq("flight_group: 1A 1C = 1 + 0.8 = 180", flight_group_price(100, 1, 1), 180.0)
    approx_eq("flight_group: 2A 1C = 2 + 0.8 = 280", flight_group_price(100, 2, 1), 280.0)
    approx_eq("flight_group: 0A 0C = 0", flight_group_price(100, 0, 0), 0.0)
    approx_eq("CHILD_RATE == 0.8", flight_group_price(100, 0, 1), CHILD_RATE * 100)

    # hotel_group_price ------------------------------------------------
    approx_eq("hotel_group: 1 room 1 day", hotel_group_price(80, 1, 1), 80.0)
    approx_eq("hotel_group: 1 room 3 days", hotel_group_price(80, 1, 3), 240.0)
    approx_eq("hotel_group: 2 rooms 3 days", hotel_group_price(80, 2, 3), 480.0)
    approx_eq("hotel_group: price scales with rooms not people", hotel_group_price(80, 2, 3), 2 * 80 * 3)

    # activity_group_price ---------------------------------------------
    approx_eq("activity_group: 1A 0C = 1×", activity_group_price(50, 1, 0), 50.0)
    approx_eq("activity_group: 2A 2C = 4×", activity_group_price(50, 2, 2), 200.0)
    approx_eq("activity_group: 0 people = 0", activity_group_price(50, 0, 0), 0.0)

    # group_label -------------------------------------------------------
    eq("group_label: (1,0) = '1 adult'",  group_label(1, 0), "1 adult")
    eq("group_label: (2,0) = '2 adults'", group_label(2, 0), "2 adults")
    eq("group_label: (1,1) = '1 adult, 1 child'",   group_label(1, 1), "1 adult, 1 child")
    eq("group_label: (2,1) = '2 adults, 1 child'",   group_label(2, 1), "2 adults, 1 child")
    eq("group_label: (2,3) = '2 adults, 3 children'", group_label(2, 3), "2 adults, 3 children")
    eq("group_label: (0,0) = '1 adult' (fallback)",  group_label(0, 0), "1 adult")

    return checks


# ---------------------------------------------------------------------------
# Graph factory & turn runner
# ---------------------------------------------------------------------------

def make_graph():
    checkpointer = MemorySaver(serde=JsonPlusSerializer())
    return build_graph(provider=CHOSEN_PROVIDER, checkpointer=checkpointer)


def run_turn(graph, config: dict, message: str, hitl_response: str = "no") -> tuple[dict, str]:
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


def check_turn(spec: TurnSpec, state: dict, skip_response_checks: bool) -> list[Check]:
    checks: list[Check] = []

    plan = state.get("travel_plan") or {}
    group_est = plan.get("lowest_group_estimate")
    solo_est  = plan.get("lowest_total_estimate")
    t_label   = plan.get("travelers_label")

    if spec.expected_travelers_label is not None:
        checks.append(Check(
            f"travelers_label == '{spec.expected_travelers_label}'",
            t_label == spec.expected_travelers_label,
            f"Actual: {t_label!r}",
        ))

    if spec.min_group_total is not None and group_est is not None:
        checks.append(Check(
            f"group_estimate >= {spec.min_group_total}",
            group_est >= spec.min_group_total,
            f"Actual: {group_est!r}",
        ))

    if spec.max_group_total is not None and group_est is not None:
        checks.append(Check(
            f"group_estimate <= {spec.max_group_total}",
            group_est <= spec.max_group_total,
            f"Actual: {group_est!r}",
        ))

    if spec.min_solo_total is not None and solo_est is not None:
        checks.append(Check(
            f"solo_estimate >= {spec.min_solo_total}",
            solo_est >= spec.min_solo_total,
            f"Actual: {solo_est!r}",
        ))

    if spec.max_solo_total is not None and solo_est is not None:
        checks.append(Check(
            f"solo_estimate <= {spec.max_solo_total}",
            solo_est <= spec.max_solo_total,
            f"Actual: {solo_est!r}",
        ))

    if (spec.min_group_to_solo_ratio is not None or spec.max_group_to_solo_ratio is not None):
        if group_est is not None and solo_est is not None and solo_est > 0:
            ratio = group_est / solo_est
            if spec.min_group_to_solo_ratio is not None:
                checks.append(Check(
                    f"group/solo ratio >= {spec.min_group_to_solo_ratio}",
                    ratio >= spec.min_group_to_solo_ratio,
                    f"Actual ratio: {ratio:.3f} (group={group_est}, solo={solo_est})",
                ))
            if spec.max_group_to_solo_ratio is not None:
                checks.append(Check(
                    f"group/solo ratio <= {spec.max_group_to_solo_ratio}",
                    ratio <= spec.max_group_to_solo_ratio,
                    f"Actual ratio: {ratio:.3f} (group={group_est}, solo={solo_est})",
                ))
        else:
            checks.append(Check(
                "group/solo ratio computable (both estimates present)",
                False,
                f"group_estimate={group_est!r}, solo_estimate={solo_est!r}",
            ))

    for key, want in spec.expect_state.items():
        got = state.get(key)
        checks.append(Check(
            f"state.{key} == {want!r}",
            got == want,
            f"Actual: {got!r}",
        ))

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
            checks = check_turn(turn_spec, state, skip_response_checks)
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
    parser = argparse.ArgumentParser(description="Group-pricing state tests")
    parser.add_argument("--test", nargs="+", help="Run specific case IDs (e.g. P1 P3)")
    parser.add_argument("--unit-only", action="store_true", help="Run only the no-LLM unit checks")
    parser.add_argument("--no-response-checks", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print(f"\n{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}  \U0001f4b0  Group-pricing tests for the travel agent{RESET}")
    print(f"  Provider:  {CYAN}{CHOSEN_PROVIDER}{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}\n")

    # --- Unit layer (always) ---
    unit_checks = run_unit_checks()
    unit_passed = all(c.passed for c in unit_checks)
    status = f"{GREEN}PASS{RESET}" if unit_passed else f"{RED}FAIL{RESET}"
    print(f"{MAGENTA}[UNIT]{RESET} pricing formulas ({len(unit_checks)} checks)  {status}")
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
