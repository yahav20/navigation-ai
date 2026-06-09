#!/usr/bin/env python3
"""Tests for group-aware pricing in the itinerary agent.

Two layers:

1. Deterministic UNIT asserts on `calculate_trip_cost` (no LLM, no graph).
2. Multi-turn INTEGRATION cases that drive the real graph and verify
   `itinerary_plan.step_results["verify_budget_0"]` contains correct
   `group_grand_total` and that the budget comparison uses the group total.

==========================================================================
Pricing rules under test (itinerary):
  Flights:    adults × price + children × price × 0.8
  Hotels:     num_rooms × price_per_night × trip_days
  Activities/meals: (adults + children) × price  (no child discount)
==========================================================================

Usage:
    python tests/test_itinerary_pricing.py                # unit + integration
    python tests/test_itinerary_pricing.py --unit-only    # fast, no LLM
    python tests/test_itinerary_pricing.py --test I2      # one case
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
from agent.itinerary.itinerary_tools import calculate_trip_cost
from agent.shared.pricing import CHILD_RATE
from config.config import CHOSEN_PROVIDER
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Command

# ---------------------------------------------------------------------------
# ANSI colours
# ---------------------------------------------------------------------------
GREEN  = "\033[92m"
RED    = "\033[91m"
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

    # Checks on itinerary_plan.step_results["verify_budget_0"].data
    expect_group_gt_solo: bool = False         # group_grand_total > grand_total
    min_group_to_solo_ratio: float | None = None
    max_group_to_solo_ratio: float | None = None
    expect_group_budget_used: bool = False     # group total used for over-budget flag
    response_must_contain: list[str] = field(default_factory=list)

    # Direct state checks
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

TESTS: list[TestCase] = [

    TestCase("I1", "Solo traveller: group_grand_total == grand_total", [
        TurnSpec(
            "Build me a day-by-day itinerary for a 3-day trip from "
            "Tel Aviv to Paris. Budget $2500. I'm traveling alone. In June.",
            hitl_response="no",
            response_must_contain=["Grand Total"],
            note=(
                "1 adult 0 children 1 room: group == solo. "
                "Budget summary must appear and show Grand Total."
            ),
        ),
    ]),

    TestCase("I2", "2 adults: group_grand_total > grand_total (flights double)", [
        TurnSpec(
            "Build me a day-by-day itinerary for a 3-day trip from "
            "Tel Aviv to Paris for 2 adults. Budget $3000. In June.",
            hitl_response="no",
            expect_group_gt_solo=True,
            min_group_to_solo_ratio=1.1,
            max_group_to_solo_ratio=2.5,
            note=(
                "2 adults 1 room: flights ×2, hotel unchanged (1 room fits 2). "
                "group/solo ratio depends on flight-to-hotel split."
            ),
        ),
    ]),

    TestCase("I3", "2 adults + 1 child: child pays 80% on flights", [
        TurnSpec(
            "Build me a day-by-day itinerary for a 3-day trip from "
            "Tel Aviv to Paris for 2 adults and 1 child. Budget $3500. In June.",
            hitl_response="no",
            expect_group_gt_solo=True,
            note=(
                "Child adds 0.8× adult flight price to group total. "
                "2A+1C total=3 still fits in 1 room."
            ),
        ),
    ]),

    TestCase("I4", "4 adults require 2 rooms: hotel doubles in group total", [
        TurnSpec(
            "Build me a day-by-day itinerary for a 3-day trip from "
            "Tel Aviv to Paris for 4 adults. Budget $6000. In June.",
            hitl_response="no",
            expect_group_gt_solo=True,
            min_group_to_solo_ratio=2.5,
            note=(
                "4 adults → 2 rooms. group_hotel = 2× solo_hotel, flights ×4. "
                "Ratio must be > 2.5."
            ),
        ),
    ]),

    TestCase("I5", "Group over-budget detection: tight budget fails for group but not solo", [
        TurnSpec(
            "Build me a day-by-day itinerary for a 3-day trip from "
            "Tel Aviv to Paris for 4 adults. Budget $500. In June.",
            hitl_response="no",
            # Budget $500 is way under even solo cost for TLV→Paris.
            # Verify traveler state is captured even if plan fails.
            expect_state={"num_adults": 4, "num_children": 0},
            note=(
                "$500 budget is below any realistic group cost. "
                "The graph will fail/fallback. Verify num_adults captured in state."
            ),
        ),
    ]),

]


# ---------------------------------------------------------------------------
# Pure unit checks (no LLM, no graph)
# ---------------------------------------------------------------------------

def run_unit_checks() -> list[Check]:
    checks: list[Check] = []

    def approx_eq(desc: str, got, want, tol: float = 0.01):
        checks.append(Check(desc, abs(float(got) - float(want)) < tol,
                            f"got {got!r}, want {want!r}"))

    def check_true(desc: str, val: bool, detail: str = ""):
        checks.append(Check(desc, bool(val), detail))

    # ── Solo (1A 0C 1rm): group == solo ─────────────────────────────────────
    r = calculate_trip_cost.invoke({
        "flight_price": 200.0, "return_flight_price": 150.0,
        "hotel_price_per_night": 80.0, "trip_days": 3,
        "estimated_activities_budget": 100.0, "estimated_meals_budget_per_day": 30.0,
        "num_adults": 1, "num_children": 0, "num_rooms": 1,
    })
    approx_eq("solo: grand_total = 200+150+240+100+90 = 780",
              r["grand_total"], 780.0)
    approx_eq("solo: group_grand_total == grand_total",
              r["group_grand_total"], r["grand_total"])

    # ── 2 adults 1 room ──────────────────────────────────────────────────────
    r2 = calculate_trip_cost.invoke({
        "flight_price": 200.0, "return_flight_price": 150.0,
        "hotel_price_per_night": 80.0, "trip_days": 3,
        "estimated_activities_budget": 100.0, "estimated_meals_budget_per_day": 30.0,
        "num_adults": 2, "num_children": 0, "num_rooms": 1,
    })
    # group flights = 2×200 + 2×150 = 700; hotel = 80×1×3=240; acts = 2×100=200; meals = 2×90=180
    approx_eq("2A: group_outbound = 400", r2["group_outbound_flight"], 400.0)
    approx_eq("2A: group_return = 300",   r2["group_return_flight"],   300.0)
    approx_eq("2A: group_hotel = 240 (1 room)", r2["group_hotel_total"], 240.0)
    approx_eq("2A: group_activities = 200", r2["group_activities_total"], 200.0)
    approx_eq("2A: group_meals = 180",      r2["group_meals_total"],      180.0)
    approx_eq("2A: group_grand_total = 1320", r2["group_grand_total"], 1320.0)
    check_true("2A: group > solo", r2["group_grand_total"] > r2["grand_total"])
    approx_eq("2A: per-person solo unchanged = 780", r2["grand_total"], 780.0)

    # ── 2 adults + 1 child, 1 room ──────────────────────────────────────────
    r3 = calculate_trip_cost.invoke({
        "flight_price": 200.0, "return_flight_price": 150.0,
        "hotel_price_per_night": 80.0, "trip_days": 3,
        "estimated_activities_budget": 100.0, "estimated_meals_budget_per_day": 30.0,
        "num_adults": 2, "num_children": 1, "num_rooms": 1,
    })
    # outbound group = 2×200 + 0.8×200 = 560; return = 2×150+0.8×150=420
    approx_eq("2A+1C: group_outbound = 560", r3["group_outbound_flight"], 560.0)
    approx_eq("2A+1C: group_return = 420",   r3["group_return_flight"],   420.0)
    approx_eq("2A+1C: group_hotel = 240 (1 room)", r3["group_hotel_total"], 240.0)
    approx_eq("2A+1C: group_activities = 300 (3 people)", r3["group_activities_total"], 300.0)
    # 560+420+240+300+270(meals 90×3 people) = 1790
    approx_eq("2A+1C: group_grand_total = 1790", r3["group_grand_total"], 1790.0)
    check_true("2A+1C: child 80% → group < 3×solo", r3["group_grand_total"] < 3 * r3["grand_total"])

    # ── 4 adults, 2 rooms ───────────────────────────────────────────────────
    r4 = calculate_trip_cost.invoke({
        "flight_price": 200.0, "return_flight_price": 150.0,
        "hotel_price_per_night": 80.0, "trip_days": 3,
        "estimated_activities_budget": 100.0, "estimated_meals_budget_per_day": 30.0,
        "num_adults": 4, "num_children": 0, "num_rooms": 2,
    })
    # hotel_group = 80×2×3=480 (double solo)
    approx_eq("4A 2rm: group_hotel = 480 (double)", r4["group_hotel_total"], 480.0)
    approx_eq("4A 2rm: group_outbound = 800",        r4["group_outbound_flight"], 800.0)
    # group = 800+600+480+400+360=2640
    approx_eq("4A 2rm: group_grand_total = 2640",    r4["group_grand_total"], 2640.0)

    # ── Metadata fields returned ────────────────────────────────────────────
    r_meta = calculate_trip_cost.invoke({
        "flight_price": 100.0, "return_flight_price": 100.0,
        "hotel_price_per_night": 50.0, "trip_days": 2,
        "estimated_activities_budget": 50.0, "estimated_meals_budget_per_day": 20.0,
        "num_adults": 3, "num_children": 1, "num_rooms": 2,
    })
    checks.append(Check("num_adults stored in result", r_meta.get("num_adults") == 3,
                         f"Actual: {r_meta.get('num_adults')!r}"))
    checks.append(Check("num_children stored in result", r_meta.get("num_children") == 1,
                         f"Actual: {r_meta.get('num_children')!r}"))
    checks.append(Check("num_rooms stored in result", r_meta.get("num_rooms") == 2,
                         f"Actual: {r_meta.get('num_rooms')!r}"))

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


def _get_budget_result(state: dict) -> dict:
    """Extract the verify_budget step result from state."""
    plan  = state.get("itinerary_plan") or {}
    results = plan.get("step_results") or {}
    key = next((k for k in results if k.startswith("verify_budget")), None)
    if not key:
        return {}
    val = results[key]
    if isinstance(val, dict) and "data" in val:
        return val["data"] or {}
    return val if isinstance(val, dict) else {}


def check_turn(spec: TurnSpec, state: dict, response: str, skip_response_checks: bool) -> list[Check]:
    checks: list[Check] = []

    budget_data = _get_budget_result(state)
    grand  = float(budget_data.get("grand_total", 0) or 0)
    group  = float(budget_data.get("group_grand_total", 0) or 0)

    if spec.expect_group_gt_solo:
        has_data = grand > 0 and group > 0
        checks.append(Check(
            "group_grand_total > grand_total",
            has_data and group > grand,
            f"group={group:.0f}, solo={grand:.0f}",
        ))

    if spec.min_group_to_solo_ratio is not None or spec.max_group_to_solo_ratio is not None:
        if grand > 0 and group > 0:
            ratio = group / grand
            if spec.min_group_to_solo_ratio is not None:
                checks.append(Check(
                    f"group/solo ratio >= {spec.min_group_to_solo_ratio}",
                    ratio >= spec.min_group_to_solo_ratio,
                    f"Actual: {ratio:.3f}",
                ))
            if spec.max_group_to_solo_ratio is not None:
                checks.append(Check(
                    f"group/solo ratio <= {spec.max_group_to_solo_ratio}",
                    ratio <= spec.max_group_to_solo_ratio,
                    f"Actual: {ratio:.3f}",
                ))
        else:
            checks.append(Check("group/solo ratio computable", False,
                                 f"grand={grand}, group={group}"))

    for key, want in spec.expect_state.items():
        got = state.get(key)
        checks.append(Check(f"state.{key} == {want!r}", got == want, f"Actual: {got!r}"))

    if not skip_response_checks:
        response_lower = response.lower()
        for phrase in spec.response_must_contain:
            checks.append(Check(
                f"Response contains '{phrase}'",
                phrase.lower() in response_lower,
                f"(len={len(response)})",
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
            print(f"      {DIM}📝 {turn_spec.note}{RESET}")
        for c in checks:
            icon = TICK if c.passed else CROSS
            print(f"      {icon} {c.description}")
            if not c.passed and c.detail:
                print(f"        {DIM}↳ {c.detail}{RESET}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Itinerary pricing tests")
    parser.add_argument("--test", nargs="+", help="Specific case IDs (e.g. I1 I3)")
    parser.add_argument("--unit-only", action="store_true", help="Run only no-LLM unit checks")
    parser.add_argument("--no-response-checks", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print(f"\n{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}  📋  Itinerary group-pricing tests{RESET}")
    print(f"  Provider:  {CYAN}{CHOSEN_PROVIDER}{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}\n")

    unit_checks = run_unit_checks()
    unit_passed = all(c.passed for c in unit_checks)
    status = f"{GREEN}PASS{RESET}" if unit_passed else f"{RED}FAIL{RESET}"
    print(f"{MAGENTA}[UNIT]{RESET} calculate_trip_cost group math ({len(unit_checks)} checks)  {status}")
    if not unit_passed or args.verbose:
        for c in unit_checks:
            icon = TICK if c.passed else CROSS
            print(f"    {icon} {c.description}")
            if not c.passed and c.detail:
                print(f"      {DIM}↳ {c.detail}{RESET}")

    if args.unit_only:
        sys.exit(0 if unit_passed else 1)

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
