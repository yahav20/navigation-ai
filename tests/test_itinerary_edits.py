#!/usr/bin/env python3
"""Test suite for mid-trip edit scenarios.

Each test builds a baseline itinerary in Turn 1, then applies a specific edit
in Turn 2 and verifies that both state and response reflect the change.

Edit types covered:
    E1 — destination change (Paris → Berlin)
    E2 — day count change: 2 → 3 days (the "add one more day" bug)
    E3 — traveler count change (1 → 2 adults)
    E4 — budget change ($2000 → $1000, explicit rebuild)
    E5 — activity removal via update_itinerary path

Usage:
    python tests/test_itinerary_edits.py              # run all
    python tests/test_itinerary_edits.py --test E2    # run one
    python tests/test_itinerary_edits.py --no-response-checks
"""

import argparse
import os
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Callable

# Windows consoles default to a non-UTF-8 codepage (e.g. cp1255) that can't
# encode the status emojis/box characters below — force UTF-8 output.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.core.graph import build_graph
from config.config import CHOSEN_PROVIDER
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Command

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

TICK  = f"{GREEN}✔{RESET}"
CROSS = f"{RED}✘{RESET}"


# ---------------------------------------------------------------------------
# Data structures (mirrors test_itinerary.py)
# ---------------------------------------------------------------------------

@dataclass
class Check:
    description: str
    passed: bool
    detail: str = ""


@dataclass
class TurnSpec:
    # Either a literal string, or a callable that receives the PREVIOUS turn's
    # final state and returns the message (lets a turn target whatever the prior
    # turn actually scheduled, instead of hardcoding non-deterministic names).
    message: "str | Callable[[dict], str]"
    hitl_response: str = "no"           # "no"=standalone, "yes"=redirect to travel agent
    # Planner step assertions
    must_plan_steps: list[str]      = field(default_factory=list)
    must_not_plan_steps: list[str]  = field(default_factory=list)
    # State assertions
    expected_destination: str | None = None
    expected_days: int | None        = None
    expected_adults: int | None      = None   # num_adults in state
    expected_budget: float | None    = None   # total_budget in state
    expected_rooms: int | None       = None   # num_rooms in state
    # Travel-plan (flights+hotels flow) assertions
    hotel_min_stars: int | None      = None   # travel_plan.hotels[0].stars >= v
    hotel_max_price: float | None    = None   # travel_plan.hotels[0].price_per_night <= v
    group_total_trend: str | None    = None   # "increase" | "decrease" vs previous turn
    expect_flights_preserved: bool   = False  # has_flights True and flight_options non-empty
    # Surgical itinerary edit: only this day's activity set changed, others identical
    surgical_update_day: int | None  = None
    # Response text assertions
    response_must_contain: list[str]     = field(default_factory=list)
    response_must_not_contain: list[str] = field(default_factory=list)
    # Feasibility
    expect_feasible: bool | None = None
    # Optional hook for dynamic assertions: receives (state, response) → extra Checks
    post_check: "Callable[[dict, str], list] | None" = None
    note: str = ""


@dataclass
class TestCase:
    id: str
    section: str
    description: str
    turns: list[TurnSpec]


# ---------------------------------------------------------------------------
# E6 dynamic activity-replace helpers
# ---------------------------------------------------------------------------
# The geo-clustering scheduler decides which day each activity lands on, so we
# can't hardcode "replace X on Day 1". Instead Turn 2's message is built from
# Turn 1's actual schedule: target = a real Day-1 activity, replacement = a
# real Paris activity not scheduled anywhere (avoids dedup conflicts). Both are
# stashed so the post_check can assert the exact swap.

_E6: dict = {}

# Central, non-food Paris activities (real DB names) used as replacement candidates.
_E6_REPLACEMENT_POOL = [
    "Arc de Triomphe", "Sainte-Chapelle", "Centre Pompidou",
    "Montmartre Walking Tour", "Jardin du Luxembourg",
    "Seine River Cruise", "Catacombs of Paris",
]


def _e6_turn2_message(prev_state: dict) -> str:
    days = _day_activity_sets(prev_state)
    day1 = sorted(days.get(1, []))
    scheduled = set().union(*days.values()) if days else set()
    target = day1[0] if day1 else "Louvre Museum"
    replacement = next(
        (a for a in _E6_REPLACEMENT_POOL if a not in scheduled),
        "Catacombs of Paris",
    )
    _E6["target"] = target
    _E6["replacement"] = replacement
    return f"Replace the {target} with the {replacement} on Day 1."


def _e6_post_check(state: dict, response: str) -> list:
    days = _day_activity_sets(state)
    day1 = days.get(1, frozenset())
    target = _E6.get("target", "")
    replacement = _E6.get("replacement", "")
    return [
        Check(
            f"Day 1 dropped the replaced activity '{target}'",
            target not in day1,
            f"Day 1 = {sorted(day1)}",
        ),
        Check(
            f"Day 1 added the requested activity '{replacement}'",
            replacement in day1,
            f"Day 1 = {sorted(day1)}",
        ),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

TESTS: list[TestCase] = [

    # =========================================================
    # SECTION E — Edit / Modification Scenarios
    # =========================================================

    TestCase("E1", "E", "Edit destination: Paris → Berlin", [
        TurnSpec(
            "Build a 2-day day-by-day itinerary for 2 adults from Tel Aviv to Paris in June 2026. Budget $3500.",
            expected_destination="Paris",
            expected_days=2,
            must_plan_steps=["fetch_activities", "build_day_schedule"],
            note=(
                "Turn 1: Build baseline 2-day Paris itinerary. Generous budget so the "
                "critic doesn't fire a terminal fetch_min_prices replan (which would "
                "leave only that step in the final plan snapshot)."
            ),
        ),
        TurnSpec(
            "Actually, change the destination to Berlin instead.",
            expected_destination="Berlin",
            response_must_contain=["Berlin"],
            response_must_not_contain=["Paris"],
            note=(
                "Turn 2: Destination pivot via update_travel_plan → adjustments → "
                "enrichment → flight_search (itinerary_mode not set yet after "
                "adjustments clears it, or HITL re-prompts). "
                "State destination_city must become 'Berlin'."
            ),
        ),
    ]),

    TestCase("E2", "E", "Edit day count: 2 days → 3 days (implicit, no rebuild keywords)", [
        TurnSpec(
            "Create a 2-day day-by-day itinerary for 2 adults from Tel Aviv to Paris in June 2026. Budget $3500.",
            expected_destination="Paris",
            expected_days=2,
            must_plan_steps=["fetch_activities", "build_day_schedule"],
            note="Turn 1: Build baseline 2-day Paris standalone itinerary.",
        ),
        TurnSpec(
            "Add one more day.",
            expected_days=3,
            response_must_contain=["Day 3"],
            note=(
                "Turn 2 — THE MAIN BUG: no 'rebuild/schedule' keywords. "
                "Guardrail 4b detects 'one more day' → update_travel_plan; "
                "adjustments sets trip_days=3; after_enrichment routes to plan_check "
                "because itinerary_mode='standalone' is set from Turn 1. The real "
                "signals that the rebuild happened are trip_days=3 in state and a "
                "'Day 3' section in the response — plan-step assertions are unreliable "
                "here because the planner runs multiple replan passes and the final "
                "snapshot only reflects the last one."
            ),
        ),
    ]),

    TestCase("E3", "E", "Edit traveler count: 1 adult → 2 adults", [
        TurnSpec(
            "Build a 2-day day-by-day itinerary from Tel Aviv to Paris in June 2026. "
            "Just me, budget $1500.",
            expected_destination="Paris",
            expected_days=2,
            must_plan_steps=["build_day_schedule"],
            note="Turn 1: Build itinerary for 1 adult.",
        ),
        TurnSpec(
            "My partner is joining. Change it to 2 adults.",
            expected_adults=2,
            note=(
                "Turn 2: Traveler-only update goes through apply_traveler_updates "
                "in adjustments.py. Does NOT trigger flight/itinerary reset. "
                "num_adults must be 2 in state."
            ),
        ),
    ]),

    TestCase("E4", "E", "Edit budget: $2000 → $1000 after itinerary is built", [
        TurnSpec(
            "Plan a 2-day day-by-day itinerary for 2 adults from Tel Aviv to Paris in June 2026. Budget $3500.",
            expected_destination="Paris",
            expected_days=2,
            must_plan_steps=["fetch_activities", "build_day_schedule"],
            note="Turn 1: Build 2-day Paris itinerary with a comfortable $3500 budget.",
        ),
        TurnSpec(
            "I need to cut the budget to $1000. Rebuild the itinerary.",
            expected_days=2,
            expected_budget=1000.0,
            response_must_contain=["Paris"],
            note=(
                "Turn 2: Budget change triggers full rebuild. total_budget must be 1000 in state. "
                "Not asserting specific plan steps because the critic may trigger fetch_min_prices "
                "depending on standalone cost estimates, which is valid behavior."
            ),
        ),
    ]),

    TestCase("E5", "E", "Edit activity: remove Eiffel Tower from Day 1", [
        TurnSpec(
            "Build a 3-day day-by-day itinerary for 2 adults from Tel Aviv to Paris in June 2026. Budget $4500.",
            expected_destination="Paris",
            expected_days=3,
            must_plan_steps=["fetch_activities", "build_day_schedule"],
            note=(
                "Turn 1: Build 3-day Paris itinerary that likely includes Eiffel Tower. "
                "Generous budget avoids a terminal fetch_min_prices replan."
            ),
        ),
        TurnSpec(
            "Remove the Eiffel Tower from Day 1.",
            expected_days=3,
            response_must_not_contain=["eiffel tower"],
            note=(
                "Turn 2: update_itinerary path rebuilds Day 1 with Eiffel Tower excluded; "
                "trip_days stays 3. The real verification is that the response no longer "
                "mentions Eiffel Tower — plan-step assertions are skipped because replan "
                "passes make the final plan snapshot unreliable."
            ),
        ),
    ]),

    TestCase("E6", "E", "Replace a specific activity on Day 1 (surgical)", [
        TurnSpec(
            "Build a 3-day day-by-day itinerary for 2 adults from Tel Aviv to Paris in June 2026. Budget $4500.",
            expected_destination="Paris",
            expected_days=3,
            must_plan_steps=["fetch_activities", "build_day_schedule"],
            note="Turn 1: Build a 3-day Paris itinerary; Day 1's real activities drive Turn 2.",
        ),
        TurnSpec(
            _e6_turn2_message,                     # built from Turn 1's actual Day-1 schedule
            expected_days=3,                       # trip length unchanged
            surgical_update_day=1,                 # only Day 1 changes; Days 2-3 identical
            post_check=_e6_post_check,             # Day 1 dropped target & added replacement
            note=(
                "Turn 2: update_itinerary single-day rebuild. The message targets a real "
                "Day-1 activity and a replacement not yet scheduled, so the swap is "
                "deterministic. Day 1 set changes while Days 2-3 stay byte-identical, "
                "proving the planner did NOT re-plan the whole trip."
            ),
        ),
    ]),

    TestCase("E7", "E", "Edit rooms: 1 -> 2 rooms (cost rises, flights preserved)", [
        TurnSpec(
            "Plan a trip from Tel Aviv to Paris for 2 adults. Budget $4000, 3 days, in June.",
            expected_destination="Paris",
            expected_rooms=1,                      # 2 adults -> 1 room default (capacity 3)
            expect_flights_preserved=True,
            note="Turn 1: Full travel-planning flow produces flights + a curated travel_plan.",
        ),
        TurnSpec(
            "Make it 2 rooms.",
            expected_rooms=2,
            group_total_trend="increase",          # hotel billed per room x nights
            expect_flights_preserved=True,         # rooms-only change must NOT reset flights
            note=(
                "Turn 2: adjustments rooms-only path -> num_rooms=2, group cost up, "
                "flights/travel_plan intact (no re-search triggered)."
            ),
        ),
    ]),

    TestCase("E8", "E", "Edit hotel: upgrade to 5-star", [
        TurnSpec(
            "Plan a trip from Tel Aviv to Paris for 2 adults. Budget $5000, 3 days, in June.",
            expected_destination="Paris",
            note="Turn 1: Build a travel plan with a comfortable budget for any hotel tier.",
        ),
        TurnSpec(
            "I want a 5-star hotel.",
            hotel_min_stars=5,                     # only the Luxury Ritz (5*, $600) qualifies
            group_total_trend="increase",
            note=(
                "Turn 2: Guardrail 4c routes this to update_travel_plan; enrichment "
                "captures min_hotel_stars=5 and the travel agent re-curates to a 5-star hotel."
            ),
        ),
    ]),

    TestCase("E9", "E", "Edit hotel: switch to a cheaper hotel under $200/night", [
        TurnSpec(
            "Plan a trip from Tel Aviv to Paris for 2 adults. Budget $5000, 3 days, in June.",
            expected_destination="Paris",
            note=(
                "Turn 1: Plain travel plan, NO star preference — keeping the thread free of a "
                "min_hotel_stars floor so the Turn 2 max-price filter has no conflict."
            ),
        ),
        TurnSpec(
            "Switch to a cheaper hotel under $200 a night.",
            hotel_max_price=200.0,                 # Ibis $80 or Hotel de Ville $150 qualify
            note=(
                "Turn 2: Guardrail 4c routes 'hotel under' to update_travel_plan; enrichment "
                "captures max_hotel_price_per_night=200 and re-curates to a hotel <= $200/night. "
                "The top curated hotel must respect the price ceiling. (No 'decrease' trend "
                "assertion: lowest_group_estimate already used the cheapest hotel, so the "
                "absolute price check is the meaningful signal that the edit applied.)"
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

def run_turn(graph, config: dict, message: str, hitl_response: str = "no") -> tuple[dict, str]:
    """Stream one turn to completion; return (final_state, response_text).

    hitl_response: answer to plan_check's HITL interrupt —
        "no"  → build a standalone itinerary (no flights/hotels needed)
        "yes" → redirect to travel agent to fetch flights/hotels first
    """
    initial_state = {"messages": [("user", message)], "step_count": 0}

    for _ in graph.stream(initial_state, config, stream_mode="values"):
        pass

    while graph.get_state(config).next:
        for _ in graph.stream(Command(resume=hitl_response), config, stream_mode="values"):
            pass

    final_state = graph.get_state(config).values

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

def _day_activity_sets(state: dict) -> dict[int, frozenset]:
    """Map each built day → the set of its activity slot names.

    Reads itinerary_plan.step_results (accumulates across replans, unlike the
    execution_plan snapshot). Used to prove a single-day edit didn't re-plan
    the other days.
    """
    out: dict[int, frozenset] = {}
    results = (state.get("itinerary_plan") or {}).get("step_results", {}) or {}
    for key, wrapped in results.items():
        if not key.startswith("build_day_schedule"):
            continue
        inner = wrapped.get("data", wrapped) if isinstance(wrapped, dict) else {}
        day = inner.get("day")
        if day is None:
            continue
        names = frozenset(
            s.get("name", "")
            for s in inner.get("slots", [])
            if s.get("slot_type") == "activity"
        )
        out[int(day)] = names
    return out


def check_turn(
    spec: TurnSpec,
    state: dict,
    response: str,
    skip_response_checks: bool = False,
    prev_ctx: dict | None = None,
) -> list[Check]:
    checks: list[Check] = []
    prev_ctx = prev_ctx or {}

    plan_data = state.get("itinerary_plan", {})
    exec_plan = plan_data.get("execution_plan", {})
    plan_steps = exec_plan.get("steps", [])
    planned_step_types = [
        s.step_type if hasattr(s, "step_type") else s.get("step_type", "")
        for s in plan_steps
    ]

    # --- Planner step checks ---
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

    # --- State checks ---
    if spec.expected_destination:
        actual_dest = state.get("destination_city", "")
        checks.append(Check(
            f"Destination state is '{spec.expected_destination}'",
            actual_dest.lower() == spec.expected_destination.lower() if actual_dest else False,
            f"Actual: {actual_dest!r}",
        ))

    if spec.expected_days is not None:
        actual_days = state.get("trip_days", 0)
        checks.append(Check(
            f"Trip days state is {spec.expected_days}",
            actual_days == spec.expected_days,
            f"Actual: {actual_days}",
        ))

    if spec.expected_adults is not None:
        actual_adults = state.get("num_adults", 0)
        checks.append(Check(
            f"num_adults state is {spec.expected_adults}",
            actual_adults == spec.expected_adults,
            f"Actual: {actual_adults}",
        ))

    if spec.expected_budget is not None:
        actual_budget = state.get("total_budget", 0)
        checks.append(Check(
            f"total_budget state is {spec.expected_budget}",
            actual_budget == spec.expected_budget,
            f"Actual: {actual_budget}",
        ))

    if spec.expect_feasible is not None:
        actual_feasible = state.get("itinerary_feasible", True)
        checks.append(Check(
            f"Feasibility is {spec.expect_feasible}",
            actual_feasible == spec.expect_feasible,
            f"Actual: {actual_feasible}",
        ))

    if spec.expected_rooms is not None:
        actual_rooms = state.get("num_rooms")
        checks.append(Check(
            f"num_rooms state is {spec.expected_rooms}",
            actual_rooms == spec.expected_rooms,
            f"Actual: {actual_rooms!r}",
        ))

    # --- Travel-plan (flights + hotels) checks ---
    travel_plan = state.get("travel_plan") or {}
    hotels = travel_plan.get("hotels") or []
    top_hotel = hotels[0] if hotels else {}

    if spec.hotel_min_stars is not None:
        stars = top_hotel.get("stars")
        checks.append(Check(
            f"Selected hotel stars >= {spec.hotel_min_stars}",
            stars is not None and stars >= spec.hotel_min_stars,
            f"Actual: {top_hotel.get('name')!r} stars={stars!r}",
        ))

    if spec.hotel_max_price is not None:
        price = top_hotel.get("price_per_night")
        checks.append(Check(
            f"Selected hotel price_per_night <= {spec.hotel_max_price}",
            price is not None and price <= spec.hotel_max_price,
            f"Actual: {top_hotel.get('name')!r} price={price!r}",
        ))

    if spec.group_total_trend is not None:
        cur = travel_plan.get("lowest_group_estimate")
        prev = prev_ctx.get("group_est")
        if cur is None or prev is None:
            checks.append(Check(
                f"group estimate trend {spec.group_total_trend} (both turns present)",
                False,
                f"prev={prev!r}, current={cur!r}",
            ))
        elif spec.group_total_trend == "increase":
            checks.append(Check(
                "group estimate increased vs previous turn",
                cur > prev,
                f"prev={prev}, current={cur}",
            ))
        else:  # "decrease"
            checks.append(Check(
                "group estimate decreased vs previous turn",
                cur < prev,
                f"prev={prev}, current={cur}",
            ))

    if spec.expect_flights_preserved:
        has_flights = bool(state.get("has_flights"))
        flight_options = state.get("flight_options") or []
        checks.append(Check(
            "flights preserved (has_flights and flight_options non-empty)",
            has_flights and len(flight_options) > 0,
            f"has_flights={has_flights}, flight_options={len(flight_options)}",
        ))

    # --- Surgical single-day edit: only this day changed, others identical ---
    if spec.surgical_update_day is not None:
        cur_days = _day_activity_sets(state)
        prev_days = prev_ctx.get("day_acts") or {}
        day = spec.surgical_update_day
        changed = (
            day in cur_days and day in prev_days
            and cur_days[day] != prev_days[day]
        )
        checks.append(Check(
            f"Day {day} activity set changed",
            changed,
            f"prev={sorted(prev_days.get(day, []))}, cur={sorted(cur_days.get(day, []))}",
        ))
        untouched = [
            d for d in prev_days
            if d != day and cur_days.get(d) != prev_days.get(d)
        ]
        checks.append(Check(
            f"Other days unchanged (no full re-plan)",
            not untouched,
            f"Days that unexpectedly changed: {untouched}",
        ))

    # --- Response text checks ---
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

    if spec.post_check is not None:
        checks.extend(spec.post_check(state, response))

    return checks


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_test(test: TestCase, skip_response_checks: bool = False):
    graph = make_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    all_turn_results = []
    elapsed = 0.0

    prev_ctx: dict = {"group_est": None, "day_acts": {}}
    prev_state: dict = {}

    for i, turn_spec in enumerate(test.turns):
        t0 = time.time()
        try:
            message = (
                turn_spec.message(prev_state)
                if callable(turn_spec.message)
                else turn_spec.message
            )
            state, response = run_turn(graph, config, message, turn_spec.hitl_response)
            checks = check_turn(turn_spec, state, response, skip_response_checks, prev_ctx)
            # Capture context for the next turn's trend / surgical comparisons.
            tp = state.get("travel_plan") or {}
            if tp.get("lowest_group_estimate") is not None:
                prev_ctx["group_est"] = tp.get("lowest_group_estimate")
            day_acts = _day_activity_sets(state)
            if day_acts:
                prev_ctx["day_acts"] = day_acts
            prev_state = state
        except Exception as exc:
            tb = traceback.format_exc()
            checks = [Check(f"Turn {i + 1} execution crashed", False, str(exc))]
            print(f"\n{DIM}{tb}{RESET}")

        elapsed += time.time() - t0
        all_turn_results.append((i + 1, turn_spec, checks))

    all_passed = all(c.passed for _, _, checks in all_turn_results for c in checks)
    return all_passed, all_turn_results, elapsed


def _format_state_snapshot(state: dict) -> str:
    return (
        f"dest={state.get('destination_city', '—')} | "
        f"days={state.get('trip_days', '—')} | "
        f"adults={state.get('num_adults', '—')} | "
        f"budget={state.get('total_budget', '—')} | "
        f"feasible={state.get('itinerary_feasible', '—')} | "
        f"mode={state.get('itinerary_mode', '—')}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run itinerary edit scenario tests")
    parser.add_argument("--test", nargs="+", help="Run specific test IDs (e.g. E1 E2)")
    parser.add_argument("--no-response-checks", action="store_true",
                        help="Skip LLM response-text checks")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed output for passing tests too")
    args = parser.parse_args()

    tests_to_run = TESTS
    if args.test:
        test_ids = {t.upper() for t in args.test}
        tests_to_run = [t for t in tests_to_run if t.id in test_ids]

    skip_resp = args.no_response_checks
    total = len(tests_to_run)

    if total == 0:
        print(f"{RED}No tests matched the filter.{RESET}")
        sys.exit(1)

    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}  ✏️  Itinerary Edit Scenarios Test Runner{RESET}")
    print(f"  Provider: {CYAN}{CHOSEN_PROVIDER}{RESET}")
    print(f"  Tests:    {total}")
    if skip_resp:
        print(f"  {YELLOW}⚠ Response text checks DISABLED{RESET}")
    print(f"{BOLD}{'='*70}{RESET}\n")

    results = []

    for i, test in enumerate(tests_to_run, 1):
        print(f"{DIM}[{i}/{total}]{RESET} "
              f"{CYAN}{test.id}{RESET} {test.description}",
              end="  ", flush=True)

        passed, turn_results, elapsed = run_test(test, skip_response_checks=skip_resp)

        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"{status}  {DIM}({elapsed:.1f}s){RESET}")

        if not passed or args.verbose:
            for turn_idx, turn_spec, checks in turn_results:
                has_failures = any(not c.passed for c in checks)
                msg_preview = turn_spec.message[:80] + ("…" if len(turn_spec.message) > 80 else "")
                if has_failures or args.verbose:
                    print(f'    Turn {turn_idx}: "{msg_preview}"')
                    if turn_spec.note:
                        print(f"      {DIM}📝 {turn_spec.note}{RESET}")
                    for c in checks:
                        icon = TICK if c.passed else CROSS
                        print(f"      {icon} {c.description}")
                        if not c.passed and c.detail:
                            print(f"        {DIM}↳ {c.detail}{RESET}")

        results.append(passed)

    passed_count = sum(results)
    print(f"\n{BOLD}{'='*70}{RESET}")
    color = GREEN if passed_count == total else RED
    print(f"  {BOLD}{color}Total: {passed_count}/{total}{RESET}")
    print(f"{BOLD}{'='*70}{RESET}\n")

    sys.exit(0 if passed_count == total else 1)


if __name__ == "__main__":
    main()
