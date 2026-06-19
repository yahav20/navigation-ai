"""Offline unit tests for multi-destination orchestration.

No LLM, no network, no DB writes — exercises the deterministic node logic only.
Run: tavenv/bin/python -m pytest tests/test_multi_dest.py -q
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.itinerary.multi_dest import (
    SegmentPlannerNode, LegDispatchNode, LegCollectNode, TripFormatterNode,
    after_segment_planner, after_leg_collect, _normalize_days, _cheapest, _demote_heading,
)


# ── _normalize_days ──────────────────────────────────────────────────────────

def test_normalize_days_sums_to_total():
    assert sum(_normalize_days([3, 3, 3], 14)) == 14
    assert sum(_normalize_days([5, 2], 7)) == 7
    assert sum(_normalize_days([10], 3)) == 3          # over-shoot clamped down
    assert all(d >= 1 for d in _normalize_days([1, 1, 1], 3))


# ── SegmentPlannerNode: single-destination baseline ──────────────────────────

def test_segment_planner_non_standalone_stays_single():
    node = SegmentPlannerNode(llm=None)
    out = node({"destination_city": "Rome", "trip_days": 4,
                "itinerary_mode": "with_travel_data", "total_budget": 1000})
    assert out["is_multi_destination"] is False
    assert len(out["trip_segments"]) == 1
    assert out["trip_segments"][0] == {
        "destination": "Rome", "days": 4, "order": 0, "drive_from_prev": None
    }
    assert out["itineraries"] == []
    assert out["total_trip_days"] == 4


def test_segment_planner_no_llm_stays_single():
    # Even in standalone mode, with no LLM the splitter can't run → single.
    node = SegmentPlannerNode(llm=None)
    out = node({"destination_city": "Rome", "trip_days": 30,
                "itinerary_mode": "standalone", "total_budget": 0})
    assert out["is_multi_destination"] is False
    assert len(out["trip_segments"]) == 1


# ── after_segment_planner routing ─────────────────────────────────────────────

def test_after_segment_planner_routes_by_multi_flag():
    assert after_segment_planner({"is_multi_destination": True}) == "multi_flight"
    assert after_segment_planner({"is_multi_destination": False}) == "leg_dispatch"
    assert after_segment_planner({}) == "leg_dispatch"


# ── LegDispatchNode ───────────────────────────────────────────────────────────

def test_leg_dispatch_single_keeps_mode_and_includes_flight():
    segs = [{"destination": "Rome", "days": 4, "order": 0, "drive_from_prev": None}]
    out = LegDispatchNode()({
        "trip_segments": segs, "seg_index": 0, "is_multi_destination": False,
        "itinerary_mode": "standalone",
    })
    assert out["destination_city"] == "Rome"
    assert out["trip_days"] == 4
    assert out["include_flight"] is True
    assert "itinerary_mode" not in out          # single must NOT override the mode
    assert out["itinerary_plan"] == {}


def test_leg_dispatch_multi_forces_standalone_hotel_only():
    segs = [
        {"destination": "Rome", "days": 4, "order": 0, "drive_from_prev": None},
        {"destination": "Florence", "days": 3, "order": 1, "drive_from_prev": "Rome"},
    ]
    out = LegDispatchNode()({
        "trip_segments": segs, "seg_index": 1, "is_multi_destination": True,
    })
    assert out["destination_city"] == "Florence"
    assert out["trip_days"] == 3
    assert out["itinerary_mode"] == "standalone"
    assert out["include_flight"] is False
    assert out["total_budget"] == 0.0


# ── LegCollectNode + after_leg_collect loop ──────────────────────────────────

def test_leg_collect_resets_on_first_and_appends_after():
    segs = [{"destination": "Rome", "days": 4, "order": 0, "drive_from_prev": None},
            {"destination": "Florence", "days": 3, "order": 1, "drive_from_prev": "Rome"}]
    # First leg (idx 0): resets any stale list from a prior turn.
    s0 = {"trip_segments": segs, "seg_index": 0, "destination_city": "Rome",
          "trip_days": 4, "itinerary_plan": {"final_markdown": "ROME"},
          "itineraries": [{"stale": True}]}
    out0 = LegCollectNode()(s0)
    assert len(out0["itineraries"]) == 1
    assert out0["itineraries"][0]["destination"] == "Rome"
    assert out0["seg_index"] == 1
    assert after_leg_collect({**s0, **out0}) == "leg_dispatch"   # one more city

    # Second leg (idx 1): appends to the running list.
    s1 = {"trip_segments": segs, "seg_index": 1, "destination_city": "Florence",
          "trip_days": 3, "itinerary_plan": {"final_markdown": "FLO"},
          "itineraries": out0["itineraries"]}
    out1 = LegCollectNode()(s1)
    assert [l["destination"] for l in out1["itineraries"]] == ["Rome", "Florence"]
    assert after_leg_collect({**s1, **out1}) == "trip_formatter"  # done


def test_after_leg_collect_update_path_no_segments():
    # update_itinerary path: no segments → finish after one leg.
    assert after_leg_collect({"seg_index": 1, "trip_segments": []}) == "trip_formatter"


# ── TripFormatterNode: single delegates to the real formatter ────────────────

def test_trip_formatter_single_delegates():
    sentinel = {"messages": ["DELEGATED"], "ui": []}

    class FakeFormatter:
        def __call__(self, state):
            return sentinel

    node = TripFormatterNode(llm=None)
    node._formatter = FakeFormatter()                 # swap in a spy
    # 0 legs (infeasible single / update) and 1 leg both delegate.
    assert node({"itineraries": []}) is sentinel
    assert node({"itineraries": [{"order": 0, "destination": "Rome"}]}) is sentinel


def test_trip_formatter_multi_stitches():
    node = TripFormatterNode(llm=None)
    legs = [
        {"order": 0, "destination": "Rome", "days": 4, "drive_from_prev": None,
         "itinerary_plan": {"final_markdown": "# Rome\n## Day 1\nColosseum",
                            "step_results": {"verify_budget_0": {"data": {"grand_total": 800}}}}},
        {"order": 1, "destination": "Florence", "days": 3, "drive_from_prev": "Rome",
         "itinerary_plan": {"final_markdown": "# Florence\n## Day 1\nUffizi",
                            "step_results": {"verify_budget_0": {"data": {"grand_total": 500}}}}},
    ]
    out = node({
        "itineraries": legs, "current_city": "Tel Aviv", "total_trip_days": 7,
        "trip_total_budget": 2000,
        "flight_options": [{"price": 300, "flight_number": "AB1"}],
        "return_flight_options": [{"price": 250, "flight_number": "AB2"}],
    })
    # Two messages: a visible overview, then a viewer-only message.
    assert len(out["messages"]) == 2
    overview, plans = out["messages"]
    assert overview.name == "trip_overview"
    assert plans.name == "itinerary_formatter"
    assert plans.content == ""

    content = overview.content
    assert "Multi-City Trip" in content
    assert "Rome" in content and "Florence" in content
    assert "drive from Rome" in content          # route outline shows the drive
    # flight 550 + legs 800+500 = 1850; under the 2000 budget
    assert "~$1850" in content
    assert "✅" in content

    # one map widget per city, all sharing the plans message id
    assert len(out["ui"]) == 2
    assert {u["metadata"]["message_id"] for u in out["ui"]} == {plans.id}


# ── pure helpers ──────────────────────────────────────────────────────────────

def test_cheapest_skips_messages_and_picks_min():
    flights = [{"message": "none"}, {"price": 500}, {"price": 200}, {"total_price": 350}]
    assert _cheapest(flights)["price"] == 200
    assert _cheapest([]) is None
    assert _cheapest([{"message": "x"}]) is None


def test_demote_heading_drops_h1():
    md = "# Your Rome Itinerary\n## Day 1\nstuff"
    out = _demote_heading(md)
    assert "# Your Rome Itinerary" not in out
    assert "## Day 1" in out
