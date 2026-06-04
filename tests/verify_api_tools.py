#!/usr/bin/env python
"""Quick verification script for new API tools.

Run from project root:
    PYTHONPATH=src ./taenv/bin/python tests/verify_api_tools.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tools import core_tools, api_tools, find_connecting_flights

tools = core_tools + [find_connecting_flights] + api_tools


def verify_tools():
    print("=" * 70)
    print(" VERIFYING NEW API TOOLS")
    print("=" * 70)
    print()

    # Categorize tools
    original = [
        "fetch_flights",
        "fetch_hotels",
        "calculate_trip_cost",
        "fetch_activities",
        "get_best_time_to_visit",
        "get_average_weather",
        "find_connecting_flights",
    ]

    api_tools_expected = [
        "fetch_hotels_gm",
        "get_hotel_filter_options_gm",
        "fetch_attractions",
        "fetch_restaurants",
        "fetch_landmarks",
        "fetch_hotels_xotelo",
        "fetch_hotels_with_ratings_xotelo",
        "geocode_location",
        "fetch_attraction_details",
        "lookup_wikidata",
    ]

    tool_names = {t.name for t in tools}

    print(f"✓ Total tools: {len(tools)}")
    print()

    # Check original tools
    print("ORIGINAL TOOLS (kept for backward compatibility):")
    missing_original = []
    for name in original:
        if name in tool_names:
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name} (MISSING)")
            missing_original.append(name)
    print()

    # Check new API tools
    print("NEW API TOOLS:")
    missing_new = []
    for name in api_tools_expected:
        if name in tool_names:
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name} (MISSING)")
            missing_new.append(name)
    print()

    # Summary
    print("=" * 70)
    if not missing_original and not missing_new:
        print(" ✅ ALL TOOLS VERIFIED SUCCESSFULLY")
        print(f" ├─ Original tools: {len(original)}")
        print(f" ├─ New API tools: {len(api_tools_expected)}")
        print(f" └─ Total: {len(tools)}")
        print("=" * 70)
        return True
    else:
        print(" ❌ VERIFICATION FAILED")
        if missing_original:
            print(f" ├─ Missing original: {missing_original}")
        if missing_new:
            print(f" └─ Missing new: {missing_new}")
        print("=" * 70)
        return False


def show_tool_list():
    print()
    print("=" * 70)
    print(" TOOL DETAILS")
    print("=" * 70)
    print()
    for tool in sorted(tools, key=lambda t: t.name):
        print(f"{tool.name}")
        print(f"  Description: {(tool.description or 'N/A')[:80]}")
        print()


if __name__ == "__main__":
    success = verify_tools()
    show_tool_list()
    sys.exit(0 if success else 1)
