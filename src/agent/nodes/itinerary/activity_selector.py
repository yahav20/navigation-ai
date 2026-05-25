# src/agent/nodes/itinerary/activity_selector.py
"""
ActivitySelector
================
The ONLY job of the LLM in the new architecture:
  Given a list of available activities and user preferences,
  return an *ordered* list of activity names for EACH day —
  sorted by strategic fit (not just rating).

The LLM knows nothing about times, transit, or costs here.
That's the ScheduleEngine's job.

Selection strategy the LLM applies:
  1. Geographic clustering — group nearby activities per day (minimises taxi cost)
  2. Energy curve — morning: museums/walks; afternoon: outdoor; evening: food/shows
  3. Preference filter — kosher, dietary, age, categories
  4. No repeats across days
  5. Fallback: sort by rating if LLM fails
    6.You MUST select a mix of sightseeing AND food venues. 
    Make sure to include at least 1-2 restaurants, cafes, or food tours (from the provided database list) for EACH day of the trip. 
    Do not only select museums and parks! The schedule engine relies on your food selections to feed the user.

Output format (structured):
  {"day_1": ["Activity A", "Activity B", ...], "day_2": [...], ...}
"""
from __future__ import annotations

import json
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

SELECTOR_SYSTEM = """
You are a senior travel curator. Your ONLY job: given a list of activities,
assign them to days in the optimal order.

STRATEGY:
1. GEOGRAPHIC CLUSTERING: Group activities close to each other on the same day.
   Minimise total transit distance per day. Never put two far-apart activities
   on the same day if there are closer alternatives.

2. ENERGY CURVE (within each day):
   - Morning (08:00-12:00): low-effort cultural (museums, markets, historic sites)
   - Afternoon (12:00-17:00): moderate outdoor (parks, walks, viewpoints)
   - Evening (17:00-22:00): food-rich, entertainment, nightlife
   This ordering within each day is MANDATORY.

3. PREFERENCE FILTER:
   - If dietary restrictions present: exclude activities with food_available=true
     that don't match (e.g. non-kosher food venues).
   - If age restrictions present: exclude activities where min_age > traveller_age.
   - If specific categories requested: prioritise matching activities.
   - If NO preferences: use pure rating + geographic cluster logic.

4. NO REPEATS: Each activity appears in exactly one day.

5. MAX ACTIVITIES PER DAY: 4 (the ScheduleEngine will decide which fit time-wise).
   Always include 5-6 candidates so the engine has buffer.

OUTPUT:
Return ONLY a JSON object like:
{
  "day_1": ["Eiffel Tower", "Louvre Museum", "Seine River Cruise", "Montmartre Walk"],
  "day_2": ["Versailles Palace", "Luxembourg Gardens", "Latin Quarter", "Pompidou Centre"],
  ...
}
No explanation, no markdown, no extra keys.
"""


def select_activities_per_day(
    llm: BaseChatModel,
    activities: list[dict],
    trip_days: int,
    prefs: dict,
    destination: str,
) -> dict[str, list[str]]:
    """
    Returns {day_N: [activity_name, ...]} mapping.
    Falls back to rating-sorted split if LLM fails.
    """
    if not activities:
        return {f"day_{d}": [] for d in range(1, trip_days + 1)}

    # 1. חובה: מיון לפי דירוג כדי שה-25 שניקח יהיו הטובים ביותר
    sorted_activities = sorted(activities, key=lambda a: -a.get("rating", 0))

    # 2. הכיווץ החכם שלך (Payload Minification)
    compact_activities = [
        {
            "name": a.get("name"),
            "categories": a.get("categories"),
            "price": a.get("price"),
            "rating": a.get("rating"),
            "area": a.get("area"), # מצוין בשביל ה-Geographic Clustering!
            "best_time_of_day": a.get("best_time_of_day"),
            "food_available": a.get("food_available"),
        }
        for a in sorted_activities[:25]
    ]

    # 3. יצירת ההודעה הרזה והחסכונית בטוקנים
    user_msg = (
        f"Destination: {destination}\n"
        f"Trip duration: {trip_days} days\n"
        f"User preferences: {json.dumps(prefs, ensure_ascii=False)}\n\n"
        f"Available activities:\n"
        + json.dumps(compact_activities, ensure_ascii=False, indent=2)
    )

    try:
        # אפשר להשאיר את הגבלת הטוקנים של התשובה כדי להיות סופר-בטוחים
        small_llm = llm.bind(max_tokens=600)
        
        raw = small_llm.invoke([
            SystemMessage(content=SELECTOR_SYSTEM),
            HumanMessage(content=user_msg),
        ]).content.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip().rstrip("```").strip()

        result = json.loads(raw)
        if isinstance(result, dict):
            # Normalise keys → day_1, day_2, ...
            normalised: dict[str, list[str]] = {}
            for d in range(1, trip_days + 1):
                key = f"day_{d}"
                val = (
                    result.get(key)
                    or result.get(str(d))
                    or result.get(f"Day {d}")
                    or []
                )
                normalised[key] = val if isinstance(val, list) else []
            return normalised
    except Exception as e:
        print(f"DEBUG LLM Activity Selector failed: {e}")
        pass

    # ── Fallback: round-robin by rating ──
    names = [a["name"] for a in sorted_activities]
    result = {}
    for d in range(1, trip_days + 1):
        result[f"day_{d}"] = names[(d - 1) * 5: d * 5]
    return result


def resolve_candidates(
    activities: list[dict],
    day_names: list[str],
) -> list["ActivityCandidate"]:
    """
    Convert raw DB dicts to ActivityCandidate objects for the schedule engine.
    Preserves the LLM-chosen order (which encodes energy curve + geo cluster).
    """
    from agent.nodes.itinerary.schedule_engine import ActivityCandidate

    index = {a["name"]: a for a in activities}
    candidates = []
    for name in day_names:
        a = index.get(name)
        if not a:
            continue
        try:
            candidates.append(ActivityCandidate(
                name=a["name"],
                lat=float(a.get("latitude") or a.get("lat") or 0),
                lng=float(a.get("longitude") or a.get("lng") or 0),
                duration_minutes=int(a.get("avg_duration_minutes") or 90),
                price=float(a.get("price") or 0),
                opening_time=a.get("opening_time") or "08:00",
                closing_time=a.get("closing_time") or "21:00",
                food_available=bool(a.get("food_available")),
                categories=str(a.get("categories") or ""),
                rating=float(a.get("rating") or 0),
                requires_booking=bool(a.get("requires_booking")),
                operating_days=str(a.get("operating_days") or "Daily"),
            ))
        except (TypeError, ValueError) as e:
            pass
    return candidates
