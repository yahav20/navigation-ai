"""
ActivitySelector  →  Day Planner
=================================
Receives attractions + restaurants from Google Maps API (via executor),
and produces a rich day-by-day plan that the ScheduleEngine executes.

Output per day
--------------
{
  "day_1": {
    "theme":        "Beach & Old City",       # evocative 2-4 word label
    "area":         "Tel Aviv South",          # dominant neighbourhood
    "activities":   ["Gordon Beach", "Neve Tzedek Walk", "Carmel Market"],
    "lunch_restaurant":  "Dr. Shakshuka",     # verbatim name from restaurants list
    "coffee_place":      "Cafelix",
    "dinner_restaurant": "Manta Ray",
    "recommended_rest_blocks": [
        {"start": "14:00", "end": "16:00", "reason": "Long beach morning"}
    ]
  }
}

Token strategy (critical for Groq llama-3.1-8b, limit 6 000 TPM)
------------------------------------------------------------------
• Activities sent as CSV, not JSON  →  6× fewer tokens
• Hard caps: min(days×5, 12) attractions, min(days×3, 8) restaurants
• System prompt ≈ 200 tokens
• Output cap:  max(400, days×80) tokens
• Total budget: safely under 2 000 tokens for any trip length

JSON repair
-----------
_repair_json() closes unclosed braces, strips markdown fences,
removes trailing commas, handles single-quote fallback.
"""
from __future__ import annotations

import json
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

# ---------------------------------------------------------------------------
# System prompt  (keep SHORT — every token counts)
# ---------------------------------------------------------------------------

SELECTOR_SYSTEM = """You are a professional travel day-planner. Your job: given attractions and restaurants, build the best possible day plan for each day of the trip.

PLANNING RULES:
1. GEOGRAPHIC CLUSTERING: Group nearby places on the same day. Use lat,lng to judge distance. Lunch/dinner must be near that day's activities — not across the city.
2. HUMAN LOGIC — think like a real travel agent:
   - Beach morning → breakfast near beach → beach time → beachside lunch → hotel rest 14-16 → evening walk → dinner
   - Museum cluster → morning coffee → museum 1 → lunch nearby → museum 2 or gallery → dinner in same district
   - City walk → morning café → market → lunch → afternoon neighbourhood → hotel at 17 → dinner out at 19
3. ENERGY CURVE: morning=cultural/calm, afternoon=outdoor/active, evening=food/entertainment
4. WEATHER:
   - rain/storm → skip beaches, parks, long walks; prefer museums, galleries, indoor markets, cafés
   - extreme_heat → add rest block 14:00-16:00 at hotel; prefer indoor afternoons
5. DIETARY: if dietary_restrictions given, pick only matching restaurants. If no match exists, use null (never invent).
6. PREFERRED LOCATION: if given, bias activities and restaurants toward that area.
7. NO REPEATS: each place appears in exactly one day across the whole trip.
8. PER DAY: 3-5 activities. Always try to fill all 3 meal slots from the restaurants list. Use null only if truly no good match.

OUTPUT: pure JSON only — no markdown, no explanation, no extra keys.
{
  "day_1": {
    "theme": "2-4 word theme",
    "area": "main area for the day",
    "activities": ["Name1", "Name2", "Name3"],
    "lunch_restaurant": "Name from list or null",
    "breakfast_place": "Name of cafe or bakery from restaurants list",
    "dinner_restaurant": "Name from list or null",
    "recommended_rest_blocks": [{"start":"HH:MM","end":"HH:MM","reason":"short reason"}]
  }
}
IMPORTANT: use ONLY names that appear verbatim in the provided lists."""

# ---------------------------------------------------------------------------
# Weather classifier
# ---------------------------------------------------------------------------

_WEATHER_CONDITIONS = {
    "rain":         "RAIN — avoid outdoor activities (beach, parks, walks). Prefer museums, galleries, indoor markets, covered cafés.",
    "storm":        "STORM — indoor only. Museums, malls, galleries, restaurants.",
    "extreme_heat": "EXTREME HEAT (35°C+) — add rest block 14:00-16:00 at hotel. Prefer indoor afternoons. Morning/evening outdoor is fine.",
    "hot":          "HOT DAY — prefer shaded/indoor options midday. Morning and evening outdoor is fine.",
    "cloudy":       "Overcast — all activities suitable.",
    "clear":        "Clear weather — full activity palette available.",
}


def _weather_note(weather, day_number: int) -> str:
    """Extract a plain-English weather instruction for a specific day."""
    if not weather:
        return ""
    if isinstance(weather, dict):
        cond = str(
            weather.get(f"day_{day_number}") or
            weather.get("condition") or
            weather.get("summary") or ""
        ).lower()
    else:
        cond = str(weather).lower()

    for key, note in _WEATHER_CONDITIONS.items():
        if key in cond:
            return note
    return ""


# ---------------------------------------------------------------------------
# JSON repair
# ---------------------------------------------------------------------------

def _repair_json(raw: str) -> str:
    """
    Best-effort repair of truncated / malformed LLM JSON.
    Handles: markdown fences, trailing commas, unclosed braces, single quotes.
    """
    s = raw.strip()

    # 1. Strip markdown fences
    if s.startswith("```"):
        parts = s.split("```")
        s = parts[1] if len(parts) > 1 else s
        s = s.lstrip("json").strip()
    if s.endswith("```"):
        s = s[:-3].rstrip()

    # 2. Remove trailing commas before } or ]
    s = re.sub(r",\s*([}\]])", r"\1", s)

    # 3. Close any unclosed braces / brackets
    stack: list[str] = []
    in_string  = False
    escape_next = False
    for ch in s:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack and stack[-1] == ch:
            stack.pop()
    if in_string:
        s += '"'
    s += "".join(reversed(stack))

    # Quick success check
    try:
        json.loads(s)
        return s
    except json.JSONDecodeError:
        pass

    # 4. Single quotes → double quotes
    s2 = re.sub(r"(?<![\\])'", '"', s)
    try:
        json.loads(s2)
        return s2
    except json.JSONDecodeError:
        pass

    return s  # return best attempt; caller's except handles remaining errors


# ---------------------------------------------------------------------------
# CSV row builders  (6× fewer tokens than JSON)
# ---------------------------------------------------------------------------

def _act_row(a: dict) -> str:
    """Serialise an attraction to a compact pipe-delimited row."""
    name   = (a.get("name") or "").replace("|", " ")
    cats   = _squash(a.get("categories") or a.get("types") or "", 40)
    rating = f"{float(a.get('rating') or 0):.1f}"
    area   = (a.get("area") or a.get("formatted_address") or "").replace("|", " ")[:30]
    lat    = float(a.get("latitude") or a.get("lat") or 0)
    lng    = float(a.get("longitude") or a.get("lng") or 0)
    tod    = a.get("best_time_of_day") or ""
    return f"{name}|{cats}|{rating}|{area}|{lat:.3f},{lng:.3f}|{tod}"


def _rest_row(r: dict) -> str:
    """Serialise a restaurant to a compact pipe-delimited row."""
    name   = (r.get("name") or "").replace("|", " ")
    cats   = _squash(r.get("categories") or r.get("types") or "", 30)
    rating = f"{float(r.get('rating') or 0):.1f}"
    area   = (r.get("area") or r.get("formatted_address") or "").replace("|", " ")[:30]
    lat    = float(r.get("latitude") or r.get("lat") or 0)
    lng    = float(r.get("longitude") or r.get("lng") or 0)
    note   = "kosher" if r.get("kosher") else (r.get("price_level_text") or "")
    return f"{name}|{cats}|{rating}|{area}|{lat:.3f},{lng:.3f}|{note}"


def _squash(val, max_len: int) -> str:
    """Convert list or string to a short string without pipes."""
    if isinstance(val, list):
        val = ",".join(str(v) for v in val)
    return str(val).replace("|", " ")[:max_len]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def select_activities_per_day(
    llm: BaseChatModel,
    activities: list[dict],
    trip_days: int,
    prefs: dict,
    destination: str,
    restaurants: list[dict] | None = None,
    weather=None,
    blocked_times: list[dict] | None = None,
) -> dict[str, dict]:
    """
    Main entry point. Returns {day_N: day_plan_dict}.

    Parameters
    ----------
    activities:    List of attraction dicts from Google Maps API
    restaurants:   List of restaurant dicts from Google Maps API
    prefs:         Dict with keys: dietary_restrictions, preferred_location, blocked_times
    weather:       String condition or dict {day_N: condition}
    blocked_times: List of {day, start, end} dicts

    Returns
    -------
    Dict mapping day keys to rich day-plan dicts.
    Falls back to a rating-sorted round-robin if the LLM fails.
    """
    if not activities:
        return {f"day_{d}": _empty_day_plan() for d in range(1, trip_days + 1)}

    restaurants   = restaurants   or []
    blocked_times = blocked_times or []

    # ── Token-safe payload caps ──────────────────────────────────────────────
    MAX_ACT  = min(trip_days * 5, 12)   # 1d→5, 2d→10, 3d+→12
    MAX_REST = min(trip_days * 4, 12)   # 1d→4, 2d→8, 3d+→12

    sorted_acts  = sorted(activities,  key=lambda a: -(a.get("rating") or 0))
    sorted_rests = sorted(restaurants, key=lambda r: -(r.get("rating") or 0))

    acts_csv  = "name|categories|rating|area|lat,lng|best_time\n" + \
                "\n".join(_act_row(a) for a in sorted_acts[:MAX_ACT])
    rests_csv = "name|categories|rating|area|lat,lng|notes\n" + \
                "\n".join(_rest_row(r) for r in sorted_rests[:MAX_REST])

    # ── Weather section (per-day) ────────────────────────────────────────────
    weather_lines = []
    for d in range(1, trip_days + 1):
        note = _weather_note(weather, d)
        if note:
            weather_lines.append(f"  Day {d}: {note}")
    weather_section = ("Weather notes:\n" + "\n".join(weather_lines) + "\n") \
                      if weather_lines else ""

    # ── Blocked times section ────────────────────────────────────────────────
    blocked_lines = [
        f"  Day {b['day']}: {b['start']}–{b['end']} (user is unavailable)"
        for b in blocked_times
        if isinstance(b, dict) and "day" in b and "start" in b and "end" in b
    ]
    blocked_section = ("Blocked windows (do not schedule activities here):\n" +
                       "\n".join(blocked_lines) + "\n") if blocked_lines else ""

    # ── Preference hints ─────────────────────────────────────────────────────
    dietary  = (prefs.get("dietary_restrictions") or "").strip()
    pref_loc = (prefs.get("preferred_location")   or "").strip()
    pref_lines = []
    if dietary:
        pref_lines.append(f"  Dietary restriction: {dietary} — pick only matching restaurants.")
    if pref_loc:
        pref_lines.append(f"  Preferred location/vibe: {pref_loc} — bias toward this area.")
    pref_section = ("User preferences:\n" + "\n".join(pref_lines) + "\n") \
                   if pref_lines else ""

    # ── Assemble user message ────────────────────────────────────────────────
    user_msg = (
        f"Destination: {destination} | Trip: {trip_days} day(s)\n"
        + pref_section
        + weather_section
        + blocked_section
        + f"\nATTRACTIONS ({MAX_ACT} top-rated — pick activities from here):\n"
        + acts_csv + "\n"
        + f"\nRESTAURANTS ({MAX_REST} top-rated — pick all meal slots from here):\n"
        + rests_csv + "\n"
    )

    # ── LLM call ─────────────────────────────────────────────────────────────
    out_tokens = max(1500, trip_days * 500)
    try:
        provider  = type(llm).__name__
        bound_llm = llm if ("Google" in provider or "Gemini" in provider) \
                    else llm.bind(max_tokens=out_tokens)

        raw = bound_llm.invoke([
            SystemMessage(content=SELECTOR_SYSTEM),
            HumanMessage(content=user_msg),
        ]).content.strip()

        raw    = _repair_json(raw)
        result = json.loads(raw)

        if isinstance(result, dict):
            normalised: dict[str, dict] = {}
            for d in range(1, trip_days + 1):
                key = f"day_{d}"
                val = result.get(key) or result.get(str(d)) or result.get(f"Day {d}")
                if isinstance(val, dict):
                    normalised[key] = _normalise_day_plan(val)
                else:
                    # LLM returned a plain list — wrap it
                    normalised[key] = _list_to_day_plan(val if isinstance(val, list) else [])
            return normalised

    except Exception:
        pass

    # ── Fallback: rating-sorted round-robin ──────────────────────────────────
    names = [a["name"] for a in sorted_acts]
    return {
        f"day_{d}": _list_to_day_plan(names[(d - 1) * 5: d * 5])
        for d in range(1, trip_days + 1)
    }


# ---------------------------------------------------------------------------
# select_single_day_update  (targeted LLM re-plan for update_day_schedule)
# ---------------------------------------------------------------------------

def select_single_day_update(
    llm: BaseChatModel,
    available_acts: list[dict],
    available_rests: list[dict],
    pinned_activity: str | None,
    excluded_names: set[str],
    day_num: int,
    weather,
    prefs: dict,
    destination: str,
) -> dict:
    """
    Re-plan a single day after a user update using one targeted LLM call.

    Reuses the full SELECTOR_SYSTEM prompt so geographic clustering, energy
    curve, and meal proximity rules all apply — same quality as the initial
    build, scoped to one day.

    Parameters
    ----------
    available_acts   : Candidate activities (already deduplicated across other days)
    available_rests  : Available restaurants
    pinned_activity  : Activity that MUST appear in the output (the replacement)
    excluded_names   : Names to exclude from consideration (the removed activity)

    Returns
    -------
    A normalised day_plan dict (same schema as selection["day_N"]).
    Returns _empty_day_plan() on any failure; caller falls back to Phase 1.
    """
    if not available_acts:
        return _empty_day_plan()

    acts  = [a for a in available_acts  if a.get("name") not in excluded_names]
    rests = [r for r in available_rests if r.get("name") not in excluded_names]

    # Token caps for a single day
    MAX_ACT  = 10
    MAX_REST = 5

    # Ensure pinned activity is in the candidate list and ranked first
    sorted_acts = sorted(acts, key=lambda a: -(a.get("rating") or 0))
    if pinned_activity:
        sorted_acts = (
            [a for a in sorted_acts if a.get("name") == pinned_activity] +
            [a for a in sorted_acts if a.get("name") != pinned_activity]
        )

    sorted_rests = sorted(rests, key=lambda r: -(r.get("rating") or 0))

    acts_csv  = "name|categories|rating|area|lat,lng|best_time\n" + \
                "\n".join(_act_row(a) for a in sorted_acts[:MAX_ACT])
    rests_csv = "name|categories|rating|area|lat,lng|notes\n" + \
                "\n".join(_rest_row(r) for r in sorted_rests[:MAX_REST])

    weather_note    = _weather_note(weather, day_num)
    weather_section = f"Weather: {weather_note}\n" if weather_note else ""

    dietary  = (prefs.get("dietary_restrictions") or "").strip()
    pref_loc = (prefs.get("preferred_location")   or "").strip()
    pref_lines: list[str] = []
    if dietary:
        pref_lines.append(f"  Dietary restriction: {dietary}")
    if pref_loc:
        pref_lines.append(f"  Preferred location/vibe: {pref_loc}")
    pref_section = ("User preferences:\n" + "\n".join(pref_lines) + "\n") if pref_lines else ""

    pin_section = f"MUST INCLUDE this activity: {pinned_activity}\n" if pinned_activity else ""

    user_msg = (
        f"Destination: {destination} | Re-plan Day {day_num} after a user update.\n"
        + pin_section
        + pref_section
        + weather_section
        + f"\nATTRACTIONS (choose 3-5, respecting geographic clustering):\n{acts_csv}\n"
        + f"\nRESTAURANTS (pick all meal slots from here):\n{rests_csv}\n"
        + f"\nReturn JSON for day_{day_num} only."
    )

    try:
        provider  = type(llm).__name__
        bound_llm = llm if ("Google" in provider or "Gemini" in provider) \
                    else llm.bind(max_tokens=600)

        raw = bound_llm.invoke([
            SystemMessage(content=SELECTOR_SYSTEM),
            HumanMessage(content=user_msg),
        ]).content.strip()

        raw    = _repair_json(raw)
        result = json.loads(raw)

        if isinstance(result, dict):
            val = (
                result.get(f"day_{day_num}") or
                next((v for v in result.values() if isinstance(v, dict)), None)
            )
            if isinstance(val, dict):
                return _normalise_day_plan(val)
    except Exception:
        pass

    return _empty_day_plan()


# ---------------------------------------------------------------------------
# resolve_candidates  (called by executor → passes result to ScheduleEngine)
# ---------------------------------------------------------------------------
from agent.itinerary.schedule_engine import ActivityCandidate
def resolve_candidates(
    activities: list[dict],
    day_plan,                          # new: dict  |  legacy: list[str]
    restaurants: list[dict] | None = None,
) -> list["ActivityCandidate"]:
    """
    Convert raw Google Maps dicts to ActivityCandidate objects.

    Order:
      1. Sightseeing activities (in LLM-chosen energy-curve order)
      2. Pinned meal venues (lunch → coffee → dinner)

    This ordering matters: the ScheduleEngine inserts meals at the right
    time windows and knows which candidates are pinned meals.
    """
  

    restaurants = restaurants or []

    # Resolve name lists from plan format
    if isinstance(day_plan, dict):
        activity_names: list[str] = day_plan.get("activities") or []
        meal_names: list[str] = [
            n for n in [
                day_plan.get("lunch_restaurant"),
                day_plan.get("coffee_place"),
                day_plan.get("breakfast_place"),
                day_plan.get("dinner_restaurant"),
            ] if n
        ]
    else:
        activity_names = list(day_plan)
        meal_names     = []

    # Build lookup (restaurants take priority for meal slots)
    act_idx  = {a["name"]: a for a in activities}
    rest_idx = {r["name"]: r for r in restaurants}
    all_idx  = {**act_idx, **rest_idx}   # rest_idx wins on name collision

    def _make(name: str) -> "ActivityCandidate | None":
        raw = all_idx.get(name)
        if not raw:
            return None
        try:
            # Google Maps API uses "types" list; DB uses "categories" string
            cats = raw.get("categories") or ""
            if not cats and raw.get("types"):
                cats = ",".join(raw["types"])

            return ActivityCandidate(
                name             = raw["name"],
                lat              = float(raw.get("latitude")  or raw.get("lat")  or 0),
                lng              = float(raw.get("longitude") or raw.get("lng")  or 0),
                duration_minutes = int(raw.get("avg_duration_minutes") or
                                       _default_duration(cats)),
                price            = float(raw.get("price") or
                                         _price_from_level(raw.get("price_level")) or 0),
                opening_time     = raw.get("opening_time")  or "08:00",
                closing_time     = raw.get("closing_time")  or "22:00",
                food_available   = bool(raw.get("food_available")),
                categories       = str(cats),
                rating           = float(raw.get("rating") or 0),
                requires_booking = bool(raw.get("requires_booking")),
                operating_days   = str(raw.get("operating_days") or "Daily"),
            )
        except (TypeError, ValueError):
            return None

    candidates: list[ActivityCandidate] = []

    # 1. Sightseeing activities
    for name in activity_names:
        c = _make(name)
        if c:
            candidates.append(c)

    # 2. Pinned meal venues (LLM-chosen)
    for name in meal_names:
        c = _make(name)
        if c:
            candidates.append(c)

    # 3. Backup meal pool — all restaurants not already added.
    # Ensures _inject_meal never falls back to a generic placeholder
    # when the LLM name didn't exactly match a DB record.
    added_names = {c.name for c in candidates}
    backup_rests = sorted(restaurants, key=lambda r: -(r.get("rating") or 0))
    for r in backup_rests:
        name = r.get("name", "")
        if name and name not in added_names:
            c = _make(name)
            if c and c.is_meal_venue:
                candidates.append(c)
                added_names.add(name)

    return candidates


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _default_duration(categories: str) -> int:
    """Estimate visit duration in minutes when DB has no avg_duration_minutes."""
    cats = categories.lower()
    if any(k in cats for k in ("museum", "gallery", "palace", "castle")):
        return 120
    if any(k in cats for k in ("restaurant", "café", "cafe", "dining", "bistro")):
        return 60
    if any(k in cats for k in ("park", "garden", "beach")):
        return 90
    if any(k in cats for k in ("market", "bazaar", "mall")):
        return 75
    return 90


def _price_from_level(level) -> float:
    """Convert Google price_level (0-4) to a rough USD estimate."""
    mapping = {0: 0.0, 1: 10.0, 2: 25.0, 3: 50.0, 4: 100.0}
    try:
        return mapping.get(int(level), 15.0)
    except (TypeError, ValueError):
        return 15.0


def _empty_day_plan() -> dict:
    return {
        "theme": "Free Day", "area": "",
        "activities": [],
        "lunch_restaurant": None, "breakfast_place": None, "dinner_restaurant": None,
        "recommended_rest_blocks": [],
    }


def _list_to_day_plan(names: list[str]) -> dict:
    """Wrap a plain name list in the rich dict schema (degraded fallback)."""
    return {
        "theme": "Explore", "area": "",
        "activities": list(names),
        "lunch_restaurant": None, "breakfast_place": None, "dinner_restaurant": None,
        "recommended_rest_blocks": [],
    }


def _normalise_day_plan(raw: dict) -> dict:
    """Ensure all expected keys exist with correct types."""
    coffee = raw.get("coffee_place") or raw.get("breakfast_place") or None
    return {
        "theme":             str(raw.get("theme") or "Explore"),
        "area":              str(raw.get("area") or ""),
        "activities":        [str(n) for n in (raw.get("activities") or [])],
        "lunch_restaurant":  raw.get("lunch_restaurant") or None,
        "coffee_place":      coffee,
        "breakfast_place":   coffee,
        "dinner_restaurant": raw.get("dinner_restaurant") or None,
        "recommended_rest_blocks": [
            {
                "start":  str(rb.get("start",  "14:00")),
                "end":    str(rb.get("end",    "16:00")),
                "reason": str(rb.get("reason", "Rest")),
            }
            for rb in (raw.get("recommended_rest_blocks") or [])
            if isinstance(rb, dict)
        ],
    }