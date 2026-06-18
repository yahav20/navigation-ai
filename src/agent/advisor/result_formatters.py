"""Convert typed tool results into clean labeled text blocks for the LLM formatter.

Each function strips the tool name and renders the result data as human-readable
key-value lines. The LLM formatter reads these blocks and writes the natural response.
No markdown decisions here — that belongs to the LLM.
"""
from __future__ import annotations

import json
from typing import Any

# ---------------------------------------------------------------------------
# Discovery tool sets
# ---------------------------------------------------------------------------

_FILTER_DISCOVERY_TOOLS = frozenset({"find_destinations_by_vibe", "find_destinations_by_tag"})

ALL_DISCOVERY_TOOLS = frozenset({
    "find_destinations_by_vibe",
    "find_destinations_by_tag",
    "get_reachable_destinations",
    "find_destinations_within_budget",
    "find_destinations_within_budget_auto",
})

# ---------------------------------------------------------------------------
# Intersection utility
# ---------------------------------------------------------------------------

def _extract_city_names(result: Any) -> set[str]:
    if not isinstance(result, list):
        return set()
    return {item["city"] for item in result if isinstance(item, dict) and "city" in item}


def compute_intersection(tool_results: list[dict]) -> set[str] | None:
    """Return the city intersection for multi-filter discovery runs, or None."""
    filter_results = [
        tr for tr in tool_results
        if tr["tool_name"] in _FILTER_DISCOVERY_TOOLS
        and isinstance(tr.get("result"), list)
        and tr["result"]
        and "message" not in tr["result"][0]
    ]
    if len(filter_results) < 2:
        return None
    city_sets = [_extract_city_names(tr["result"]) for tr in filter_results]
    return city_sets[0].intersection(*city_sets[1:])


# ---------------------------------------------------------------------------
# Human-friendly section headers (no tool names exposed)
# ---------------------------------------------------------------------------

def _header(tool_name: str, args: dict) -> str:
    city    = args.get("city") or args.get("destination") or ""
    origin  = args.get("origin", "")
    season  = args.get("season", "")
    topic   = args.get("topic", "")
    artist  = args.get("artist", "")
    month   = args.get("month", "")

    mapping = {
        "get_currency_exchange":           "Currency Exchange Rate",
        "get_travel_safety_info":          f"Travel Safety{' — ' + city if city else ''}",
        "get_visa_requirements":           f"Visa Requirements{' — ' + city if city else ''}",
        "get_packing_list":                f"Packing List{' — ' + city if city else ''}{', ' + season if season else ''}",
        "get_local_customs":               f"Local Customs & Etiquette{' — ' + city if city else ''}",
        "get_wikipedia_summary":           f"About: {topic}" if topic else "Reference Info",
        "get_city_overview":               f"City Overview — {city}" if city else "City Overview",
        "get_trip_duration_advisor":       f"Recommended Duration — {city}" if city else "Recommended Duration",
        "fetch_activities":                f"Activities in {city}" if city else "Activities",
        "get_best_time_to_visit":          f"Best Time to Visit{' ' + city if city else ''}",
        "get_average_weather":             f"Weather{' — ' + city if city else ''}{', ' + season if season else ''}",
        "find_destinations_by_vibe":       "Destinations by Activity Type",
        "find_destinations_by_tag":        "Destinations by Travel Style",
        "get_reachable_destinations":      f"Destinations Reachable from {origin}" if origin else "Reachable Destinations",
        "find_destinations_within_budget": f"Destinations Within Budget from {origin}" if origin else "Budget-Friendly Destinations",
        "find_destinations_within_budget_auto": f"Destinations Within Budget from {origin}" if origin else "Budget-Friendly Destinations",
        "search_concerts":                 "Concert Events" + (
            " — " + ", ".join(filter(None, [artist, city, month]))
            if any([artist, city, month]) else ""
        ),
    }
    return mapping.get(tool_name, tool_name.replace("_", " ").title())


# ---------------------------------------------------------------------------
# Per-tool result body formatters
# ---------------------------------------------------------------------------

def _fmt_list(items: list) -> str:
    return ", ".join(str(i) for i in items) if items else "—"


def _fmt_currency(result: dict, _args: dict) -> str:
    if "error" in result:
        return f"error: {result['error']}"
    lines = [
        f"from: {result.get('from', '?')}",
        f"to: {result.get('to', '?')}",
        f"rate: {result.get('rate', '?')}",
    ]
    amount = float(result.get("amount", 1.0))
    if amount != 1.0:
        lines.append(f"converted: {result.get('converted', '?')} (for {amount:g} {result.get('from', '?')})")
    lines.append(f"updated: {result.get('updated', 'unknown')}")
    return "\n".join(lines)


def _fmt_safety(result: dict, _args: dict) -> str:
    if "message" in result:
        return result["message"]
    return "\n".join([
        f"city: {result.get('city', '?')}",
        f"level: {result.get('level', '?')} ({result.get('label', '')})",
        f"notes: {result.get('notes', '')}",
        f"disclaimer: {result.get('disclaimer', '')}",
    ])


def _fmt_visa(result: dict, _args: dict) -> str:
    if "message" in result:
        return "status: Passport nationality not provided — ask the user which country their passport is from."
    passport = result.get("passport", "")
    if not passport or passport.lower() in ("unknown", "?", ""):
        dest = result.get("destination", "the destination")
        return f"status: Passport nationality unknown — ask the user which country their passport is from before answering about {dest}."
    visa_req = result.get("visa_required")
    return "\n".join(filter(None, [
        f"passport: {passport}",
        f"destination: {result.get('destination', '?')}",
        f"visa_required: {'Yes' if visa_req is True else 'No' if visa_req is False else 'Unknown'}",
        f"details: {result.get('details', '')}",
        f"disclaimer: {result.get('disclaimer', '')}",
    ]))


def _fmt_packing(result: dict, _args: dict) -> str:
    if "message" in result:
        return result["message"]
    lines = [
        f"destination: {result.get('destination', '?')}",
        f"season: {result.get('season', '?')}",
        f"climate: {result.get('climate', '?')}",
        f"trip_days: {result.get('trip_days', '?')}",
        f"trip_type: {result.get('trip_type', '?')}",
        f"essentials: {_fmt_list(result.get('essentials') or [])}",
        f"clothing: {_fmt_list(result.get('clothing') or [])}",
        f"toiletries: {_fmt_list(result.get('toiletries') or [])}",
        f"extras: {_fmt_list(result.get('extras_for_trip_type') or [])}",
    ]
    tip = result.get("tip")
    if tip:
        lines.append(f"tip: {tip}")
    return "\n".join(lines)


def _fmt_customs(result: dict, _args: dict) -> str:
    if "note" in result:
        tips = _fmt_list(result.get("general_tips") or [])
        return result["note"] + ("\ntips: " + tips if tips else "")
    lines = []
    for field in ("tipping", "greetings", "dress", "dining", "customs"):
        val = result.get(field)
        if val:
            lines.append(f"{field}: {val}")
    phrases = result.get("useful_phrases") or []
    if phrases:
        lines.append(f"useful_phrases: {_fmt_list(phrases)}")
    return "\n".join(lines)


def _fmt_wikipedia(result: dict | str, _args: dict) -> str:
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except Exception:
            return str(result)
    if "error" in result:
        return f"error: {result['error']}"
    lines = [f"summary: {result.get('summary', '')}"]
    url = result.get("url", "")
    if url:
        lines.append(f"source: {url}")
    return "\n".join(lines)


def _fmt_city_overview(result: dict, _args: dict) -> str:
    if "message" in result:
        return f"status: {result['message']}"
    lines = []
    for cat in (result.get("activity_categories") or []):
        name  = cat.get("category", "")
        acts  = cat.get("activities", "")
        count = cat.get("count", "")
        lines.append(f"activities_{name.lower()}: {acts} ({count} total)")
    best = result.get("best_time_to_visit") or {}
    if best.get("months"):
        lines.append(f"best_months: {', '.join(best['months'])}")
    if best.get("reason"):
        lines.append(f"best_months_reason: {best['reason']}")
    for season, temp in (result.get("weather_by_season") or {}).items():
        lines.append(f"weather_{season.lower()}: {temp}")
    return "\n".join(lines) if lines else "status: No city data available"


def _fmt_trip_duration(result: dict | list, _args: dict) -> str:
    entries = result if isinstance(result, list) else [result]
    lines = []
    for e in entries:
        if "message" in e:
            lines.append(e["message"])
        else:
            notes = e.get("notes", "")
            line  = f"{e.get('city', '?')}: {e.get('min_days', '?')}–{e.get('max_days', '?')} days"
            if notes:
                line += f" — {notes}"
            lines.append(line)
    return "\n".join(lines)


def _fmt_best_time(result: dict, _args: dict) -> str:
    if "message" in result:
        return result["message"]
    months = result.get("months") or []
    reason = result.get("reason", "")
    return ("months: " + ", ".join(months)) + (f"\nreason: {reason}" if reason else "")


def _fmt_weather(result: dict, _args: dict) -> str:
    if "message" in result:
        return result["message"]
    if "season" in result and "temperature" in result:
        return f"season: {result['season']}\ntemperature: {result['temperature']}"
    if result.get("available") and "weather_by_season" in result:
        return "\n".join(f"{e['season']}: {e['temperature']}" for e in result["weather_by_season"])
    return "status: No weather data available"


def _fmt_activities(result: list, _args: dict) -> str:
    if not result:
        return "status: No activities found"
    lines = []
    for act in result[:20]:
        name     = act.get("name", "?")
        price    = act.get("price")
        duration = act.get("avg_duration_minutes")
        rating   = act.get("rating")
        raw_cats = act.get("categories", "[]")
        try:
            cats = json.loads(raw_cats) if isinstance(raw_cats, str) else raw_cats
        except Exception:
            cats = []
        parts: list[str] = []
        if price is not None:
            parts.append(f"${price:.0f}" if price > 0 else "Free")
        if duration:
            parts.append(f"{duration / 60:.1f}h")
        if rating:
            parts.append(f"rating {rating:.1f}")
        if cats:
            parts.append(f"[{', '.join(cats)}]")
        lines.append(f"- {name}: {' | '.join(parts)}" if parts else f"- {name}")
    return "\n".join(lines)


def _fmt_destinations(items: list, _args: dict) -> str:
    if not items:
        return "status: No destinations found"
    if len(items) == 1 and "message" in items[0]:
        return f"status: {items[0]['message']}"
    lines = []
    for item in items:
        if not isinstance(item, dict) or "city" not in item:
            continue
        city    = item["city"]
        country = item.get("country", "")
        parts   = [f"{city}{', ' + country if country else ''}"]
        if "cheapest_flight" in item:
            parts.append(f"flight ~${item['cheapest_flight']:.0f}")
        elif "min_price" in item:
            parts.append(f"flight ~${item['min_price']:.0f}")
        if "cheapest_hotel_per_night" in item:
            parts.append(f"hotel ~${item['cheapest_hotel_per_night']:.0f}/night")
        if "estimated_min_total" in item:
            parts.append(f"total ~${item['estimated_min_total']:.0f}")
        if "recommended_days" in item:
            parts.append(f"{item['recommended_days']} days")
        if "min_duration_hours" in item:
            parts.append(f"{float(item['min_duration_hours']):.1f}h flight")
        if "activity_count" in item:
            parts.append(f"{item['activity_count']} activities")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_FORMATTERS: dict[str, Any] = {
    "get_currency_exchange":            _fmt_currency,
    "get_travel_safety_info":           _fmt_safety,
    "get_visa_requirements":            _fmt_visa,
    "get_packing_list":                 _fmt_packing,
    "get_local_customs":                _fmt_customs,
    "get_wikipedia_summary":            _fmt_wikipedia,
    "get_city_overview":                _fmt_city_overview,
    "get_trip_duration_advisor":        _fmt_trip_duration,
    "fetch_activities":                 _fmt_activities,
    "get_best_time_to_visit":           _fmt_best_time,
    "get_average_weather":              _fmt_weather,
    "find_destinations_by_vibe":        _fmt_destinations,
    "find_destinations_by_tag":         _fmt_destinations,
    "get_reachable_destinations":       _fmt_destinations,
    "find_destinations_within_budget":  _fmt_destinations,
    "find_destinations_within_budget_auto": _fmt_destinations,
}

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def format_results_for_llm(tool_results: list[dict]) -> str:
    """Convert all tool results to a single clean data block for the LLM formatter."""
    blocks: list[str] = []

    # Group all discovery tools into one merged destination block
    discovery_results = [tr for tr in tool_results if tr["tool_name"] in ALL_DISCOVERY_TOOLS]
    discovery_seen: set[str] = set()

    if discovery_results:
        # Merge all destination dicts across discovery tools
        merged: dict[str, dict] = {}
        for tr in discovery_results:
            result = tr.get("result") or []
            if isinstance(result, list):
                for item in result:
                    if isinstance(item, dict) and "city" in item:
                        city = item["city"]
                        if city not in merged:
                            merged[city] = dict(item)
                        else:
                            merged[city].update({k: v for k, v in item.items() if k not in merged[city]})

        # Apply intersection filter when multiple filter-discovery tools ran
        intersection = compute_intersection(tool_results)
        if intersection is not None:
            if intersection:
                merged   = {c: d for c, d in merged.items() if c in intersection}
                note     = f"(showing {len(merged)} destinations matching ALL filters)\n"
            else:
                note     = "(no destinations matched all filters — showing closest matches per filter)\n"
        else:
            note = ""

        header = _header(discovery_results[0]["tool_name"], discovery_results[0].get("args", {}))
        body   = _fmt_destinations(list(merged.values()), {})
        blocks.append(f"[{header}]\n{note}{body}")
        discovery_seen = {tr["tool_name"] for tr in discovery_results}

    # All other tools in execution order
    for tr in tool_results:
        tn     = tr["tool_name"]
        result = tr.get("result")
        args   = tr.get("args", {})

        if tn in discovery_seen:
            continue
        formatter = _FORMATTERS.get(tn)
        if formatter is None:
            continue

        header = _header(tn, args)
        body   = formatter(result, args)
        blocks.append(f"[{header}]\n{body}")

    return "\n\n".join(blocks)


def closer_guidance_for(tool_names: set[str], has_origin: bool) -> str:
    """Return a one-sentence instruction for the LLM about what closer to write."""
    if "search_concerts" in tool_names:
        return "Offer to search for flights around the event dates or build a full trip around the event."
    city_tools = {"get_city_overview", "fetch_activities", "get_best_time_to_visit",
                  "get_average_weather", "get_trip_duration_advisor"}
    if tool_names & city_tools:
        return "Offer to build a day-by-day itinerary for the destination."
    discovery_tools = ALL_DISCOVERY_TOOLS
    if tool_names & discovery_tools:
        if has_origin:
            return "Offer to build a full itinerary around one of the recommended destinations."
        return "Ask if they'd like you to check for flights from their location."
    return "Offer to help with anything else they need for their trip."
