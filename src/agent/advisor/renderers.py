"""Deterministic section renderers for every advisor tool result.

No LLM is involved here. Each render_* function takes a typed result dict
and returns a formatted markdown string. The formatter assembles these into
a final response without exposing tool names or user messages to any LLM.
"""
from __future__ import annotations

import json

# ---------------------------------------------------------------------------
# Tool metadata — human-friendly labels and section titles
# ---------------------------------------------------------------------------

_TOOL_LABELS: dict[str, str] = {
    "get_best_time_to_visit":          "best time to visit",
    "get_average_weather":             "typical weather",
    "get_packing_list":                "packing list",
    "get_visa_requirements":           "visa requirements",
    "get_local_customs":               "local customs & etiquette",
    "get_currency_exchange":           "currency exchange rate",
    "get_travel_safety_info":          "travel safety",
    "get_city_overview":               "city overview",
    "get_trip_duration_advisor":       "recommended trip duration",
    "fetch_activities":                "top activities",
    "get_wikipedia_summary":           "destination info",
    "find_destinations_by_vibe":       "destination recommendations",
    "find_destinations_by_tag":        "destination recommendations",
    "get_reachable_destinations":      "reachable destinations",
    "find_destinations_within_budget": "destinations within budget",
    "find_destinations_within_budget_auto": "destinations within budget",
    "search_concerts":                 "upcoming concerts",
}

_TOOL_SECTION_TITLES: dict[str, str] = {
    "get_best_time_to_visit":          "When to Visit",
    "get_average_weather":             "Typical Weather",
    "get_packing_list":                "What to Pack",
    "get_visa_requirements":           "Visa Requirements",
    "get_local_customs":               "Local Customs & Etiquette",
    "get_currency_exchange":           "Currency Exchange",
    "get_travel_safety_info":          "Travel Safety",
    "get_city_overview":               "City Overview",
    "get_trip_duration_advisor":       "Trip Duration",
    "fetch_activities":                "Top Activities",
    "get_wikipedia_summary":           "About",
    "find_destinations_by_vibe":       "Matching Destinations",
    "find_destinations_by_tag":        "Matching Destinations",
    "get_reachable_destinations":      "Reachable Destinations",
    "find_destinations_within_budget": "Destinations Within Your Budget",
    "find_destinations_within_budget_auto": "Destinations Within Your Budget",
}


def get_section_title(tool_name: str) -> str:
    return _TOOL_SECTION_TITLES.get(tool_name, "")


def get_topic_labels(tool_names: list[str]) -> list[str]:
    """Return deduplicated human-friendly topic labels for the given tools."""
    seen: set[str] = set()
    labels: list[str] = []
    for t in tool_names:
        label = _TOOL_LABELS.get(t)
        if label and label not in seen:
            seen.add(label)
            labels.append(label)
    return labels

# ---------------------------------------------------------------------------
# Opener pools — indexed by response type
# ---------------------------------------------------------------------------

_OPENER_POOLS: dict[str, list[str]] = {
    "currency":       ["Here's the latest exchange rate!", "Let's look at the numbers!", "Here's your currency conversion!", "Let me pull up that rate for you."],
    "safety":         ["Here's the travel safety picture.", "Let me walk you through the safety overview.", "Here's what you should know before you go.", "Safety first — here's the rundown."],
    "visa":           ["Here's the visa situation for your trip.", "Let me break down the entry requirements.", "Here's what you need to know about getting in.", "Good question — here are the entry requirements."],
    "packing":        ["Here's your packing list!", "Let me put together a packing list for you.", "Here's what to pack!", "Packing sorted — here's your list."],
    "customs":        ["Here's the local customs guide.", "Let me walk you through the local etiquette.", "Here's what to know before you arrive.", "A few cultural notes to keep in mind."],
    "wikipedia":      ["Here's a quick overview!", "Here's what you should know!", "Let me share some background.", "Here's the rundown on that."],
    "discovery":      ["Let's find your perfect destination!", "I've got some great picks for you!", "Here are some destinations that fit!", "There are some fantastic choices here!", "Let me share some ideas!", "A few strong options come to mind!"],
    "city_info":      ["Here's everything you need to know!", "Great city to explore!", "Let me walk you through this destination.", "Here's a full overview!", "Here's what I found for you!"],
    "duration":       ["Here's the trip length advice!", "Let me give you the duration breakdown.", "Here's how long you should plan for."],
    "weather":        ["Here's the weather breakdown!", "Let me walk you through what to expect.", "Here's the climate picture.", "Here's what the weather looks like there."],
    "activities":     ["Here's what to do there!", "Plenty to keep you busy!", "Here are the top things to do!", "Here's your activity guide."],
    "concerts":       ["Let me check what's happening!", "Here's what I found on the live events front!", "Here's the concert info!", "Let me pull up those event listings."],
    "conversational": ["Of course!", "Happy to help with that!", "Let me pull that together for you.", "Sure — here's what I have."],
    "default":        ["Here's what I found!", "Happy to help!", "Let me share what I know.", "Here's the info you need."],
}

# Rotate through openers using turn_count to avoid repeats
def pick_opener(response_type: str, turn_count: int) -> str:
    pool = _OPENER_POOLS.get(response_type, _OPENER_POOLS["default"])
    return pool[turn_count % len(pool)]


# ---------------------------------------------------------------------------
# Closer templates — deterministic per response type
# ---------------------------------------------------------------------------

def build_closer(
    response_type: str,
    has_origin: bool,
    shown_cities: list[str],
    had_data: bool = True,
) -> str:
    # When we couldn't answer the question, don't pitch follow-ups
    if not had_data:
        return "Is there anything else I can help you with, or would you like to try a different destination?"

    city_hint = shown_cities[-1] if shown_cities else None

    if response_type == "discovery":
        if has_origin:
            return "Want to build a full itinerary around one of these destinations?"
        return "Would you like me to check for flights from your location?"

    if response_type == "city_info":
        suffix = f" for {city_hint}" if city_hint else ""
        return f"Want me to build a day-by-day itinerary{suffix}?"

    if response_type == "activities":
        suffix = f" in {city_hint}" if city_hint else ""
        return f"Want me to plan a full itinerary{suffix} using these activities?"

    if response_type == "concerts":
        return "Want me to search for flights around those dates, or build a full trip around the event?"

    if response_type in ("weather", "duration", "wikipedia"):
        return "Want me to plan a trip there, or do you have other questions?"

    # Type A practical tools
    return "Let me know if you have any other questions!"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bullet(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items if item)


def _section(title: str, body: str) -> str:
    return f"**{title}**\n{body}"


# ---------------------------------------------------------------------------
# Type A — Informational renderers
# ---------------------------------------------------------------------------

def render_currency_exchange(result: dict) -> str:
    if "error" in result:
        return f"Sorry, I couldn't fetch that rate: {result['error']}"

    from_c = result.get("from", "?")
    to_c   = result.get("to", "?")
    rate   = result.get("rate", "?")
    amount = float(result.get("amount", 1.0))
    converted = result.get("converted", "?")
    updated   = result.get("updated", "unknown")

    lines = [f"**1 {from_c} = {rate} {to_c}**"]
    if amount != 1.0:
        lines.append(f"**{amount:g} {from_c} = {converted} {to_c}**")
    lines.append(f"*(Rate last updated: {updated})*")
    return "\n\n".join(lines)


_SAFETY_ICONS = {1: "🟢", 2: "🟡", 3: "🔴", 4: "⛔"}


def render_travel_safety_info(result: dict) -> str:
    city       = result.get("city", "?")
    level      = result.get("level", 1)
    label      = result.get("label", "")
    notes      = result.get("notes", "")
    disclaimer = result.get("disclaimer", "")

    icon = _SAFETY_ICONS.get(level, "ℹ️")
    parts = [f"**{city} — {icon} Level {level}: {label}**"]
    if notes:
        parts.append(notes)
    if disclaimer:
        parts.append(f"*{disclaimer}*")
    return "\n\n".join(parts)


def render_visa_requirements(result: dict) -> str:
    # Error result — tool was called without passport_nationality
    if "message" in result:
        dest = result.get("destination", "your destination")
        return (
            f"To check visa requirements for **{dest}**, I need to know your passport nationality.\n\n"
            "Which country is your passport from?"
        )

    passport    = result.get("passport", "")
    destination = result.get("destination", "?")
    visa_req    = result.get("visa_required")
    details     = result.get("details", "")
    disclaimer  = result.get("disclaimer", "")

    # Missing or unknown passport nationality
    is_unknown = not passport or passport.lower() in ("unknown", "?", "")
    if is_unknown:
        return (
            f"To check visa requirements for **{destination}**, I need to know your passport nationality.\n\n"
            "Which country is your passport from?"
        )

    if visa_req is True:
        status = "**Visa required ✗**"
    elif visa_req is False:
        status = "**No visa required ✓**"
    else:
        status = "**Specific requirements not found — please verify at your nearest embassy**"

    parts = [
        f"**{passport} passport → {destination}**",
        status,
        details,
        f"*{disclaimer}*" if disclaimer else "",
    ]
    return "\n\n".join(p for p in parts if p)


def render_packing_list(result: dict) -> str:
    destination = result.get("destination", "?")
    season      = result.get("season", "?")
    trip_type   = result.get("trip_type", "city")
    trip_days   = result.get("trip_days", "?")

    sections = [f"**Packing list — {destination}, {season} ({trip_days} days, {trip_type} trip)**"]

    essentials = result.get("essentials") or []
    if essentials:
        sections.append(_section("Essentials", _bullet(essentials)))

    clothing = result.get("clothing") or []
    if clothing:
        sections.append(_section("Clothing", _bullet(clothing)))

    toiletries = result.get("toiletries") or []
    if toiletries:
        sections.append(_section("Toiletries", _bullet(toiletries)))

    extras = result.get("extras_for_trip_type") or []
    if extras:
        sections.append(_section(f"Extras — {trip_type} trip", _bullet(extras)))

    tip = result.get("tip")
    if tip:
        sections.append(f"💡 *{tip}*")

    return "\n\n".join(sections)


def render_local_customs(result: dict) -> str:
    destination = result.get("destination", "?")

    if "note" in result:
        parts = [f"**Local customs — {destination}**", result["note"]]
        tips = result.get("general_tips") or []
        if tips:
            parts.append(_section("General tips", _bullet(tips)))
        return "\n\n".join(parts)

    sections = [f"**Local customs & etiquette — {destination}**"]

    field_labels = [
        ("tipping",  "Tipping"),
        ("greetings","Greetings"),
        ("dress",    "Dress code"),
        ("dining",   "Dining"),
        ("customs",  "Local customs"),
    ]
    for field, label in field_labels:
        val = result.get(field)
        if val:
            sections.append(f"**{label}:** {val}")

    phrases = result.get("useful_phrases") or []
    if phrases:
        sections.append(_section("Useful phrases", _bullet(phrases)))

    return "\n\n".join(sections)


def render_wikipedia_summary(result: dict | str) -> str:
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except Exception:
            return result  # raw string fallback

    if "error" in result:
        return f"Couldn't find information: {result['error']}"

    topic   = result.get("topic", "?")
    summary = result.get("summary", "No summary available.")
    url     = result.get("url", "")

    parts = [f"**{topic}**", summary]
    if url:
        parts.append(f"[Read more on Wikipedia]({url})")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Type B — Destination advisory renderers
# ---------------------------------------------------------------------------

_FILTER_DISCOVERY_TOOLS = frozenset({"find_destinations_by_vibe", "find_destinations_by_tag"})
_ALL_DISCOVERY_TOOLS = frozenset({
    "find_destinations_by_vibe",
    "find_destinations_by_tag",
    "get_reachable_destinations",
    "find_destinations_within_budget",
    "find_destinations_within_budget_auto",
})


def _extract_city_names(result: object) -> set[str]:
    if not isinstance(result, list):
        return set()
    return {item["city"] for item in result if isinstance(item, dict) and "city" in item}


def compute_intersection(tool_results: list[dict]) -> set[str] | None:
    """Return the city intersection for multi-filter discovery runs, or None if not applicable."""
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


def render_destinations_list(tool_results: list[dict], intersection: set[str] | None = None) -> str:
    """Merge all discovery tool results into a single city list, filtered by intersection when present."""
    all_cities: dict[str, dict] = {}

    for tr in tool_results:
        result = tr.get("result", [])
        if not isinstance(result, list):
            continue
        for item in result:
            if not isinstance(item, dict) or "city" not in item:
                continue
            city = item["city"]
            if city not in all_cities:
                all_cities[city] = dict(item)
            else:
                # enrich with any new fields from this result
                all_cities[city].update({k: v for k, v in item.items() if k not in all_cities[city]})

    if intersection is not None:
        if intersection:
            display = {c: d for c, d in all_cities.items() if c in intersection}
            header = "*Destinations matching all your filters:*\n\n"
        else:
            # No perfect match — show best partial matches
            display = all_cities
            header = "*No destinations matched every filter exactly — here are the closest options:*\n\n"
    else:
        display = all_cities
        header = ""

    if not display:
        return "No matching destinations found for your criteria."

    lines: list[str] = []
    for city, data in display.items():
        country    = data.get("country", "")
        city_line  = f"**{city}**" + (f", {country}" if country else "")
        details: list[str] = []

        if "cheapest_flight" in data:
            details.append(f"Flight from ~${data['cheapest_flight']:.0f}")
        elif "min_price" in data:
            details.append(f"Flight from ~${data['min_price']:.0f}")

        if "cheapest_hotel_per_night" in data:
            details.append(f"Hotel from ~${data['cheapest_hotel_per_night']:.0f}/night")

        if "estimated_min_total" in data:
            details.append(f"Est. total ~${data['estimated_min_total']:.0f}")

        if "recommended_days" in data:
            details.append(f"~{data['recommended_days']} day stay")

        if "min_duration_hours" in data:
            h = float(data["min_duration_hours"])
            details.append(f"~{h:.1f}h flight")

        if "activity_count" in data and not details:
            details.append(f"{data['activity_count']} activities")

        city_line += "\n" + " | ".join(details) if details else ""
        lines.append(city_line)

    return header + "\n\n".join(lines)


def render_city_overview(result: dict) -> str:
    if "message" in result:
        return result["message"]

    city     = result.get("city", "?")
    sections = [f"**{city}**"]

    activity_categories = result.get("activity_categories") or []
    if activity_categories:
        act_lines = []
        for cat in activity_categories:
            cat_name   = cat.get("category", "")
            activities = cat.get("activities", "")
            count      = cat.get("count", "")
            if activities:
                act_lines.append(f"- **{cat_name}** ({count}): {activities}")
            else:
                act_lines.append(f"- **{cat_name}** ({count})")
        sections.append(_section("Things to do", "\n".join(act_lines)))

    best_time = result.get("best_time_to_visit") or {}
    if best_time.get("months"):
        months_str = ", ".join(best_time["months"])
        reason     = best_time.get("reason", "")
        btt_body   = f"**{months_str}**"
        if reason:
            btt_body += f" — {reason}"
        sections.append(_section("Best time to visit", btt_body))

    weather = result.get("weather_by_season") or {}
    if weather:
        weather_lines = [f"- **{season}**: {temp}" for season, temp in weather.items()]
        sections.append(_section("Average temperatures", "\n".join(weather_lines)))

    return "\n\n".join(sections)


def render_trip_duration(result: dict | list) -> str:
    entries = result if isinstance(result, list) else [result]
    lines: list[str] = []

    for entry in entries:
        if "message" in entry:
            lines.append(entry["message"])
            continue
        city  = entry.get("city", "?")
        min_d = entry.get("min_days", "?")
        max_d = entry.get("max_days", "?")
        notes = entry.get("notes", "")
        line  = f"**{city}**: {min_d}–{max_d} days"
        if notes:
            line += f"\n{notes}"
        lines.append(line)

    return "\n\n".join(lines)


def render_best_time(result: dict, args: dict) -> str:
    if "message" in result:
        return result["message"]

    city   = args.get("city", "")
    months = result.get("months") or []
    reason = result.get("reason", "")

    months_str = ", ".join(months) if months else "No data available"
    city_str   = f" {city}" if city else ""
    parts      = [f"**Best months to visit{city_str}:** {months_str}"]
    if reason:
        parts.append(reason)
    return "\n\n".join(parts)


def render_average_weather(result: dict, args: dict) -> str:
    if "message" in result:
        return result["message"]

    # Single-season result
    if "season" in result and "temperature" in result:
        city   = args.get("city", "")
        season = result["season"]
        temp   = result["temperature"]
        city_str = f" in {city}" if city else ""
        return f"**{season}{city_str}:** {temp}"

    # All-seasons result
    if result.get("available") and "weather_by_season" in result:
        city  = result.get("city") or args.get("city", "?")
        lines = [f"**Average weather in {city}:**"]
        for entry in result["weather_by_season"]:
            lines.append(f"- **{entry['season']}**: {entry['temperature']}")
        return "\n".join(lines)

    return "No weather data available."


def render_activities(result: list) -> str:
    if not result:
        return "No activities found for this city."

    by_category: dict[str, list[dict]] = {}
    for act in result:
        raw_cats = act.get("categories", "[]")
        try:
            cats = json.loads(raw_cats) if isinstance(raw_cats, str) else raw_cats
        except Exception:
            cats = []
        for cat in (cats or ["Other"]):
            by_category.setdefault(cat, []).append(act)

    sections: list[str] = []
    for cat, acts in by_category.items():
        lines = [f"**{cat}**"]
        for act in acts[:6]:
            name     = act.get("name", "?")
            price    = act.get("price")
            duration = act.get("avg_duration_minutes")
            rating   = act.get("rating")

            parts: list[str] = []
            if price is not None:
                parts.append(f"${price:.0f}" if price > 0 else "Free")
            if duration:
                parts.append(f"{duration / 60:.1f}h")
            if rating:
                parts.append(f"⭐ {rating:.1f}")

            detail = " | ".join(parts)
            lines.append(f"- **{name}**" + (f" — {detail}" if detail else ""))
        sections.append("\n".join(lines))

    return "\n\n".join(sections)
