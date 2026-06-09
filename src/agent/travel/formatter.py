"""Render the curated TravelPlan as a Markdown message for the user."""

from langchain_core.messages import AIMessage

from agent.core.state import AgentState


class FormatterNode:
    """Convert the structured TravelPlan in state into a Markdown AIMessage."""

    def __call__(self, state: AgentState) -> dict:
        plan = state.get("travel_plan")
        if not plan:
            return {}
        content = _render_travel_plan(plan)
        msg = AIMessage(content=content)
        msg.name = "formatter_output"
        return {"messages": [msg]}


# ── Rendering helpers ──────────────────────────────────────────────────────────

def _render_travel_plan(plan: dict) -> str:
    sections: list[str] = []

    intro = plan.get("intro", "")
    if intro:
        sections.append(intro)

    sections.append(_render_header(plan))

    flight_pairings = plan.get("flight_pairings") or []
    if flight_pairings:
        sections.append(_render_flights(flight_pairings))

    hotels = plan.get("hotels") or []
    if hotels:
        sections.append(_render_hotels(hotels))

    activities = plan.get("activities") or []
    if activities:
        sections.append(_render_activities(activities))

    restaurants = plan.get("restaurants") or []
    if restaurants:
        sections.append(_render_restaurants(restaurants))

    weather = plan.get("weather") or {}
    best_time = plan.get("best_time") or {}
    if weather or best_time:
        sections.append(_render_insights(weather, best_time))

    sections.append(
        "**Ready to turn this into a real plan?**\n"
        'If these flights and hotels fit your budget, just tell me: **"Build my daily schedule"** '
        "(or ask to change the budget, dates, or destination!)"
    )

    return "\n\n".join(sections)


def _render_header(plan: dict) -> str:
    destination = plan.get("destination", "")
    lines = [f"### ✨ **Your {destination} Escape** ✨", ""]
    lines.append(f"**Origin:** {plan.get('origin', '')}")
    lines.append(f"**Destination:** {destination}")
    lines.append(f"**Trip Days:** {plan.get('trip_days', '')} days")
    if plan.get("trip_start"):
        lines.append(f"**Approximate Start:** {plan['trip_start']}")
    if plan.get("total_budget") is not None:
        lines.append(f"**Total Budget:** ${float(plan['total_budget']):.0f}")
    if plan.get("lowest_total_estimate") is not None:
        lines.append(f"**Lowest Total Estimate:** ${float(plan['lowest_total_estimate']):.0f}")
    return "\n".join(lines)


def _render_flights(pairings: list) -> str:
    lines = ["### ✈️ **Flight Options** (3 round-trip pairings)", ""]
    for i, pairing in enumerate(pairings, 1):
        total = float(pairing.get("total_price", 0) or 0)
        lines.append(f"#### **Option {i}** — **Total: ${total:.0f}**")
        desc = pairing.get("description", "")
        if desc:
            lines.append(desc)
        for direction_label, key in [("Outbound", "outbound"), ("Return", "return_flight")]:
            flight = pairing.get(key) or {}
            if flight:
                lines.append(_render_flight_bullet(direction_label, flight))
        if i < len(pairings):
            lines.append("")
    lines.append("\n---")
    return "\n".join(lines)


def _render_flight_bullet(direction_label: str, f: dict) -> str:
    airline  = f.get("airline", "")
    label    = f.get("label", "")
    price    = float(f.get("price", 0) or 0)
    stops    = f.get("stops")
    duration = f.get("duration_minutes")

    stops_badge = "Direct"
    if stops is not None:
        s = int(stops)
        if s == 1:
            stops_badge = "1 stop"
        elif s > 1:
            stops_badge = f"{s} stops"

    parts = [f"* **{direction_label}**"]
    name_part = " — ".join(filter(None, [airline, label]))
    if name_part:
        parts.append(f"| **{name_part}**")
    parts.append(f"| **Price:** ${price:.0f}")
    parts.append(f"| **{stops_badge}**")
    if duration is not None:
        h, m = divmod(int(duration), 60)
        parts.append(f"| **Duration:** {h}h {m:02d}m")

    bullet = " ".join(parts)
    sub: list[str] = []

    dep = f.get("departure_time", "")
    if dep:
        dep_fmt = _fmt_iso(dep)
        dest_airport = f.get("destination_airport", "")
        dep_line = f"  * **Departs:** {dep_fmt}"
        if dest_airport:
            dep_line += f" — arrives {dest_airport}"
        sub.append(dep_line)

    stop_airports = f.get("stop_airports") or []
    if stop_airports:
        sub.append(f"  * **Via:** {' → '.join(stop_airports)}")

    legs = f.get("legs") or []
    for j, leg in enumerate(legs, 1):
        fc = leg.get("from_city", "")
        tc = leg.get("to_city", "")
        la = leg.get("airline", "")
        fn = leg.get("flight_number", "")
        sub.append(f"  * **Leg {j}:** {fc} ➔ {tc} | **Airline:** {la} (**Flight:** {fn})")

    return "\n".join([bullet] + sub)


def _render_hotels(hotels: list) -> str:
    lines = ["### 🏨 **Hotels**", ""]
    for i, h in enumerate(hotels, 1):
        name  = h.get("name", "")
        stars = h.get("stars")
        price = float(h.get("price_per_night", 0) or 0)
        desc  = h.get("description", "")
        star_str = f" ({int(stars)} ⭐)" if stars is not None else ""
        lines.append(f"**{i}. {name}**{star_str}")
        lines.append(f"* **Price Per Night:** ${price:.0f}")
        if desc:
            lines.append(f"* {desc}")
        lines.append("")
    lines.append("---")
    return "\n".join(lines)


def _render_activities(activities: list) -> str:
    lines = ["### 🎯 **Activities**", ""]
    for a in activities:
        name = a.get("name", "")
        desc = a.get("description", "")
        lines.append(f"* **{name}** — {desc}")
    lines.append("\n---")
    return "\n".join(lines)


def _render_restaurants(restaurants: list) -> str:
    lines = ["### 🍽️ **Restaurants**", ""]
    for i, r in enumerate(restaurants, 1):
        name       = r.get("name", "")
        price_tier = r.get("price_tier", "")
        rating     = r.get("rating")
        desc       = r.get("description", "")
        meta_parts = []
        if price_tier:
            meta_parts.append(f"({price_tier})")
        if rating is not None:
            meta_parts.append(f"{rating}★")
        meta = " ".join(meta_parts)
        lines.append(f"**{i}. {name}**" + (f" {meta}" if meta else ""))
        if desc:
            lines.append(f"* {desc}")
        lines.append("")
    lines.append("---")
    return "\n".join(lines)


def _render_insights(weather: dict, best_time: dict) -> str:
    lines = ["### 🌤️ **Destination Insights**", ""]
    if best_time:
        months = ", ".join(best_time.get("months") or [])
        reason = best_time.get("reason", "")
        bt_line = f"* **Best Time to Visit:** {months}"
        if reason:
            bt_line += f" — {reason}"
        lines.append(bt_line)
    if weather:
        lines.append("* **Average Weather:**")
        for season in ("Spring", "Summer", "Autumn", "Winter"):
            val = weather.get(season)
            if val:
                lines.append(f"  * **{season}:** {val}")
    return "\n".join(lines)


def _fmt_iso(iso: str) -> str:
    if not iso:
        return ""
    try:
        if "T" in iso:
            date_part, time_part = iso.split("T", 1)
            return f"{date_part} at {time_part[:5]}"
        return iso[:5]
    except Exception:
        return iso
