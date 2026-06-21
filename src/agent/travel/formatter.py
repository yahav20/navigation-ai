"""Emit the curated TravelPlan as a generative-UI message for TravelPlanViewer."""

from langchain_core.messages import AIMessage

from agent.core.state import AgentState


class FormatterNode:
    """Emit a TravelPlanViewer UI message from the structured TravelPlan in state."""

    def __call__(self, state: AgentState) -> dict:
        plan = state.get("travel_plan")
        if not plan:
            return {}

        ui_props   = _build_viewer_props(plan)
        # Web renders `ui_props` as a TravelPlanViewer card and hides the text
        # on a message that has a viewer attached, but the CLI has no such
        # renderer — so the message content must stand on its own as a plain
        # markdown summary, not be left blank.
        ai_message = AIMessage(content=_render_markdown(plan), name="formatter_output")
        ui_message = {
            "type":     "ui",
            "name":     "TravelPlanViewer",
            "props":    ui_props,
            "metadata": {"message_id": ai_message.id},
        }
        return {"messages": [ai_message], "ui": [ui_message]}


# ── Plain-text fallback (CLI has no TravelPlanViewer renderer) ─────────────

def _render_markdown(plan: dict) -> str:
    """Render the curated TravelPlan as markdown, for clients with no UI renderer."""
    lines: list[str] = [plan.get("intro", "").strip()]

    for i, pairing in enumerate(plan.get("flight_pairings") or [], start=1):
        outbound = pairing.get("outbound") or {}
        ret      = pairing.get("return_flight") or {}
        lines.append(
            f"\n**Option {i} — ${pairing.get('total_price', 0):,.0f}**\n"
            f"- Outbound: {outbound.get('label', 'flight')} "
            f"({outbound.get('airline', '')}) — ${outbound.get('price', 0):,.0f}\n"
            f"- Return: {ret.get('label', 'flight')} "
            f"({ret.get('airline', '')}) — ${ret.get('price', 0):,.0f}\n"
            f"- {pairing.get('description', '')}"
        )

    if plan.get("hotels"):
        lines.append("\n**Hotels:**")
        for hotel in plan["hotels"]:
            lines.append(
                f"- {hotel.get('name', 'Hotel')} — ${hotel.get('price_per_night', 0):,.0f}/night "
                f"— {hotel.get('description', '')}"
            )

    if plan.get("activities"):
        lines.append("\n**Activities:**")
        for activity in plan["activities"]:
            lines.append(f"- {activity.get('name', '')} — {activity.get('description', '')}")

    if plan.get("restaurants"):
        lines.append("\n**Restaurants:**")
        for restaurant in plan["restaurants"]:
            lines.append(f"- {restaurant.get('name', '')} — {restaurant.get('description', '')}")

    estimate = plan.get("lowest_group_estimate") or plan.get("lowest_total_estimate")
    if estimate is not None:
        lines.append(f"\nEstimated total: **${estimate:,.0f}**")

    lines.append(("\n" + plan.get("sign_off", "")).strip())
    return "\n".join(line for line in lines if line).strip()


# ── Props builder ──────────────────────────────────────────────────────────────

def _build_viewer_props(plan: dict) -> dict:
    """Pack all plan fields into the shape TravelPlanViewer.tsx expects."""
    flight_pairings = [
        {
            "total_price":   float(p.get("total_price", 0) or 0),
            "description":   p.get("description", ""),
            "outbound":      _norm_flight(p.get("outbound") or {}),
            "return_flight": _norm_flight(p.get("return_flight") or {}),
        }
        for p in (plan.get("flight_pairings") or [])
    ]

    hotels = [
        {
            "name":            h.get("name", ""),
            "stars":           h.get("stars"),
            "price_per_night": float(h.get("price_per_night", 0) or 0),
            "description":     h.get("description", ""),
        }
        for h in (plan.get("hotels") or [])
    ]

    activities = [
        {"name": a.get("name", ""), "description": a.get("description", "")}
        for a in (plan.get("activities") or [])
    ]

    restaurants = [
        {
            "name":        r.get("name", ""),
            "price_tier":  r.get("price_tier", ""),
            "rating":      r.get("rating"),
            "description": r.get("description", ""),
        }
        for r in (plan.get("restaurants") or [])
    ]

    budget = plan.get("total_budget")
    group  = plan.get("lowest_group_estimate")

    return {
        "destination":           plan.get("destination", ""),
        "origin":                plan.get("origin", ""),
        "trip_days":             plan.get("trip_days"),
        "trip_start":            plan.get("trip_start", ""),
        "total_budget":          float(budget) if budget is not None else None,
        "travelers_label":       plan.get("travelers_label") or "1 adult",
        "lowest_group_estimate": float(group)  if group  is not None else None,
        "flight_pairings":       flight_pairings,
        "hotels":                hotels,
        "activities":            activities,
        "restaurants":           restaurants,
        "weather":               plan.get("weather") or {},
        "best_time":             plan.get("best_time") or {},
    }


def _norm_flight(f: dict) -> dict:
    """Normalize a raw flight dict to a clean shape for the frontend."""
    if not f:
        return {}
    stops = f.get("stops")
    dur   = f.get("duration_minutes")
    return {
        "airline":             f.get("airline", ""),
        "label":               f.get("label", ""),
        "price":               float(f.get("price", 0) or 0),
        "stops":               int(stops) if stops is not None else None,
        "duration_minutes":    int(dur)   if dur   is not None else None,
        "departure_time":      f.get("departure_time", ""),
        "destination_airport": f.get("destination_airport", ""),
    }
