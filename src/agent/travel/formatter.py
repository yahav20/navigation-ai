"""Emit the curated TravelPlan as a generative-UI message for TravelPlanViewer."""

from langchain_core.messages import AIMessage

from agent.core.state import AgentState


class FormatterNode:
    """Emit a TravelPlanViewer UI message from the structured TravelPlan in state."""

    def __call__(self, state: AgentState) -> dict:
        plan = state.get("travel_plan")
        if not plan:
            return {}

        ui_props = _build_viewer_props(plan)

        events = state.get("special_events_data") or []
        if events:
            ui_props["special_events"] = [
                {"name": ev["name"]}
                for ev in events if ev.get("name")
            ]

        ai_message = AIMessage(content="", name="formatter_output")
        ui_message = {
            "type":     "ui",
            "name":     "TravelPlanViewer",
            "props":    ui_props,
            "metadata": {"message_id": ai_message.id},
        }
        return {"messages": [ai_message], "ui": [ui_message]}


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
