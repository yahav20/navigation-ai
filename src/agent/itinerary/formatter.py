# src/agent/nodes/itinerary/formatter.py
"""
ItineraryFormatterNode
======================
Renders the final output to the user.

Success path:
  - Prepends a deterministic Flights & Accommodation section (mode-aware)
  - Then appends the LLM-generated day schedule from replanner's final_markdown
  - *** NEW: also emits a UI message so ItineraryViewer renders in the frontend ***

Failure path:
  - Renders a friendly error with suggestions

Mode-aware header:
  with_travel_data  — shows real airline/flight number/price and hotel name/stars/price
  standalone        — shows average estimated prices with a ~ marker and "estimated" label
"""
from __future__ import annotations

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from agent.core.state import AgentState
from agent.shared.pricing import activity_group_price, flight_group_price, hotel_group_price, group_label
from agent.shared.travelers import compute_default_rooms

REASON_HUMANIZE_SYSTEM = """You are a friendly travel assistant.

You receive a machine-generated failure reason (possibly JSON) explaining why a
trip schedule could not be built. Rewrite it as ONE short, warm, plain-language
sentence the traveller will understand.

Rules:
- No JSON, no field names, no error codes, no internal jargon.
- Explain the gist in human terms (e.g. too few activities, over budget).
- One sentence, no more than ~25 words. No markdown, no quotes.
"""


# ---------------------------------------------------------------------------
# UI-message builder  (NEW)
# ---------------------------------------------------------------------------

def _build_ui_props(state: AgentState) -> dict:
    """
    Assemble the props dict that ItineraryViewer.tsx expects.
    Called only on the success path.
    """
    mode        = state.get("itinerary_mode", "standalone")
    destination = state.get("destination_city", "")
    origin      = state.get("current_city", "")
    trip_days   = int(state.get("trip_days") or 0)
    plan_state  = state.get("itinerary_plan", {})
    results     = plan_state.get("step_results", {})

    # ── Flights ──────────────────────────────────────────────────────────────
    flights: list[dict] = []

    if mode == "with_travel_data":
        ob = state.get("itinerary_selected_outbound_flight") or {}
        ret = state.get("itinerary_selected_return_flight") or {}
        if ob:
            flights.append({
                "direction":     "outbound",
                "airline":       ob.get("airline", ""),
                "flight_number": ob.get("flight_number", ""),
                "route":         f"{origin} → {destination}",
                "datetime":      _fmt_time(ob.get("departure_time", ""))
                                 + (" · " + _fmt_time(ob.get("arrival_time", ""))
                                    if ob.get("arrival_time") else ""),
                "price":         f"${float(ob.get('price', 0) or 0):.0f}",
            })
        if ret:
            flights.append({
                "direction":     "return",
                "airline":       ret.get("airline", ""),
                "flight_number": ret.get("flight_number", ""),
                "route":         f"{destination} → {origin}",
                "datetime":      _fmt_time(ret.get("departure_time", ""))
                                 + (" · " + _fmt_time(ret.get("arrival_time", ""))
                                    if ret.get("arrival_time") else ""),
                "price":         f"${float(ret.get('price', 0) or ret.get('total_price', 0) or 0):.0f}",
            })
    else:
        # standalone — read from verify_budget avg_prices
        budget_key = next((k for k in results if k.startswith("verify_budget")), None)
        avg_p: dict = {}
        if budget_key:
            inner = _unwrap_result(results[budget_key])
            avg_p = inner.get("avg_prices") or {}
        p_out = float(avg_p.get("avg_flight_price", 0) or 0)
        p_ret = float(avg_p.get("avg_return_flight_price", 0) or 0)
        if p_out:
            flights.append({
                "direction": "outbound",
                "route":     f"{origin} → {destination}",
                "datetime":  "estimated",
                "price":     f"~${p_out:.0f}",
            })
        if p_ret:
            flights.append({
                "direction": "return",
                "route":     f"{destination} → {origin}",
                "datetime":  "estimated",
                "price":     f"~${p_ret:.0f}",
            })

    # ── Hotels ───────────────────────────────────────────────────────────────
    hotels: list[dict] = []

    if mode == "with_travel_data":
        travel_plan = state.get("travel_plan") or {}
        raw_hotels  = travel_plan.get("hotels", [])
        if raw_hotels:
            h0 = raw_hotels[0]
            hotels.append({
                "name":            h0.get("name", "Hotel"),
                "address":         h0.get("address", ""),
                "stars":           h0.get("stars"),
                "price_per_night": f"${float(h0.get('price_per_night', 0) or 0):.0f}/night",
                "nights":          trip_days,
                "lat":             h0.get("lat"),
                "lng":             h0.get("lng"),
            })
    else:
        budget_key = next((k for k in results if k.startswith("verify_budget")), None)
        avg_p_hotel: dict = {}
        if budget_key:
            inner = _unwrap_result(results[budget_key])
            avg_p_hotel = inner.get("avg_prices") or {}
        p_hotel = float(avg_p_hotel.get("avg_hotel_per_night", 0) or 0)
        if p_hotel:
            hotels.append({
                "name":            f"Hotel in {destination}",
                "price_per_night": f"~${p_hotel:.0f}/night",
                "nights":          trip_days,
            })

    # ── Coordinate lookup from already-fetched activities ────────────────────
    _coord_idx: dict[str, tuple[float, float]] = {}
    acts_key = next((k for k in results if k.startswith("fetch_activities")), None)
    if acts_key:
        acts_inner = _unwrap_result(results[acts_key])
        for pool in ("activities", "restaurants"):
            for a in acts_inner.get(pool, []):
                name = a.get("name", "")
                lat  = float(a.get("latitude") or a.get("lat") or 0)
                lng  = float(a.get("longitude") or a.get("lng") or 0)
                if name and (lat or lng):
                    _coord_idx[name] = (lat, lng)

    # Supplement with coords from update_day_schedule results — newly-searched
    # replacement activities are not in fetch_activities but are stored in coord_index.
    for k, v in results.items():
        if k.startswith("build_day_schedule"):
            inner = _unwrap_result(v)
            for name, coords in (inner.get("coord_index") or {}).items():
                if name not in _coord_idx and isinstance(coords, list) and len(coords) == 2:
                    lat, lng = float(coords[0]), float(coords[1])
                    if lat or lng:
                        _coord_idx[name] = (lat, lng)

    # ── Days ─────────────────────────────────────────────────────────────────
    days: list[dict] = []

    for d in range(1, trip_days + 1):
        key = next(
            (k for k in results
             if k.startswith("build_day_schedule")
             and isinstance(_unwrap_result(results[k]), dict)
             and _unwrap_result(results[k]).get("day") == d),
            None,
        )
        if not key:
            continue
        day_data = _unwrap_result(results[key])
        raw_slots = day_data.get("slots", [])

        # Enrich slots with lat/lng from the activities lookup
        num_adults   = int(state.get("num_adults") or 1)
        num_children = int(state.get("num_children") or 0)
        slots = []
        for s in raw_slots:
            slot = dict(s)
            if slot.get("lat") is None or slot.get("lng") is None:
                coords = _coord_idx.get(slot.get("name", ""))
                if coords:
                    slot["lat"], slot["lng"] = coords
            cost_pp = float(s.get("estimated_cost") or 0)
            slot["cost_per_person"] = cost_pp
            slot["group_cost"] = activity_group_price(cost_pp, num_adults, num_children)
            slots.append(slot)

        # Derive center from mapped slots (skip transport/rest which have no fixed location)
        mapped = [
            (slot["lat"], slot["lng"])
            for slot in slots
            if slot.get("lat") and slot.get("lng")
            and slot.get("slot_type") not in ("transport", "rest")
        ]
        center_lat = sum(c[0] for c in mapped) / len(mapped) if mapped else None
        center_lng = sum(c[1] for c in mapped) / len(mapped) if mapped else None

        theme = day_data.get("theme", "")
        days.append({
            "day":        d,
            "label":      f"Day {d}" + (f" — {theme}" if theme else ""),
            "center_lat": center_lat,
            "center_lng": center_lng,
            "slots":      slots,
        })

    return {
        "destination": destination,
        "origin":      origin,
        "flights":     flights,
        "hotels":      hotels,
        "days":        days,
    }


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class ItineraryFormatterNode:
    def __init__(self, llm: BaseChatModel | None = None) -> None:
        self.llm = llm

    def __call__(self, state: AgentState) -> dict:
        plan_state    = state.get("itinerary_plan", {})
        feasible      = state.get("itinerary_feasible", True)
        critic_action = state.get("critic_action", "pass")

        if not feasible:
            return self._format_error(state)

        if critic_action == "abort":
            return self._format_abort(state)

        final_markdown = plan_state.get("final_markdown", "")
        header  = self._build_travel_header(state)
        content = (header + "\n\n" + final_markdown).strip() if header else final_markdown

        # Always append the budget summary from the latest verify_budget_0 —
        # this ensures it reflects min prices if the critic updated that result.
        results   = plan_state.get("step_results", {})
        budget    = float(state.get("total_budget") or 0)
        mode      = state.get("itinerary_mode", "standalone")
        budget_md = _budget_section_md(results, budget, mode, state)
        if budget_md:
            content = content + "\n" + budget_md

        if critic_action == "ignore_budget":
            content = self._prepend_over_budget_banner(state, content)

        # ── NEW: build the UI message for ItineraryViewer ─────────────────────
        ai_message = AIMessage(content="")
        ui_props   = _build_ui_props(state)
        ui_message = {
            "type":     "ui",
            "name":     "ItineraryViewer",
            "props":    ui_props,
            # tie this UI message to the AI message so CustomComponent finds it
            "metadata": {"message_id": ai_message.id},
        }

        return {
            "messages": [ai_message],
            "ui":       [ui_message],
        }

    # ── Travel header (flights + accommodation) ────────────────────────────
    # (Everything below is UNCHANGED from the original formatter)

    def _build_travel_header(self, state: AgentState) -> str:
        mode        = state.get("itinerary_mode", "standalone")
        origin      = state.get("current_city", "")
        destination = state.get("destination_city", "")
        trip_days   = state.get("trip_days", 0)

        if mode == "with_travel_data":
            return self._header_with_travel_data(state, origin, destination)
        else:
            return self._header_standalone(state, origin, destination, trip_days)

    def _header_with_travel_data(
        self, state: AgentState, origin: str, destination: str
    ) -> str:
        outbound_raw = state.get("itinerary_selected_outbound_flight") or {}
        return_raw   = state.get("itinerary_selected_return_flight") or {}
        travel_plan  = state.get("travel_plan") or {}
        hotels       = travel_plan.get("hotels", [])

        num_adults   = int(state.get("num_adults")   or 1)
        num_children = int(state.get("num_children") or 0)
        num_rooms    = int(state.get("num_rooms")    or compute_default_rooms(num_adults, num_children))
        is_group     = num_adults + num_children > 1
        glabel       = group_label(num_adults, num_children)

        lines = ["## 🛫 Flights & Accommodation", ""]

        # ── Outbound ──────────────────────────────────────────────────────
        if outbound_raw:
            ob_line = _fmt_flight_line("✈️ **Outbound**", outbound_raw, origin, destination)
            if is_group:
                ob_price = float(outbound_raw.get("price", 0) or 0)
                ob_group = flight_group_price(ob_price, num_adults, num_children)
                ob_line += f" | 👥 **${ob_group:.0f} for {glabel}**"
            lines.append(ob_line)
        elif origin and destination:
            lines.append(f"✈️ **Outbound:** {origin} → {destination}")

        # ── Return ────────────────────────────────────────────────────────
        if return_raw:
            ret_line = _fmt_flight_line("🔙 **Return**", return_raw, destination, origin)
            if is_group:
                ret_price = float(return_raw.get("price", 0) or return_raw.get("total_price", 0) or 0)
                ret_group = flight_group_price(ret_price, num_adults, num_children)
                ret_line += f" | 👥 **${ret_group:.0f} for {glabel}**"
            lines.append(ret_line)
        elif destination and origin:
            lines.append(f"🔙 **Return:** {destination} → {origin} *(details not confirmed)*")

        # ── Hotel ─────────────────────────────────────────────────────────
        if hotels:
            h0       = hotels[0]
            name     = h0.get("name", "Hotel")
            stars    = h0.get("stars")
            price    = float(h0.get("price_per_night", 0) or 0)
            star_str = f" — {stars:.0f}★" if stars else ""
            hotel_line = f"🏨 **Hotel:** {name}{star_str} | **${price:.0f}/night**"
            if is_group and price:
                trip_days = state.get("trip_days", 0)
                h_group   = hotel_group_price(price, num_rooms, int(trip_days))
                hotel_line += f" | 👥 **${h_group:.0f} for {num_rooms} room{'s' if num_rooms > 1 else ''} × {trip_days} nights**"
            lines.append("")
            lines.append(hotel_line)

        return "\n\n".join(lines)

    def _header_standalone(
        self, state: AgentState, origin: str, destination: str, trip_days: int
    ) -> str:
        plan_state    = state.get("itinerary_plan", {})
        results       = plan_state.get("step_results", {})
        critic_action = state.get("critic_action", "pass")

        stored_prices: dict = {}
        budget_key = next((k for k in results if k.startswith("verify_budget")), None)
        if budget_key:
            b = results[budget_key]
            inner = b.get("data", b) if isinstance(b, dict) else {}
            if isinstance(inner, dict):
                stored_prices = inner.get("avg_prices") or {}

        if not stored_prices:
            return ""

        is_min  = "minimum" in str(stored_prices.get("note", "")).lower()
        route   = f"{origin} ↔ {destination}" if origin and destination else destination
        budget  = float(state.get("total_budget") or 0)

        price_out   = float(stored_prices.get("avg_flight_price", 0) or 0)
        price_ret   = float(stored_prices.get("avg_return_flight_price", 0) or 0)
        price_hotel = float(stored_prices.get("avg_hotel_per_night", 0) or 0)
        hotel_total = price_hotel * trip_days if trip_days else price_hotel

        num_adults   = int(state.get("num_adults")   or 1)
        num_children = int(state.get("num_children") or 0)
        num_rooms    = int(state.get("num_rooms")    or compute_default_rooms(num_adults, num_children))
        is_group     = num_adults + num_children > 1
        glabel       = group_label(num_adults, num_children)

        g_flight_total = (flight_group_price(price_out, num_adults, num_children)
                          + flight_group_price(price_ret, num_adults, num_children))
        g_hotel_total  = hotel_group_price(price_hotel, num_rooms, trip_days) if trip_days else 0.0

        group_suffix_flights = f" | 👥 **~${g_flight_total:.0f} for {glabel}**" if is_group else ""
        group_suffix_hotel   = (f" | 👥 **~${g_hotel_total:.0f} for {num_rooms} room{'s' if num_rooms > 1 else ''} × {trip_days} nights**"
                                if is_group and price_hotel else "")

        if is_min and critic_action == "min_travel":
            b_data    = _unwrap_result(results.get(budget_key, {}))
            min_total = float(b_data.get("group_grand_total") or b_data.get("grand_total", 0)) if isinstance(b_data, dict) else 0.0

            lines = [
                "## 💡 Approximate Travel Cost *(no booking confirmed)*",
                "",
                f"✈️ **Flights** ({route}): cheapest available **~${price_out + price_ret:.0f}**{group_suffix_flights}",
                "",
                f"🏨 **Hotel** ({destination}): cheapest available **~${price_hotel:.0f}/night**{group_suffix_hotel}",
                "",
                "> *Prices shown are the lowest currently available — actual rates may vary.*",
                "",
                (
                    f"> ⚠️ **Budget Advisory:** Average market prices would exceed your "
                    f"**${budget:.0f}** budget, but the cheapest available options bring the total "
                    f"to ~**${min_total:.0f}**. "
                    f"To stay on budget, look for flights totalling under **~${price_out + price_ret:.0f}** "
                    f"and a hotel under **~${price_hotel:.0f}/night**."
                ),
            ]
            return "\n".join(lines)

        if is_min:
            lines = [
                "## 💡 Approximate Travel Cost *(minimum available — no booking confirmed)*",
                "",
                f"✈️ **Flights** ({route}): **~${price_out + price_ret:.0f}** *(minimum available)*{group_suffix_flights}",
                "",
                f"🏨 **Hotel** ({destination}): **~${price_hotel:.0f}/night**"
                + (f" (~${hotel_total:.0f} for {trip_days} nights)" if trip_days else "")
                + f" *(minimum available)*{group_suffix_hotel}",
                "",
                "> *Prices reflect the cheapest currently available options. Actual booking rates may vary.*",
            ]
            return "\n".join(lines)

        lines = [
            "## 💡 Approximate Travel Cost *(no booking confirmed)*",
            "",
            f"✈️ **Flights** ({route}): **~${price_out + price_ret:.0f}**"
            + (f" *(~${price_out:.0f} outbound + ~${price_ret:.0f} return)*" if price_out and price_ret else "")
            + group_suffix_flights,
            "",
            f"🏨 **Hotel** ({destination}): **~${price_hotel:.0f}/night**"
            + (f" (~${hotel_total:.0f} for {trip_days} nights)" if trip_days else "")
            + group_suffix_hotel,
            "",
            "> *Market averages for planning purposes only. Actual prices may vary.*",
        ]
        return "\n".join(lines)

    # ── Critic-driven render paths ─────────────────────────────────────────

    def _prepend_over_budget_banner(self, state: AgentState, content: str) -> str:
        plan_state  = state.get("itinerary_plan") or {}
        results     = plan_state.get("step_results", {})
        budget      = float(state.get("total_budget") or 0)
        grand_total = 0.0
        budget_key  = next((k for k in results if k.startswith("verify_budget")), None)
        if budget_key:
            data        = results[budget_key].get("data") or {}
            grand_total = float(data.get("group_grand_total") or data.get("grand_total", 0))
        overage = grand_total - budget
        banner = (
            f"> ⚠️ **Over Budget Notice:** This itinerary costs approximately "
            f"**${grand_total:.0f}**, which is **${overage:.0f} over** your "
            f"**${budget:.0f}** budget.\n"
        )
        return banner + "\n" + content

    def _format_abort(self, state: AgentState) -> dict:
        destination = state.get("destination_city") or "your destination"
        budget      = state.get("total_budget") or 0
        message = (
            f"### No problem!\n\n"
            f"I've cancelled the itinerary planning for **{destination}**. "
            f"Feel free to start again whenever you're ready — "
            f"perhaps with a higher budget, fewer days, or a different destination. "
            f"I'm here to help whenever you'd like! 😊"
        )
        if budget:
            message += f"\n\n*(Current budget on file: **${float(budget):.0f}**)*"
        return {"messages": [AIMessage(content=message)]}

    # ── Error path ─────────────────────────────────────────────────────────

    def _format_error(self, state: AgentState) -> dict:
        reason_raw  = state.get("itinerary_fallback_reason", "")
        reason      = self._humanize_reason(reason_raw)
        destination = state.get("destination_city", "your destination")

        message = (
            f"### 😕 Couldn't complete the itinerary\n\n"
            f"Unfortunately I wasn't able to build a full schedule for **{destination}** — "
            f"{reason}.\n\n"
            "You can try:\n"
            "- Adjusting the number of days\n"
            "- Relaxing any dietary or preference filters\n"
            "- Choosing a different destination"
        )
        return {"messages": [AIMessage(content=message)]}

    def _humanize_reason(self, reason_raw: str) -> str:
        gist = _extract_gist(reason_raw)
        if not self.llm:
            return gist
        try:
            out = self.llm.invoke([
                SystemMessage(content=REASON_HUMANIZE_SYSTEM),
                HumanMessage(content=str(reason_raw)),
            ])
            sentence = (out.content or "").strip().strip('"')
            if sentence and "{" not in sentence and "_" not in sentence:
                return sentence
        except Exception:
            pass
        return gist


# ---------------------------------------------------------------------------
# Pure helpers  (unchanged from original)
# ---------------------------------------------------------------------------

def _generate_fallback_markdown(
    results: dict, trip_days: int, budget: float, mode: str = "standalone",
    state: dict | None = None,
) -> str:
    """Plain-text itinerary renderer — used when the LLM quality review fails."""
    num_adults   = int((state or {}).get("num_adults")   or 1)
    num_children = int((state or {}).get("num_children") or 0)
    is_group     = num_adults + num_children > 1
    glabel       = group_label(num_adults, num_children) if is_group else ""

    lines = ["# ✈️ Your Trip Itinerary\n"]

    hotel_name = None
    for key, val in results.items():
        if key.startswith("build_day_schedule"):
            inner = _unwrap_result(val)
            hotel = inner.get("hotel") if isinstance(inner, dict) else None
            if hotel and hotel.strip():
                hotel_name = hotel
                break
    if hotel_name:
        lines.append(f"## 🏨 Accommodation\n\n**{hotel_name}**\n")

    for d in range(1, trip_days + 1):
        key = next(
            (k for k in results
             if k.startswith("build_day_schedule")
             and isinstance(_unwrap_result(results[k]), dict)
             and _unwrap_result(results[k]).get("day") == d),
            None,
        )
        if not key:
            continue
        day_data = _unwrap_result(results[key])
        lines.append(f"\n## 📅 Day {d} — {day_data.get('theme', '')}")
        if is_group:
            lines.append(f"\n| Time | Activity | Duration | Per Person | Group ({glabel}) |")
            lines.append("|------|----------|----------|-----------|----------------|")
        else:
            lines.append("\n| Time | Activity | Duration | Est. Cost |")
            lines.append("|------|----------|----------|-----------|")
        for slot in day_data.get("slots", []):
            icon = {"activity": "🎯", "meal": "🍽️", "transport": "🚕",
                    "rest": "😴", "checkin": "🏨"}.get(slot.get("slot_type", ""), "•")
            cost = float(slot.get("estimated_cost", 0))
            if is_group:
                group_cost = activity_group_price(cost, num_adults, num_children)
                lines.append(
                    f"| {slot.get('time', '')} | {icon} {slot.get('name', '')} | "
                    f"{slot.get('duration_minutes', '')} min | ${cost:.0f} | ${group_cost:.0f} |"
                )
            else:
                lines.append(
                    f"| {slot.get('time', '')} | {icon} {slot.get('name', '')} | "
                    f"{slot.get('duration_minutes', '')} min | ${cost:.0f} |"
                )
        day_cost = float(day_data.get("day_cost", 0))
        if is_group:
            day_group = activity_group_price(day_cost, num_adults, num_children)
            lines.append(f"\n**Day total: ${day_cost:.0f} pp | ${day_group:.0f} for {glabel}**")
        else:
            lines.append(f"\n**Day total: ${day_cost:.0f}**")

    return "\n".join(lines)


def _budget_section_md(results: dict, budget: float, mode: str, state: dict | None = None) -> str:
    """Deterministic budget summary section — appended after the LLM day-schedule markdown."""
    budget_key = next((k for k in results if k.startswith("verify_budget")), None)
    if not budget_key:
        return ""
    b = _unwrap_result(results[budget_key])
    if not isinstance(b, dict) or b.get("grand_total") is None:
        return ""

    avg_p  = b.get("avg_prices") or {}
    is_min = "minimum" in str(avg_p.get("note", "")).lower()

    num_adults   = int((state or {}).get("num_adults")   or b.get("num_adults",   1))
    num_children = int((state or {}).get("num_children") or b.get("num_children", 0))
    num_rooms    = int((state or {}).get("num_rooms")    or b.get("num_rooms",    1))
    is_group     = num_adults + num_children > 1
    glabel       = group_label(num_adults, num_children)

    has_group = is_group and b.get("group_grand_total") is not None

    lines: list[str] = ["\n\n---------------------------------------------------------------------------------------\n"]
    lines.append("## 💰 Budget Summary\n")

    if has_group:
        lines.append(f"| Category | Per Person | Group ({glabel}) |")
        lines.append("|----------|-----------|----------------|")
    else:
        lines.append("| Category | Cost |")
        lines.append("|----------|------|")

    def _row(label: str, solo: float, group_val: float | None = None) -> str:
        if has_group and group_val is not None:
            return f"| {label} | ${solo:.0f} | ${group_val:.0f} |"
        return f"| {label} | ${solo:.0f} |"

    if mode == "with_travel_data":
        ob_price  = float(b.get("outbound_flight", 0) or 0)
        ret_price = float(b.get("return_flight",   0) or 0)
        if ob_price:
            lines.append(_row("Outbound Flight", ob_price, b.get("group_outbound_flight")))
        if ret_price:
            lines.append(_row("Return Flight", ret_price, b.get("group_return_flight")))
        hotel_label_prefix = ""
        hotel_suffix       = ""
    else:
        out = float(avg_p.get("avg_flight_price", 0) or 0)
        ret = float(avg_p.get("avg_return_flight_price", 0) or 0)
        if out or ret:
            flight_label = "Flights (minimum available)" if is_min else "~ Flights (estimated)"
            g_flights = (float(b.get("group_outbound_flight", 0) or 0)
                         + float(b.get("group_return_flight", 0) or 0))
            lines.append(_row(flight_label, out + ret, g_flights if has_group else None))
        hotel_label_prefix = "" if is_min else "~ "
        hotel_suffix       = " (minimum available)" if is_min else " (estimated)"

    _skip = {"grand_total", "group_grand_total", "avg_prices",
             "outbound_flight", "return_flight",
             "group_outbound_flight", "group_return_flight",
             "num_adults", "num_children", "num_rooms"}
    for cat, val in b.items():
        if cat in _skip:
            continue
        if not isinstance(val, (int, float)):
            continue
        if cat.startswith("group_"):
            continue
        label = cat.replace("_", " ").title()
        if "hotel" in cat.lower():
            label = f"{hotel_label_prefix}{label}{hotel_suffix}"
            if is_group and num_rooms > 1:
                label += f" ({num_rooms} rooms)"
        group_key = f"group_{cat}"
        group_val = float(b[group_key]) if group_key in b and has_group else None
        lines.append(_row(label, float(val), group_val))

    grand       = float(b.get("grand_total", 0))
    group_grand = float(b.get("group_grand_total", grand))

    if has_group:
        lines.append(f"\n**Grand Total: ${grand:.0f} pp | ${group_grand:.0f} for {glabel}**")
    else:
        lines.append(f"\n**Grand Total: ${grand:.0f}**")

    if budget:
        compare = group_grand if has_group else grand
        remaining = budget - compare
        emoji = "✅" if remaining >= 0 else "⚠️"
        lines.append(f"\n{emoji} Budget remaining: ${remaining:.0f}")

    return "\n".join(lines)


def _unwrap_result(val: dict) -> dict:
    """Unwrap a step result dict to its inner data."""
    if not isinstance(val, dict):
        return {}
    if "data" in val and isinstance(val["data"], dict):
        return val["data"]
    return val


def _fmt_time(iso: str) -> str:
    """Convert an ISO datetime string to a compact display like '07 Jun, 19:40'."""
    if not iso:
        return ""
    try:
        if "T" in iso:
            date_part, time_part = iso.split("T", 1)
            time_part = time_part[:5]
            year, month, day = date_part.split("-")
            months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            month_name = months[int(month) - 1]
            return f"{int(day)} {month_name}, {time_part}"
        return iso[:5]
    except Exception:
        return iso


def _fmt_flight_line(label: str, flight: dict, from_city: str, to_city: str) -> str:
    """Format a single flight as a compact markdown line."""
    airline  = flight.get("airline", "")
    num      = flight.get("flight_number", "")
    price    = float(flight.get("price", 0) or 0)
    dep_raw  = flight.get("departure_time", "")
    arr_raw  = flight.get("arrival_time", "")
    stops    = flight.get("transfers") or flight.get("stops")
    duration = flight.get("duration_minutes")

    parts = [f"{label}:"]
    if airline or num:
        parts.append(f"{airline} {num}".strip())
    if from_city and to_city:
        parts.append(f"({from_city} → {to_city})")

    dep_str = _fmt_time(dep_raw)
    arr_str = _fmt_time(arr_raw)
    if dep_str and arr_str:
        parts.append(f"| Dep: {dep_str} — Arr: {arr_str}")
    elif dep_str:
        parts.append(f"| Dep: {dep_str}")

    if duration:
        h, m = divmod(int(duration), 60)
        parts.append(f"| {h}h {m:02d}m" if h else f"| {m}min")

    if stops is not None:
        stops_str = "Direct" if int(stops) == 0 else f"{stops} stop{'s' if int(stops) > 1 else ''}"
        parts.append(f"| {stops_str}")

    parts.append(f"| **${price:.0f}**")
    return " ".join(parts)


def _extract_gist(reason_raw) -> str:
    if not reason_raw:
        return "we couldn't complete the schedule within your constraints"
    text = reason_raw
    if isinstance(reason_raw, str):
        try:
            ctx = json.loads(reason_raw)
        except (json.JSONDecodeError, TypeError):
            ctx = None
        if isinstance(ctx, dict):
            text = ctx.get("error_message") or ctx.get("replan_hint") or "planning failed"
    elif isinstance(reason_raw, dict):
        text = reason_raw.get("error_message") or reason_raw.get("replan_hint") or "planning failed"
    text = str(text)
    marker = "Last error:"
    if marker in text:
        text = text.split(marker, 1)[1].strip()
    return text or "we couldn't complete the schedule within your constraints"