# src/agent/nodes/itinerary/formatter.py
"""
ItineraryFormatterNode
======================
Renders the final output to the user.

Success path:
  - Prepends a deterministic Flights & Accommodation section (mode-aware)
  - Then appends the LLM-generated day schedule from replanner's final_markdown

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
from agent.state import AgentState

REASON_HUMANIZE_SYSTEM = """You are a friendly travel assistant.

You receive a machine-generated failure reason (possibly JSON) explaining why a
trip schedule could not be built. Rewrite it as ONE short, warm, plain-language
sentence the traveller will understand.

Rules:
- No JSON, no field names, no error codes, no internal jargon.
- Explain the gist in human terms (e.g. too few activities, over budget).
- One sentence, no more than ~25 words. No markdown, no quotes.
"""


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
        budget_md = _budget_section_md(results, budget, mode)
        if budget_md:
            content = content + "\n" + budget_md

        if critic_action == "ignore_budget":
            content = self._prepend_over_budget_banner(state, content)

        return {"messages": [AIMessage(content=content)]}

    # ── Travel header (flights + accommodation) ────────────────────────────

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

        lines = ["## 🛫 Flights & Accommodation", ""]

        # ── Outbound ──────────────────────────────────────────────────────
        if outbound_raw:
            lines.append(_fmt_flight_line("✈️ **Outbound**", outbound_raw, origin, destination))
        elif origin and destination:
            lines.append(f"✈️ **Outbound:** {origin} → {destination}")

        # ── Return ────────────────────────────────────────────────────────
        if return_raw:
            lines.append(_fmt_flight_line("🔙 **Return**", return_raw, destination, origin))
        elif destination and origin:
            lines.append(f"🔙 **Return:** {destination} → {origin} *(details not confirmed)*")

        # ── Hotel ─────────────────────────────────────────────────────────
        if hotels:
            h0       = hotels[0]
            name     = h0.get("name", "Hotel")
            stars    = h0.get("stars")
            price    = h0.get("price_per_night", 0)
            star_str = f" — {stars:.0f}★" if stars else ""
            lines.append("")
            lines.append(f"🏨 **Hotel:** {name}{star_str} | **${float(price):.0f}/night**")

        return "\n\n".join(lines)

    def _header_standalone(
        self, state: AgentState, origin: str, destination: str, trip_days: int
    ) -> str:
        plan_state    = state.get("itinerary_plan", {})
        results       = plan_state.get("step_results", {})
        critic_action = state.get("critic_action", "pass")

        # Read price data from verify_budget_0 — keys are always avg_* regardless of mode;
        # the "note" field distinguishes average vs minimum prices.
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

        if is_min and critic_action == "min_travel":
            # Read the min grand_total from verify_budget_0 data
            b_data    = _unwrap_result(results.get(budget_key, {}))
            min_total = float(b_data.get("grand_total", 0)) if isinstance(b_data, dict) else 0.0

            lines = [
                "## 💡 Approximate Travel Cost *(no booking confirmed)*",
                "",
                f"✈️ **Flights** ({route}): cheapest available **~${price_out + price_ret:.0f}**",
                "",
                f"🏨 **Hotel** ({destination}): cheapest available **~${price_hotel:.0f}/night**",
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
                f"✈️ **Flights** ({route}): **~${price_out + price_ret:.0f}** *(minimum available)*",
                "",
                f"🏨 **Hotel** ({destination}): **~${price_hotel:.0f}/night**"
                + (f" (~${hotel_total:.0f} for {trip_days} nights)" if trip_days else "")
                + " *(minimum available)*",
                "",
                "> *Prices reflect the cheapest currently available options. Actual booking rates may vary.*",
            ]
            return "\n".join(lines)

        # ── Default: show market averages ────────────────────────────────────
        lines = [
            "## 💡 Approximate Travel Cost *(no booking confirmed)*",
            "",
            f"✈️ **Flights** ({route}): **~${price_out + price_ret:.0f}**"
            + (f" *(~${price_out:.0f} outbound + ~${price_ret:.0f} return)*" if price_out and price_ret else ""),
            "",
            f"🏨 **Hotel** ({destination}): **~${price_hotel:.0f}/night**"
            + (f" (~${hotel_total:.0f} for {trip_days} nights)" if trip_days else ""),
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
            grand_total = float(data.get("grand_total", 0))
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


def _generate_fallback_markdown(results: dict, trip_days: int, budget: float, mode: str = "standalone") -> str:
    """Plain-text itinerary renderer — used when the LLM quality review fails."""
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
        lines.append("\n| Time | Activity | Duration | Est. Cost |")
        lines.append("|------|----------|----------|-----------|")
        for slot in day_data.get("slots", []):
            icon = {"activity": "🎯", "meal": "🍽️", "transport": "🚕",
                    "rest": "😴", "checkin": "🏨"}.get(slot.get("slot_type", ""), "•")
            lines.append(
                f"| {slot.get('time', '')} | {icon} {slot.get('name', '')} | "
                f"{slot.get('duration_minutes', '')} min | ${slot.get('estimated_cost', 0):.0f} |"
            )
        lines.append(f"\n**Day total: ${day_data.get('day_cost', 0):.0f}**")

    return "\n".join(lines)


def _budget_section_md(results: dict, budget: float, mode: str) -> str:
    """Deterministic budget summary section — appended after the LLM day-schedule markdown."""
    budget_key = next((k for k in results if k.startswith("verify_budget")), None)
    if not budget_key:
        return ""
    b = _unwrap_result(results[budget_key])
    if not isinstance(b, dict) or b.get("grand_total") is None:
        return ""

    # Determine if this is a min-price calculation or avg-price (for standalone labels)
    avg_p  = b.get("avg_prices") or {}
    is_min = "minimum" in str(avg_p.get("note", "")).lower()

    lines: list[str] = ["\n\n---------------------------------------------------------------------------------------\n"]
    lines.append("## 💰 Budget Summary\n")
    lines.append("| Category | Cost |")
    lines.append("|----------|------|")

    if mode == "with_travel_data":
        ob_price  = float(b.get("outbound_flight", 0) or 0)
        ret_price = float(b.get("return_flight",   0) or 0)
        if ob_price:
            lines.append(f"| Outbound Flight | ${ob_price:.0f} |")
        if ret_price:
            lines.append(f"| Return Flight | ${ret_price:.0f} |")
        hotel_label_prefix = ""
        hotel_suffix       = ""
    else:
        # Standalone: show flight estimate row with appropriate label
        out = float(avg_p.get("avg_flight_price", 0) or 0)
        ret = float(avg_p.get("avg_return_flight_price", 0) or 0)
        if out or ret:
            flight_label = "Flights (minimum available)" if is_min else "~ Flights (estimated)"
            lines.append(f"| {flight_label} | ${out + ret:.0f} |")
        hotel_label_prefix = "" if is_min else "~ "
        hotel_suffix       = " (minimum available)" if is_min else " (estimated)"

    for cat, val in b.items():
        if cat in ("grand_total", "avg_prices", "outbound_flight", "return_flight"):
            continue
        if not isinstance(val, (int, float)):
            continue
        label = cat.replace("_", " ").title()
        if "hotel" in cat.lower():
            label = f"{hotel_label_prefix}{label}{hotel_suffix}"
        lines.append(f"| {label} | ${float(val):.0f} |")

    grand = float(b.get("grand_total", 0))
    lines.append(f"\n**Grand Total: ${grand:.0f}**")
    if budget:
        remaining = budget - grand
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
        # Handle both "2026-06-07T19:40:00+03:00" and "19:40" plain strings
        if "T" in iso:
            date_part, time_part = iso.split("T", 1)
            time_part = time_part[:5]               # HH:MM
            year, month, day = date_part.split("-")
            months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            month_name = months[int(month) - 1]
            return f"{int(day)} {month_name}, {time_part}"
        return iso[:5]  # already HH:MM
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
