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
        plan_state = state.get("itinerary_plan", {})
        results    = plan_state.get("step_results", {})

        # Read average prices stored by executor in verify_budget result
        avg_prices: dict = {}
        budget_key = next((k for k in results if k.startswith("verify_budget")), None)
        if budget_key:
            b = results[budget_key]
            inner = b.get("data", b) if isinstance(b, dict) else {}
            if isinstance(inner, dict):
                avg_prices = inner.get("avg_prices") or {}

        if not avg_prices:
            return ""

        avg_out   = float(avg_prices.get("avg_flight_price", 0) or 0)
        avg_ret   = float(avg_prices.get("avg_return_flight_price", 0) or 0)
        avg_hotel = float(avg_prices.get("avg_hotel_per_night", 0) or 0)
        total_flights = avg_out + avg_ret
        hotel_total   = avg_hotel * trip_days if trip_days else avg_hotel

        route = f"{origin} ↔ {destination}" if origin and destination else destination

        lines = [
            "## 💡 Approximate Travel Cost *(no booking confirmed)*",
            "",
            f"✈️ **Flights** ({route}): **~${total_flights:.0f}**"
            + (f" *(~${avg_out:.0f} outbound + ~${avg_ret:.0f} return)*" if avg_out and avg_ret else ""),
            "",
            f"🏨 **Hotel** ({destination}): **~${avg_hotel:.0f}/night**"
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
