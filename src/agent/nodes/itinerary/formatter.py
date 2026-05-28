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
        plan_state = state.get("itinerary_plan", {})
        feasible   = state.get("itinerary_feasible", True)

        if not feasible:
            return self._format_error(state)

        final_markdown = plan_state.get("final_markdown", "")
        header = self._build_travel_header(state)

        content = (header + "\n\n" + final_markdown).strip() if header else final_markdown
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
        # Use raw resolved flights (have actual flight numbers, times, prices)
        # travel_plan["flights"] are curated OUTBOUND options — not split outbound/return
        outbound_raw = state.get("itinerary_selected_outbound_flight") or {}
        return_raw   = state.get("itinerary_selected_return_flight") or {}
        travel_plan  = state.get("travel_plan") or {}
        hotels       = travel_plan.get("hotels", [])

        lines = ["## 🛫 Flights & Accommodation"]

        # ── Outbound ──────────────────────────────────────────────────────
        if outbound_raw:
            out_num     = outbound_raw.get("flight_number", "")
            out_airline = outbound_raw.get("airline", "")
            out_dep     = outbound_raw.get("departure_time", "")
            out_arr     = outbound_raw.get("arrival_time", "")
            out_price   = float(outbound_raw.get("price", 0) or 0)
            dep_str     = f" | Dep: {out_dep} → Arr: {out_arr}" if out_dep else ""
            lines.append(
                f"**Outbound:** {out_airline} {out_num}{dep_str} | **${out_price:.0f}**"
            )
        elif origin and destination:
            lines.append(f"**Outbound:** {origin} → {destination}")

        # ── Return ────────────────────────────────────────────────────────
        if return_raw:
            ret_num     = return_raw.get("flight_number", "")
            ret_airline = return_raw.get("airline", "")
            ret_dep     = return_raw.get("departure_time", "")
            ret_arr     = return_raw.get("arrival_time", "")
            ret_price   = float(return_raw.get("price", 0) or 0)
            dep_str     = f" | Dep: {ret_dep} → Arr: {ret_arr}" if ret_dep else ""
            lines.append(
                f"**Return:** {ret_airline} {ret_num}{dep_str} | **${ret_price:.0f}**"
            )
        elif destination and origin:
            lines.append(f"**Return:** {destination} → {origin} *(details not confirmed)*")

        # ── Hotel ─────────────────────────────────────────────────────────
        if hotels:
            h0 = hotels[0]
            name     = h0.get("name", "Hotel")
            stars    = h0.get("stars")
            price    = h0.get("price_per_night", 0)
            star_str = f" — {stars}★" if stars else ""
            lines.append(f"\n**🏨 Hotel:** {name}{star_str} | ${price:.0f}/night")

        return "\n".join(lines)

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
            return ""  # no estimates available — skip header

        avg_out    = avg_prices.get("avg_flight_price", 0)
        avg_ret    = avg_prices.get("avg_return_flight_price", 0)
        avg_hotel  = avg_prices.get("avg_hotel_per_night", 0)
        note       = avg_prices.get("note", "")

        hotel_total = avg_hotel * trip_days if trip_days else avg_hotel

        lines = [
            "## 💡 Estimated Prices *(averages — no booking confirmed)*",
            "",
            f"✈️ Average outbound flight {origin} → {destination}: **~${avg_out:.0f}**",
            f"✈️ Average return flight {destination} → {origin}: **~${avg_ret:.0f}**",
            f"🏨 Average hotel in {destination}: **~${avg_hotel:.0f}/night**"
            + (f" (~${hotel_total:.0f} total for {trip_days} nights)" if trip_days else ""),
            "",
            "> *These are market averages for planning purposes. Actual prices may vary.*",
        ]
        return "\n".join(lines)

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
