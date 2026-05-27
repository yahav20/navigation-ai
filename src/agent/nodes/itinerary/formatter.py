# src/agent/nodes/itinerary/formatter.py
from __future__ import annotations

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from agent.state import AgentState

# ── LLM prompt: turn the structured failure reason into one friendly line ───
REASON_HUMANIZE_SYSTEM = """You are a friendly travel assistant.

You receive a machine-generated failure reason (possibly JSON) explaining why a
trip itinerary could not be built. Rewrite it as ONE short, warm, plain-language
sentence the traveler will understand.

Rules:
- No JSON, no field names, no error codes, no internal jargon
  (e.g. never say "LLM_REJECT", "max_retries", "build_day_schedule").
- Explain the gist in human terms (e.g. too few activities to fill a day,
  over budget, no flights available).
- One sentence, no more than ~25 words. No markdown, no quotes, no trailing period stuffing.
"""


class ItineraryFormatterNode:
    """
    Dedicated formatting node for the travel itinerary (Itinerary Formatter).
    Its role is to take the raw State data and turn it into formatted,
    interactive, and inviting text for the user, while handling all edge cases (success/alternatives/errors).
    """

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        self.llm = llm

    def __call__(self, state: AgentState) -> dict:
        plan_state = state.get("itinerary_plan", {})
        feasible = state.get("itinerary_feasible", True)
        
        if not feasible:
            return self._format_fallback_response(state)

        final_markdown = plan_state.get("final_markdown")
        if final_markdown:
            return {"messages": [AIMessage(content=final_markdown)]}

        return self._format_intermediate_response(state)

    def _format_fallback_response(self, state: AgentState) -> dict:
        """Formats a nice response for the user when the original destination is not feasible and suggests culturally close alternatives"""
        original_dest = state.get("destination_city", "the requested destination")
        origin = state.get("current_city", "")
        alternatives = state.get("alternative_destinations", [])
        reason_raw = state.get("itinerary_fallback_reason", "budget constraints or flight availability")
        reason = self._humanize_reason(reason_raw)

        lines = [
            f"### 🗺️ Oops, we had to recalculate the route...",
            f"We couldn't build a complete travel itinerary for **{original_dest}** that meets all your conditions and budget ({reason}).",
            f"\nBut don't worry! We found **3 amazing alternative destinations** for you with a similar culture, atmosphere, and vibe, with direct flights from {origin}:\n"
        ]

        for i, city in enumerate(alternatives, 1):
            lines.append(f"**{i}. 📍 {city}**")
            lines.append(f"Flights available from {origin}, travel experience and cultural atmosphere very close to {original_dest}.\n")

        lines.append("---")
        lines.append("**Which destination sounds best to you?** Just tell me its name, and I'll build a perfect itinerary for it! 😊")
        
        formatted_text = "\n".join(lines)
        return {"messages": [AIMessage(content=formatted_text)]}

    # ── Reason humanizer ────────────────────────────────────────────────────

    def _humanize_reason(self, reason_raw) -> str:
        """
        Turn the structured/JSON fallback reason into one friendly sentence via
        an LLM call. Falls back to a deterministic plain-text extraction when no
        LLM is configured or the call fails — so we never leak a JSON blob.
        """
        gist = _extract_gist(reason_raw)

        if not self.llm:
            return gist

        try:
            out = self.llm.invoke([
                SystemMessage(content=REASON_HUMANIZE_SYSTEM),
                HumanMessage(content=str(reason_raw)),
            ])
            sentence = (out.content or "").strip().strip('"').strip()
            # Guard against the model echoing JSON/jargon back at us.
            if sentence and "{" not in sentence and "_" not in sentence:
                return sentence
        except Exception:
            pass
        return gist

    def _format_intermediate_response(self, state: AgentState) -> dict:
        """If the graph stops in the middle or requires manual intervention, displays a formatted technical status"""
        plan_state = state.get("itinerary_plan", {})
        results = plan_state.get("step_results", {})
        current_index = state.get("current_step_index", 0)
        
        lines = [
            "### 🛠️ Itinerary Building Status",
            "The system is in the middle of a step-by-step planning process. Here are the steps completed so far:\n"
        ]
        
        for key, val in results.items():
            status = "❌ Error" if isinstance(val, dict) and "error" in val else "✅ Completed"
            if isinstance(val, dict) and val.get("reason") == "cached":
                status = "⏩ Skipped (Fetched from cache)"
            lines.append(f"- **{key}**: {status}")
            
        lines.append(f"\nNext step in line: number {current_index + 1}")

        formatted_text = "\n".join(lines)
        return {"messages": [AIMessage(content=formatted_text)]}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_gist(reason_raw) -> str:
    """
    Pull a human-readable phrase out of the fallback reason without an LLM.
    Handles the JSON replan-context blob and plain strings alike, and strips
    the internal 'max_retries_exceeded after N attempts. Last error:' wrapper.
    """
    if not reason_raw:
        return "we couldn't complete the plan within your constraints"

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
    # Strip the internal retry wrapper, keep only the underlying cause.
    marker = "Last error:"
    if marker in text:
        text = text.split(marker, 1)[1].strip()
    return text or "we couldn't complete the plan within your constraints"