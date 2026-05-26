"""
ItineraryFallbackNode — v2
==========================
Called when Observer detects an unrecoverable issue or MAX_RETRIES reached.

Strategy waterfall:
  budget_exceeded  → try ±1..3 days adjustment
                   → if still over, bump $500
                   → if still impossible, suggest alternatives
  no_flights       → suggest culturally-similar alternatives
  no_hotels        → suggest culturally-similar alternatives
  max_retries      → suggest alternatives
  unknown          → suggest alternatives

State updates:
  itinerary_fallback_action        — what was done
  itinerary_fallback_alternatives  — list[str] of city names
  itinerary_feasible               — True if a retry is warranted
  messages                         — AIMessage for user-facing output
"""
from __future__ import annotations

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Import the updated tools
from .itinerary_tools import fetch_flights
from agent.state import AgentState

# ── Termination invariant ──────────────────────────────────────────────────
# The number of times Fallback may hand control back to the Planner before it
# gives up and suggests alternatives. This counter lives in
# `itinerary_plan["recovery_attempts"]`, is incremented on every retry hand-back,
# and is NEVER reset by Planner/Observer/Fallback. It is the single monotonic
# measure that guarantees the Fallback↔Planner cycle terminates.
MAX_RECOVERY_ATTEMPTS = 2

# Only genuine budget overages may be remedied by adjusting days / bumping budget.
# Every other failure class (quality rejects, empty days, missing flights/hotels)
# cannot be fixed by those levers, so it routes straight to alternatives.
BUDGET_ERROR_CODES = {"BUDGET_EXCEEDED"}


def _parse_reason(reason_raw: str) -> tuple[str, str]:
    """
    Resolve the structured error code from the fallback reason.

    The reason is either a JSON replan-context blob (written by the Observer) or
    a plain string. Returns the *effective* error code to dispatch on: for a
    MAX_RETRIES wrapper this is the `original_error_code` that actually caused
    the failure, so a quality reject that exhausted its retries is not mistaken
    for a budget problem.
    """
    if not reason_raw:
        return "UNKNOWN", ""
    try:
        ctx = json.loads(reason_raw)
    except (json.JSONDecodeError, TypeError):
        # Plain string — classify only the unambiguous budget signal.
        text = reason_raw.lower()
        if "budget" in text:
            return "BUDGET_EXCEEDED", ""
        return "UNKNOWN", ""
    if not isinstance(ctx, dict):
        return "UNKNOWN", ""
    code = ctx.get("error_code", "UNKNOWN")
    if code == "MAX_RETRIES":
        # Unwrap to the failure that actually triggered the retry exhaustion.
        return ctx.get("original_error_code") or "UNKNOWN", ctx.get("error_message", "")
    return code, ctx.get("error_message", "")


CULTURAL_SIMILARITY_PROMPT = """You are a travel geography expert.
Given a destination city, suggest exactly 3 alternative cities that are:
1. In the SAME cultural/geographic region.
2. Similar travel vibe (beach, culture, city-break, food, nature).
3. Likely to have direct or easy connecting flights from the given origin.
4. Within a similar price range.

Return ONLY a JSON array of 3 city name strings.
Example: ["Lisbon", "Barcelona", "Rome"]
No explanation, no markdown, no extra keys.
"""


class ItineraryFallbackNode:
    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm

    def __call__(self, state: AgentState) -> dict:
        reason = state.get("itinerary_fallback_reason", "") or ""
        destination = state.get("destination_city", "")
        origin = state.get("current_city", "")
        budget = state.get("total_budget", 0)
        trip_days = state.get("trip_days", 3)
        plan_state = state.get("itinerary_plan") or {}
        results = plan_state.get("step_results", {})

        # ── Termination invariant ──────────────────────────────────────────
        # recovery_attempts only ever grows (Planner/Observer carry it, never
        # reset it). Once we have already handed back to the Planner the maximum
        # number of times, stop trying to repair and suggest alternatives — this
        # is what guarantees the Fallback↔Planner cycle cannot run forever.
        recovery_attempts = plan_state.get("recovery_attempts", 0)
        if recovery_attempts >= MAX_RECOVERY_ATTEMPTS:
            return self._suggest_alternatives(destination, origin, budget, trip_days)

        # ── Dispatch on the structured error code, not a substring scan ─────
        error_code, _ = _parse_reason(reason)
        if error_code in BUDGET_ERROR_CODES:
            return self._handle_budget(
                results, budget, trip_days, destination, origin,
                plan_state, recovery_attempts,
            )

        # Quality rejects, empty days, missing flights/hotels, unknown failures:
        # none are fixable by day/budget adjustment → suggest alternatives.
        return self._suggest_alternatives(destination, origin, budget, trip_days)


    # ── Budget handler ─────────────────────────────────────────────────────

    def _handle_budget(
        self, results: dict, budget: float, trip_days: int, destination: str,
        origin: str, plan_state: dict, recovery_attempts: int,
    ) -> dict:
        min_flight = _cheapest_price(results, "fetch_flights")
        min_ret = _cheapest_price(results, "fetch_return_flights")

        realistic_hotel = _realistic_hotel_night(results)

        base_cost = min_flight + min_ret  # flights are fixed costs

        # Only REDUCE days: a budget overage is never cured by adding nights, and
        # searching downward makes the day count a strictly-decreasing measure so
        # the search cannot oscillate (the old ±delta list toggled 2↔1 forever).
        for delta in (-1, -2, -3):
            new_days = trip_days + delta

            if new_days < 1:
                continue

            est_cost = (base_cost + realistic_hotel * new_days) * 1.15

            if budget and est_cost <= budget * 1.05:

                # Caching: Preserve expensive fetch results, discard day builds
                cleared_results = {
                    k: v for k, v in results.items()
                    if k.startswith("fetch_flights")
                    or k.startswith("fetch_return_flights")
                    or k.startswith("fetch_hotels")
                }

                return {
                    "trip_days": new_days,
                    "itinerary_fallback_action": f"days_adjusted_to_{new_days}",
                    "itinerary_feasible": True,
                    "itinerary_fallback_alternatives": [],
                    # Retry Loop Setup
                    "itinerary_plan": {
                        **plan_state,
                        # Reset BOTH planner counters so the hand-back actually
                        # produces a fresh plan instead of bouncing off the
                        # Planner's MAX_REPLANS hard-stop right back to here.
                        "retry_count": 0,
                        "replan_count": 0,
                        "observer_reason": "",
                        # Advance the monotonic termination invariant.
                        "recovery_attempts": recovery_attempts + 1,
                        "step_results": cleared_results,
                    },
                    "messages": [AIMessage(
                        content=f"✅ **Fallback:** adjusting trip from `{trip_days}` ➔ `{new_days}` days",
                        name="fallback_log"
                    )]
                }

        # Try +$500 budget bump if day adjustments don't work
        bumped = budget + 500
        est_cost = (base_cost + realistic_hotel * trip_days) * 1.15

        if est_cost <= bumped * 1.05:
            return {
                "total_budget": bumped,
                "itinerary_fallback_action": "budget_bumped_500",
                "itinerary_feasible": True,
                "itinerary_fallback_alternatives": [],
                "itinerary_plan": {
                    **plan_state,
                    "retry_count": 0,
                    "replan_count": 0,
                    "observer_reason": "",
                    "recovery_attempts": recovery_attempts + 1,
                },
                "messages": [AIMessage(
                    content=f"✅ **Fallback:** bumping budget `${budget}` ➔ `${bumped}`",
                    name="fallback_log"
                )],
            }

        # Nothing worked → Drop down to suggest alternatives
        return self._suggest_alternatives(destination, origin, bumped, trip_days)

    # ── Alternative city suggester ─────────────────────────────────────────

    def _suggest_alternatives(self, destination: str, origin: str, budget: float, trip_days: int) -> dict:
        prompt = (
            f"Destination: {destination}\nOrigin: {origin}\n"
            f"Trip days: {trip_days}\nBudget: ${budget or 'flexible'}"
        )
        raw = self.llm.invoke([
            SystemMessage(content=CULTURAL_SIMILARITY_PROMPT),
            HumanMessage(content=prompt),
        ]).content.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip().rstrip("```").strip()

        try:
            alternatives: list[str] = json.loads(raw)
            if not isinstance(alternatives, list):
                alternatives = []
        except Exception:
            alternatives = []

        # Validate: check if flights are actually available for each suggestion
        validated: list[str] = []
        for city in alternatives[:3]:
            try:
                flights = fetch_flights.invoke({"origin": origin, "destination": city})
                if flights:
                    validated.append(city)
            except Exception:
                pass  # skip unavailable

        final = validated or alternatives[:3]
        md = _render_alternatives(destination, final, origin, budget, trip_days)

        return {
            "itinerary_fallback_action": "suggested_alternatives",
            "itinerary_fallback_alternatives": final,
            "itinerary_feasible": False,
            "alternative_destinations": final,
            "messages": [AIMessage(content=md)],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unwrap(val: dict) -> list | dict:
    """Helper to unwrap the v3 Executor StepExecutionResult."""
    if not isinstance(val, dict):
        return val
    if "status" in val and "data" in val:
        return val["data"]
    return val


def _cheapest_price(results: dict, prefix: str) -> float:
    for k, v in results.items():
        if k.startswith(prefix):
            unwrapped = _unwrap(v)
            items = unwrapped if isinstance(unwrapped, list) else [unwrapped]
            prices = [f.get("price", f.get("total_price", 9999)) for f in items if isinstance(f, dict)]
            if prices:
                return min(prices)
    return 0.0


def _realistic_hotel_night(results: dict) -> float:
    for k, v in results.items():
        if k.startswith("fetch_hotels"):
            unwrapped = _unwrap(v)
            
            # Unwrapped might be a list of hotels, or a dict wrapping {"hotels": [...]}
            hotels = unwrapped if isinstance(unwrapped, list) else []
            if isinstance(unwrapped, dict):
                hotels = unwrapped.get("hotels", [])
            
            prices = [float(h.get("price_per_night", 9999)) for h in hotels if isinstance(h, dict)]
            
            if prices:
                prices.sort()
                top_3 = prices[:3]
                return sum(top_3) / len(top_3)
    return 0.0


def _render_alternatives(
    original: str, alternatives: list[str], origin: str,
    budget: float, trip_days: int,
) -> str:
    lines = [
        "# 🗺️ Alternative Destinations\n",
        f"Unfortunately we couldn't build a complete itinerary for **{original}** "
        f"within your constraints (${budget} · {trip_days} days).\n",
        "Here are culturally similar alternatives with available flights:\n",
    ]
    emojis = ["1️⃣", "2️⃣", "3️⃣"]
    for i, city in enumerate(alternatives):
        # Prevent index out of bounds if more than 3 alternatives bypass the slicing somehow
        emoji = emojis[i] if i < len(emojis) else "📍"
        lines.append(f"## {emoji} {city}")
        lines.append(f"Direct or connecting flights available from **{origin}**. "
                     f"Similar culture and travel vibe to {original}.\n")
    lines += [
        "---",
        "💬 Reply with a city name and I'll plan your full itinerary!",
    ]
    return "\n".join(lines)