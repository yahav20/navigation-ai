"""
ItineraryCriticNode — Budget Reflection gate between Replanner and Formatter.

Strategy (applied in order):
  1. Pass:             verify_budget returned "success" → route to formatter unchanged.
  2. Replan cheaper:   first over-budget hit → inject budget constraint, loop to planner.
  3. Switch travel:    second hit, with_travel_data mode → silently swap to cheapest
                       available flight/hotel if combined savings cover the gap.
  4. HITL:             still over budget → interrupt() with 4 user options:
                         ignore_budget  → formatter (with over-budget banner)
                         reduce_day     → decrement trip_days, loop to planner
                         adjust_prefs   → second interrupt for free-text, loop to planner
                         abort          → formatter (sorry message)
"""
from __future__ import annotations

import json
from typing import Optional

from langgraph.types import interrupt

from agent.state import AgentState

MAX_CRITIC_ATTEMPTS = 1  # auto-replan attempts before escalating to HITL


class ItineraryCriticNode:
    """Reflection/Critic gate: validates budget and applies a layered correction strategy."""

    def __call__(self, state: AgentState) -> dict:
        plan_state = state.get("itinerary_plan") or {}
        results    = plan_state.get("step_results", {})
        budget     = float(state.get("total_budget") or 0)
        trip_days  = int(state.get("trip_days") or 1)

        # ── 1. Locate verify_budget raw result ────────────────────────────
        budget_key = next((k for k in results if k.startswith("verify_budget")), None)
        if not budget_key or not budget:
            return {"critic_action": "pass"}

        raw_result = results[budget_key]
        status     = raw_result.get("status", "success")

        if status != "over_budget":
            return {"critic_action": "pass"}

        data        = raw_result.get("data") or {}
        grand_total = float(data.get("grand_total", 0))
        overage     = grand_total - budget

        critic_attempts = int(state.get("critic_attempts") or 0)

        # ── 2. First failure: auto replan with cheaper activities ─────────
        if critic_attempts < MAX_CRITIC_ATTEMPTS:
            max_act_per_day = max(10.0, round((budget * 0.25) / trip_days, 0))
            replan_hint = (
                f"Budget exceeded by ${overage:.0f} "
                f"(itinerary total: ${grand_total:.0f}, user budget: ${budget:.0f}). "
                "Rebuild selecting only budget-friendly activities. "
                "Avoid premium or expensive venues. "
                "Prefer free or low-cost options (parks, markets, free museums, walking tours). "
                f"Keep estimated activity cost under ${max_act_per_day:.0f} per day."
            )
            updated_plan = dict(plan_state)
            updated_plan["replan_context"] = json.dumps({
                "error_code":    "over_budget",
                "error_message": f"Budget exceeded by ${overage:.0f}.",
                "failed_step":   "verify_budget",
                "replan_hint":   replan_hint,
            })
            updated_plan["replan_count"] = 0
            return {
                "critic_attempts": critic_attempts + 1,
                "critic_action":   "replan_cheaper",
                "itinerary_plan":  updated_plan,
            }

        # ── 3. Second failure: try silent switch_travel ───────────────────
        switch_result = self._try_switch_travel(state, overage, trip_days)
        if switch_result is not None:
            updated_plan = dict(plan_state)
            updated_plan["replan_context"] = json.dumps({
                "error_code":    "switch_travel",
                "error_message": (
                    f"Switched to cheaper travel options "
                    f"(saving ~${switch_result['savings']:.0f})."
                ),
                "failed_step":   "verify_budget",
                "replan_hint":   (
                    "Rebuild the itinerary with the updated (cheaper) flight and hotel. "
                    "The travel costs have been reduced; ensure activity costs stay modest."
                ),
            })
            updated_plan["replan_count"] = 0
            updates: dict = {
                "critic_action": "switch_travel",
                "itinerary_plan": updated_plan,
            }
            if switch_result.get("outbound_flight"):
                updates["itinerary_selected_outbound_flight"] = switch_result["outbound_flight"]
            if switch_result.get("hotel"):
                updates["itinerary_selected_hotel"] = switch_result["hotel"]
            return updates

        # ── 4. HITL: present options to user ─────────────────────────────
        dest_label = state.get("destination_city") or "your destination"

        user_choice: str = interrupt({
            "question": (
                f"I wasn't able to build an itinerary for **{dest_label}** "
                f"within your **${budget:.0f}** budget.\n\n"
                f"The best I could achieve was **${grand_total:.0f}** "
                f"(over by **${overage:.0f}**).\n\n"
                "How would you like to proceed?"
            ),
            "options": [
                ("ignore_budget", "Continue anyway (I'll mark it as over budget)"),
                ("reduce_day",    f"Reduce the trip by 1 day ({max(1, trip_days - 1)} days instead of {trip_days})"),
                ("adjust_prefs",  "Try with different preferences"),
                ("abort",         "Cancel planning"),
            ],
        })

        choice = (user_choice or "").strip().lower()

        if choice == "ignore_budget":
            return {"critic_action": "ignore_budget"}

        if choice == "reduce_day":
            new_days     = max(1, trip_days - 1)
            updated_plan = dict(plan_state)
            updated_plan["replan_count"] = 0
            return {
                "critic_action":  "reduce_day",
                "critic_attempts": 0,
                "trip_days":      new_days,
                "itinerary_plan": updated_plan,
            }

        if choice == "adjust_prefs":
            mode      = state.get("itinerary_mode", "standalone")
            mode_hint = (
                "I'll rebuild the itinerary with your new preferences "
                "(keeping your current flights and hotel)."
                if mode == "with_travel_data"
                else "I'll rebuild the day-by-day schedule with your new preferences."
            )

            adjustment_text: str = interrupt({
                "question": (
                    f"{mode_hint}\n\n"
                    "Please describe the changes you'd like to make.\n"
                    "For example: *'prefer free outdoor activities'*, "
                    "*'skip expensive restaurants'*, "
                    "*'fewer activities per day'*."
                ),
                "options": [],
            })

            updated_plan = dict(plan_state)
            updated_plan["replan_count"] = 0
            updated_plan["replan_context"] = json.dumps({
                "error_code":    "user_adjustment",
                "error_message": "User requested preference adjustment after budget exceeded.",
                "failed_step":   "verify_budget",
                "replan_hint":   (
                    f"User preference adjustment request: {adjustment_text}. "
                    "Rebuild the itinerary incorporating these preferences. "
                    "Also keep activity costs modest to stay within budget."
                ),
            })
            return {
                "critic_action":             "adjust_prefs",
                "critic_attempts":           0,
                "critic_adjustment_request": adjustment_text,
                "itinerary_plan":            updated_plan,
            }

        # default / "abort"
        return {"critic_action": "abort"}

    # ── Silent switch_travel helper ────────────────────────────────────────

    def _try_switch_travel(
        self, state: AgentState, overage: float, trip_days: int
    ) -> Optional[dict]:
        """
        Returns {savings, outbound_flight, hotel} if switching to cheaper
        available alternatives covers the budget gap. Only for with_travel_data mode.
        """
        if state.get("itinerary_mode") != "with_travel_data":
            return None

        current_outbound  = state.get("itinerary_selected_outbound_flight") or {}
        current_hotel     = state.get("itinerary_selected_hotel") or {}
        travel_plan       = state.get("travel_plan") or {}

        current_flight_price = float(current_outbound.get("price", 0) or 0)
        current_hotel_ppn    = float(current_hotel.get("price_per_night", 0) or 0)

        # Cheapest alternative outbound flight (different flight number)
        all_flights     = [
            f for f in (state.get("flight_options") or [])
            if isinstance(f, dict) and not f.get("message")
        ]
        current_fn      = current_outbound.get("flight_number", "")
        alt_flights     = [f for f in all_flights if f.get("flight_number") != current_fn]
        cheapest_flight = (
            min(alt_flights, key=lambda f: float(f.get("price", 9999)))
            if alt_flights else None
        )

        # Cheapest alternative hotel (different name) from travel_plan suggestions
        all_hotels      = [h for h in travel_plan.get("hotels", []) if isinstance(h, dict)]
        current_hn      = current_hotel.get("name", "")
        alt_hotels      = [h for h in all_hotels if h.get("name", "") != current_hn]
        cheapest_hotel  = (
            min(alt_hotels, key=lambda h: float(h.get("price_per_night", 9999)))
            if alt_hotels else None
        )

        flight_savings = (
            current_flight_price - float(cheapest_flight.get("price", current_flight_price))
            if cheapest_flight else 0.0
        )
        hotel_savings = (
            (current_hotel_ppn - float(cheapest_hotel.get("price_per_night", current_hotel_ppn)))
            * trip_days
            if cheapest_hotel else 0.0
        )
        total_savings = flight_savings + hotel_savings

        if total_savings >= overage:
            return {
                "savings":        total_savings,
                "outbound_flight": cheapest_flight,
                "hotel":           cheapest_hotel,
            }
        return None
