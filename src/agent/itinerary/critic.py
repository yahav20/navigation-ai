"""
ItineraryCriticNode — Budget Reflection gate between Replanner and Formatter.

Decision logic when over-budget:

  Step 1 — Can cheaper activities cover the overage?
    Condition: critic_attempts == 0  AND  NOT use_min_prices_for_budget
               AND  overage ≤ (activities_cost + meals_cost)
    Action:    one replan with cheaper-activity hint (critic_attempts → 1)

  Step 2 — Overage exceeds what activities can cover, OR step-1 replan failed:
    Standalone mode:
      If use_min_prices_for_budget is False → trigger fetch_min_prices as a proper
      plan step (planner → executor → replanner → critic again with min prices).
      If use_min_prices_for_budget is True  → min prices already verified by
      replanner but still over budget → HITL.
    With-travel-data mode:
      Try swapping to the cheapest available flight/hotel in state.
      If swap covers the gap → switch_travel (replan).
      Otherwise → HITL.

  Pass after min-prices fetch:
    When use_min_prices_for_budget is True and verify_budget_0 is NOT over_budget,
    the critic returns critic_action="min_travel" so the formatter knows to show
    the min-prices header and booking advisory.

Budget is pre-computed by the Replanner after all day schedules are built and
stored under "verify_budget_0" in step_results. The Critic reads it from there
(or recomputes it if absent).
"""
from __future__ import annotations

import json
from typing import Optional

from langgraph.types import interrupt

from agent.itinerary.step_handlers import handle_verify_budget, _extract_activity_costs
from agent.core.state import AgentState

MAX_CRITIC_ATTEMPTS = 3  # safety ceiling (step 1 is still limited to one attempt)


def _plan_for_replan(plan_state: dict, replan_context_json: str) -> dict:
    """Cleaned plan_state for a full schedule rebuild.

    Strips stale build_day_schedule and verify_budget results.
    Preserves fetch_avg_prices and fetch_min_prices (they are cached).
    """
    clean_results = {
        k: v for k, v in plan_state.get("step_results", {}).items()
        if not k.startswith("build_day_schedule") and not k.startswith("verify_budget")
    }
    updated = dict(plan_state)
    updated["step_results"]   = clean_results
    updated["replan_count"]   = 0
    if replan_context_json:
        updated["replan_context"] = replan_context_json
    return updated


def _plan_for_switch_travel(plan_state: dict) -> dict:
    """Cleaned plan_state for a switch_travel_options-only replan.

    Keeps all existing day schedules intact — only strips verify_budget so the
    replanner recomputes it with the newly-selected (cheaper) flight + hotel.
    Because build_day_schedule results are preserved, the planner will see all
    days as completed and emit only switch_travel_options.
    """
    clean_results = {
        k: v for k, v in plan_state.get("step_results", {}).items()
        if not k.startswith("verify_budget")
    }
    updated = dict(plan_state)
    updated["step_results"]   = clean_results
    updated["replan_count"]   = 0
    updated["replan_context"] = json.dumps({
        "error_code":    "need_switch_travel",
        "error_message": "Cheaper activities didn't resolve budget — switching to cheapest travel options.",
        "failed_step":   "verify_budget",
        "replan_hint":   (
            "Add switch_travel_options step. "
            "DO NOT rebuild day schedules — keep the existing schedule unchanged."
        ),
    })
    return updated


def _plan_for_fetch_min(plan_state: dict) -> dict:
    """Cleaned plan_state for a fetch_min_prices-only replan.

    Keeps all existing day schedules intact — only strips verify_budget so
    the replanner recomputes it using the newly-fetched minimum prices.
    """
    clean_results = {
        k: v for k, v in plan_state.get("step_results", {}).items()
        if not k.startswith("verify_budget")
    }
    updated = dict(plan_state)
    updated["step_results"]   = clean_results
    updated["replan_count"]   = 0
    updated["replan_context"] = json.dumps({
        "error_code":    "need_min_prices",
        "error_message": "Overage exceeds activity costs — fetching minimum available prices to re-verify budget.",
        "failed_step":   "verify_budget",
        "replan_hint":   (
            "Add fetch_min_prices step. "
            "DO NOT rebuild day schedules — keep existing schedule unchanged."
        ),
    })
    return updated


class ItineraryCriticNode:
    """Reflection/Critic gate: validates budget and applies a two-step correction strategy."""

    def __call__(self, state: AgentState) -> dict:
        plan_state = state.get("itinerary_plan") or {}
        results    = plan_state.get("step_results", {})
        budget     = float(state.get("total_budget") or 0)
        trip_days  = int(state.get("trip_days") or 1)
        mode       = state.get("itinerary_mode", "standalone")
        use_min    = bool(state.get("use_min_prices_for_budget"))

        if not budget:
            return {"critic_action": "pass"}

        # ── Obtain budget result ───────────────────────────────────────────
        raw_result = results.get("verify_budget_0")
        if raw_result is None:
            raw_result = handle_verify_budget(
                results, budget, trip_days,
                state.get("destination_city", ""),
                state.get("current_city", ""),
                mode, state,
            )
            # Persist so the formatter can read it without recomputing
            results    = {**results, "verify_budget_0": raw_result}
            plan_state = {**plan_state, "step_results": results}

        # If budget is satisfied and we used min prices → signal min_travel to formatter
        if raw_result.get("status") != "over_budget":
            if use_min:
                return {"critic_action": "min_travel", "itinerary_plan": plan_state}
            return {"critic_action": "pass", "itinerary_plan": plan_state}

        data        = raw_result.get("data") or {}
        grand_total = float(data.get("grand_total", 0))
        overage     = grand_total - budget
        critic_attempts = int(state.get("critic_attempts") or 0)

        # ── Step 1: Replan with cheaper activities (once, if overage is coverable) ──
        activities_cost, meals_cost, _ = _extract_activity_costs(results, trip_days)
        total_activity_cost = activities_cost + meals_cost

        if critic_attempts == 0 and not use_min and overage <= total_activity_cost:
            max_act_per_day = max(10.0, round((budget * 0.25) / trip_days, 0))
            replan_hint = (
                f"Budget exceeded by ${overage:.0f} "
                f"(itinerary total: ${grand_total:.0f}, user budget: ${budget:.0f}). "
                "Rebuild selecting only budget-friendly activities. "
                "Avoid premium or expensive venues. "
                "Prefer free or low-cost options (parks, markets, free museums, walking tours). "
                f"Keep estimated activity cost under ${max_act_per_day:.0f} per day."
            )
            updated_plan = _plan_for_replan(plan_state, json.dumps({
                "error_code":    "over_budget",
                "error_message": f"Budget exceeded by ${overage:.0f}.",
                "failed_step":   "verify_budget",
                "replan_hint":   replan_hint,
            }))
            return {
                "critic_attempts": 1,
                "critic_action":   "replan_cheaper",
                "itinerary_plan":  updated_plan,
                "progress_log": [
                    f"🔍 **CRITIC → REPLAN** Budget exceeded by **${overage:.0f}** "
                    f"(total: ${grand_total:.0f}, budget: ${budget:.0f}). "
                    f"Overage (${overage:.0f}) ≤ activity costs (${total_activity_cost:.0f}) "
                    "— rebuilding with cheaper activities."
                ],
            }

        # ── Step 2: Overage too large for activities (or step-1 replan failed) ──

        if mode == "standalone":
            return self._standalone_step2(
                state, plan_state, budget, trip_days, overage, grand_total, use_min
            )
        else:
            return self._with_travel_step2(
                state, plan_state, budget, trip_days, overage, grand_total
            )

    # ── Step 2 — Standalone ───────────────────────────────────────────────

    def _standalone_step2(
        self,
        state: AgentState,
        plan_state: dict,
        budget: float,
        trip_days: int,
        overage: float,
        grand_total: float,
        use_min: bool,
    ) -> dict:
        if not use_min:
            # Trigger fetch_min_prices as a proper plan step
            updated_plan = _plan_for_fetch_min(plan_state)
            return {
                "critic_action":             "fetch_min_prices",
                "use_min_prices_for_budget": True,
                "itinerary_plan":            updated_plan,
                "progress_log": [
                    f"💡 **CRITIC → FETCH MIN PRICES** Overage (${overage:.0f}) exceeds "
                    f"activity costs — dispatching fetch_min_prices step to re-verify budget "
                    "against cheapest available options."
                ],
            }

        # Already used min prices but still over budget → HITL
        return self._hitl(state, budget, grand_total, overage, trip_days, plan_state)

    # ── Step 2 — With-travel-data ─────────────────────────────────────────

    def _with_travel_step2(
        self,
        state: AgentState,
        plan_state: dict,
        budget: float,
        trip_days: int,
        overage: float,
        grand_total: float,
    ) -> dict:
        results = plan_state.get("step_results", {})

        # If switch_travel_options already ran but budget is still over → HITL
        already_switched = any(k.startswith("switch_travel_options") for k in results)
        if already_switched:
            return self._hitl(state, budget, grand_total, overage, trip_days, plan_state)

        # Trigger switch_travel_options through the plan → execute → replan pipeline.
        # _plan_for_switch_travel keeps existing day schedules intact (only strips
        # verify_budget), so the planner emits ONLY switch_travel_options. The
        # replanner then recomputes the budget with the new (cheaper) travel costs
        # against the already-built day costs — no schedule rebuild needed.
        updated_plan = _plan_for_switch_travel(plan_state)
        return {
            "critic_action":         "switch_travel",
            "switch_travel_triggered": True,
            "itinerary_plan":        updated_plan,
            "progress_log": [
                f"🔄 **CRITIC → SWITCH TRAVEL** Overage (${overage:.0f}) exceeds "
                "activity costs — dispatching switch_travel_options step to select "
                "the cheapest available flight + hotel, then rebuilding the schedule."
            ],
        }

    # ── HITL ──────────────────────────────────────────────────────────────

    def _hitl(
        self,
        state: AgentState,
        budget: float,
        grand_total: float,
        overage: float,
        trip_days: int,
        plan_state: dict,
    ) -> dict:
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
            updated_plan = _plan_for_replan(plan_state, "")
            return {
                "critic_action":   "reduce_day",
                "critic_attempts": 0,
                "trip_days":       new_days,
                "itinerary_plan":  updated_plan,
                "progress_log": [
                    f"📅 **CRITIC → REDUCE DAY** Rebuilding as a {new_days}-day itinerary."
                ],
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
            updated_plan = _plan_for_replan(plan_state, json.dumps({
                "error_code":    "user_adjustment",
                "error_message": "User requested preference adjustment after budget exceeded.",
                "failed_step":   "verify_budget",
                "replan_hint":   (
                    f"User preference adjustment request: {adjustment_text}. "
                    "Rebuild the itinerary incorporating these preferences. "
                    "Also keep activity costs modest to stay within budget."
                ),
            }))
            return {
                "critic_action":             "adjust_prefs",
                "critic_attempts":           0,
                "critic_adjustment_request": adjustment_text,
                "itinerary_plan":            updated_plan,
                "progress_log": [
                    f"⚙️ **CRITIC → ADJUST PREFS** Rebuilding with user preferences: '{adjustment_text}'."
                ],
            }

        return {"critic_action": "abort"}

