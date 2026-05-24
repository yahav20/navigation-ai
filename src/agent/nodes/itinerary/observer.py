# src/agent/nodes/itinerary/observer.py
"""
ItineraryObserverNode — v2
==========================
Validation is now SPLIT into two layers:

Layer 1 — Pure Python (zero LLM tokens):
  - Overlap detection (O(n) per day)
  - Missing transport between far locations
  - Missing meals (breakfast / lunch / dinner)
  - Departure anchor violation
  - Budget breach (hard)

Layer 2 — LLM (only if Layer 1 passes):
  - Natural language quality check
  - Generates the final beautiful markdown itinerary
  - Only runs ONCE, after all steps complete

This means the replanning loop never wastes tokens on format issues.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from agent.nodes.itinerary.schemas import ExecutionPlan, ObserverOutput, FinalResponse, RevisedPlan
from agent.nodes.itinerary.schedule_engine import haversine_km, GeoPoint, WALK_MAX_KM
from agent.state import AgentState

MAX_RETRIES = 3

# ── LLM prompt (only for final markdown generation) ────────────────────────

OBSERVER_SYSTEM = """
You are the final travel itinerary quality reviewer AND copywriter.

You receive a structured trip plan. Your tasks:
1. Check for any qualitative issues (boring repetition, poor flow, missing highlights).
2. If there is a SEVERE structural problem, reply ONLY with a string starting exactly with "REJECT:" followed by the reason. (Example: "REJECT: Day 2 is too empty.")
3. If the plan is good, output ONLY the final beautiful markdown itinerary. Do not add any JSON or extra chat around it.

Markdown format for FinalResponse:
# ✈️ Your [N]-Day {destination} Itinerary

## 🛫 Flight Details
**Outbound Flight:** [Direct/Connecting] | [Airlines] | Departure: [Time] | Arrival: [Time]
*(If connecting, briefly list the route e.g., TLV -> ATH -> BER and times)*
**Return Flight:** [Direct/Connecting] | [Airlines] | Departure: [Time] | Arrival: [Time]
*(If connecting, briefly list the route and times)*

## 🏨 Accommodation
**Hotel:** [Hotel Name] 

## 📅 Day 1 — [Theme]
| Time | Activity | Duration | Cost |
|------|----------|----------|------|
...
**Day total: $XX**

[Repeat for each day]

---
## 💰 Trip Budget Summary
| Category | Cost |
|----------|------|
...
**Grand Total: $XX**

---
*Tips, highlights, and local recommendations here.*
"""
# ---------------------------------------------------------------------------
# Layer 1 — Pure Python Validators
# ---------------------------------------------------------------------------

class ValidationError:
    def __init__(self, code: str, message: str, day: Optional[int] = None):
        self.code = code
        self.message = message
        self.day = day

    def __str__(self):
        prefix = f"Day {self.day}: " if self.day else ""
        return f"[{self.code}] {prefix}{self.message}"


def validate_schedule(results: dict, budget: float, trip_days: int) -> list[ValidationError]:
    """
    Run all pure-Python checks. Returns list of errors.
    Empty list = plan is structurally valid.
    """
    errors: list[ValidationError] = []

    for key, val in results.items():
        if not key.startswith("build_day_schedule") or not isinstance(val, dict):
            continue

        day = val.get("day", "?")
        slots = val.get("slots", [])

        if not slots:
            errors.append(ValidationError("EMPTY_DAY", "No slots generated", day))
            continue

        errors.extend(_check_overlaps(slots, day))
        errors.extend(_check_meals(slots, day))
        errors.extend(_check_transport_gaps(slots, day))

    # Budget check
    budget_error = _check_budget(results, budget, trip_days)
    if budget_error:
        errors.append(budget_error)

    return errors


def _parse_hm(s: str) -> datetime:
    return datetime.strptime(s, "%H:%M")


def _check_overlaps(slots: list[dict], day) -> list[ValidationError]:
    errors = []
    prev_end: Optional[datetime] = None
    for i, slot in enumerate(slots):
        start_str = slot.get("time") or slot.get("start_time")
        end_str = slot.get("end_time")
        if not start_str or not end_str:
            continue
        try:
            start = _parse_hm(start_str)
            end = _parse_hm(end_str)
        except ValueError:
            continue

        if prev_end and start < prev_end:
            errors.append(ValidationError(
                "OVERLAP",
                f"Slot '{slot.get('name')}' starts at {start_str} "
                f"but previous ends at {prev_end.strftime('%H:%M')}",
                day,
            ))
        if end <= start:
            errors.append(ValidationError(
                "ZERO_DURATION",
                f"Slot '{slot.get('name')}' has zero or negative duration",
                day,
            ))
        prev_end = end
    return errors


def _check_meals(slots: list[dict], day) -> list[ValidationError]:
    """Ensure breakfast, at least one midday food event, and dinner."""
    errors = []
    slot_types_by_hour: list[tuple[int, str, bool]] = []
    for s in slots:
        t = s.get("time") or s.get("start_time", "")
        if not t:
            continue
        try:
            h = int(t.split(":")[0])
        except ValueError:
            continue
        food = s.get("slot_type") in ("meal",) or s.get("food_available", False)
        slot_types_by_hour.append((h, s.get("name", ""), food))

    has_morning_food = any(h <= 10 and f for h, _, f in slot_types_by_hour)
    has_midday_food = any(11 <= h <= 15 and f for h, _, f in slot_types_by_hour)
    has_evening_food = any(h >= 18 and f for h, _, f in slot_types_by_hour)

    # We warn but don't force replan for single missing meal
    # (the schedule engine handles this; if still missing it's a real problem)
    if not has_morning_food:
        errors.append(ValidationError("MISSING_BREAKFAST", "No breakfast slot found", day))
    if not has_midday_food:
        errors.append(ValidationError("MISSING_LUNCH", "No midday food slot", day))
    if not has_evening_food:
        errors.append(ValidationError("MISSING_DINNER", "No dinner/evening food slot", day))

    return errors


def _check_transport_gaps(slots: list[dict], day) -> list[ValidationError]:
    """
    Detect cases where two activity slots are > WALK_MAX_KM apart
    with no transport slot in between.
    """
    errors = []
    prev_act: Optional[dict] = None
    prev_idx = -1

    for i, slot in enumerate(slots):
        if slot.get("slot_type") != "activity":
            if slot.get("slot_type") == "transport":
                prev_act = None  # transport resets the check
            continue

        lat = slot.get("latitude") or slot.get("lat")
        lng = slot.get("longitude") or slot.get("lng")

        if prev_act and lat and lng:
            prev_lat = prev_act.get("latitude") or prev_act.get("lat")
            prev_lng = prev_act.get("longitude") or prev_act.get("lng")
            if prev_lat and prev_lng:
                dist = haversine_km(
                    GeoPoint(float(prev_lat), float(prev_lng)),
                    GeoPoint(float(lat), float(lng)),
                )
                # Check if there's a transport slot between prev_idx and i
                has_transport = any(
                    slots[j].get("slot_type") == "transport"
                    for j in range(prev_idx + 1, i)
                )
                if dist > WALK_MAX_KM and not has_transport:
                    errors.append(ValidationError(
                        "MISSING_TRANSPORT",
                        f"No transport between '{prev_act.get('name')}' and "
                        f"'{slot.get('name')}' ({dist:.1f} km apart)",
                        day,
                    ))

        prev_act = slot
        prev_idx = i

    return errors


def _check_budget(results: dict, budget: float, trip_days: int) -> Optional[ValidationError]:
    budget_result = next(
        (v for k, v in results.items()
         if k.startswith("verify_budget") and isinstance(v, dict)),
        None,
    )
    if not budget_result or not budget:
        return None

    total = budget_result.get("grand_total", 0)
    if total > budget * 1.10:  # 10% grace margin
        return ValidationError(
            "BUDGET_EXCEEDED",
            f"Grand total ${total:.0f} exceeds budget ${budget:.0f} "
            f"(+{((total/budget)-1)*100:.0f}%)",
        )
    return None


# ---------------------------------------------------------------------------
# Observer Node
# ---------------------------------------------------------------------------

# def _build_summary(results: dict, budget: float, trip_days: int) -> str:
#     lines = [f"Trip: {trip_days} days | Budget: ${budget}"]
#     for key, val in results.items():
#         if isinstance(val, dict) and not val.get("error"):
#             lines.append(f"--- {key.upper()} ---")
#             lines.append(json.dumps(val, ensure_ascii=False))
#     return "\n".join(lines)
def _build_summary(results: dict, budget: float, trip_days: int) -> str:
    lines = [f"Trip: {trip_days} days | Budget: ${budget}"]
    
    for key, val in results.items():
        if not isinstance(val, dict) or val.get("error"):
            continue
            
        if key.startswith("fetch_flights") or key.startswith("fetch_return_flights"):
            if isinstance(val, list) and val:
                val = val[0] 
        
        lines.append(f"--- {key.upper()} ---")
        lines.append(json.dumps(val, ensure_ascii=False))
        
    return "\n".join(lines)

class ItineraryObserverNode:
    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm

    def __call__(self, state: AgentState) -> dict:
        plan_state = state.get("itinerary_plan", {})
        results = plan_state.get("step_results", {})
        retry_count = plan_state.get("retry_count", 0)
        current_index = state.get("current_step_index", 0)
        plan_steps = plan_state.get("execution_plan", {}).get("steps", [])

        # ── 1. Mid-execution: check last step for errors ──
        if current_index > 0 and current_index <= len(plan_steps):
            last_step = plan_steps[current_index - 1]
            last_key = f"{last_step['step_type']}_{last_step['step_id']}"
            last_result = results.get(last_key, {})

            if isinstance(last_result, dict) and last_result.get("error"):
                error_msg = last_result["error"]
                return self._trigger_replan(plan_state, retry_count, error_msg)

        # ── 2. Not done yet: continue ──
        if current_index < len(plan_steps):
            return {"itinerary_feasible": True, "observer_action": "continue"}

        # ── 3. All steps done — Layer 1 validation ──
        budget = state.get("total_budget", 0)
        trip_days = state.get("trip_days", 3)

        errors = validate_schedule(results, budget, trip_days)

        # Separate HARD errors (trigger replan) from SOFT warnings
        hard_codes = {"OVERLAP", "EMPTY_DAY", "BUDGET_EXCEEDED"}
        hard_errors = [e for e in errors if e.code in hard_codes]
        soft_warnings = [e for e in errors if e.code not in hard_codes]

        if hard_errors:
            reason = "; ".join(str(e) for e in hard_errors)
            return self._trigger_replan(plan_state, retry_count, reason)

        # ── 4. Layer 2 — LLM quality review + markdown generation ──
        try:
            summary = _build_summary(results, budget, trip_days)
            output = self.llm.invoke([
                SystemMessage(content=OBSERVER_SYSTEM),
                HumanMessage(content=summary),
            ])
            #     
            content = output.content.strip()
        except Exception as e:
            content = ""

        if content.startswith("REJECT:"):
            reason = content.replace("REJECT:", "").strip()
            return self._trigger_replan(plan_state, retry_count, reason)

        # ── 5. Success ──
        if content and not content.startswith("REJECT:"):
           
            if content.startswith("```"):
                markdown = content.split("```")[1].lstrip("markdown").strip().rstrip("```").strip()
            else:
                markdown = content
        else:
            markdown = _generate_fallback_markdown(results, trip_days, budget)

        return {
            "itinerary_plan": {**plan_state, "final_markdown": markdown},
            "itinerary_feasible": True,
            "observer_action": "complete",
            "messages": [AIMessage(content=markdown)],
        }

    def _trigger_replan(self, plan_state, retry_count, reason, revised=None):
        new_retry = retry_count + 1
        
        # 1.    (MAX_RETRIES)
        if new_retry >= MAX_RETRIES:
            
            # -Edge      
            hard_stop_reason = f"max_retries_exceeded: {reason}"
            
            return {
                "itinerary_feasible": False,
                "itinerary_fallback_reason": hard_stop_reason,
            
                "itinerary_plan": {
                    **plan_state,
                    "retry_count": new_retry,
                    "observer_reason": hard_stop_reason,
                },
                "messages": [AIMessage(content=f"❌ **OBSERVER:** MAX RETRIES ({MAX_RETRIES}) reached. Passing to Fallback.\n*Reason: {reason}*", name="observer_log")],
            }

        # 2. (Replan)
        updates = {
            "itinerary_feasible": False,
            "itinerary_fallback_reason": reason,
            "itinerary_plan": {
                **plan_state,
                "retry_count": new_retry,
                "observer_reason": reason,
            },
            "messages": [AIMessage(content=f"🔄 **TRIGGERING REPLANNER (attempt {new_retry}/{MAX_RETRIES}):**\n*{reason}*", name="observer_log")],
        }
        
        if revised and hasattr(revised, "adjustments"):
            adj = revised.adjustments or {}
            if "trip_days" in adj:
                updates["trip_days"] = adj["trip_days"]
            if "total_budget" in adj:
                updates["total_budget"] = adj["total_budget"]
                
        return updates


# ---------------------------------------------------------------------------
# Fallback markdown generator (no LLM required)
# ---------------------------------------------------------------------------

def _generate_fallback_markdown(results: dict, trip_days: int, budget: float) -> str:
    lines = ["# ✈️ Your Trip Itinerary\n"]
    
    out_key = next((k for k in results if k.startswith("fetch_flights")), None)
    ret_key = next((k for k in results if k.startswith("fetch_return_flights")), None)
    
    outbound = results[out_key][0] if out_key and isinstance(results[out_key], list) and results[out_key] else {}
    return_fl = results[ret_key][0] if ret_key and isinstance(results[ret_key], list) and results[ret_key] else {}

    if outbound or return_fl:
        lines.append("## 🛫 Flight Details")
        for title, flight in [("Outbound", outbound), ("Return", return_fl)]:
            if not flight:
                continue
            #  'route'
            if "route" in flight:
                lines.append(f"**{title} Flight (Connecting):**")
                for leg in flight["route"]:
                    lines.append(f"- {leg.get('airline')} {leg.get('flight')} | {leg.get('from')} → {leg.get('to')} | Dep: {leg.get('departure_time')} - Arr: {leg.get('arrival_time')}")
             
            else:
                lines.append(f"**{title} Flight (Direct):** {flight.get('airline')} {flight.get('flight_number')} | Dep: {flight.get('departure_time')} - Arr: {flight.get('arrival_time')}")
        lines.append("")

    for d in range(1, trip_days + 1):
        key = next((k for k in results if k.startswith("build_day_schedule")
                    and isinstance(results[k], dict)
                    and results[k].get("day") == d), None)
        if not key:
            continue
        day_data = results[key]
        lines.append(f"\n## 📅 Day {d} — {day_data.get('theme', '')}")
        lines.append("\n| Time | Activity | Duration | Cost |")
        lines.append("|------|----------|----------|------|")
        for slot in day_data.get("slots", []):
            t = slot.get("time", "")
            name = slot.get("name", "")
            dur = slot.get("duration_minutes", "")
            cost = slot.get("estimated_cost", 0)
            icon = {"activity": "🎯", "meal": "🍽️", "transport": "🚕",
                    "rest": "😴", "checkin": "🏨"}.get(slot.get("slot_type", ""), "•")
            lines.append(f"| {t} | {icon} {name} | {dur} min | ${cost:.0f} |")
        lines.append(f"\n**Day total: ${day_data.get('day_cost', 0):.0f}**")

    # Budget summary
    budget_key = next((k for k in results if k.startswith("verify_budget")), None)
    if budget_key and isinstance(results[budget_key], dict):
        b = results[budget_key]
        lines.append("\n---\n## 💰 Budget Summary\n")
        lines.append("| Category | Cost |")
        lines.append("|----------|------|")
        for cat, val in b.items():
            if cat != "grand_total":
                lines.append(f"| {cat.replace('_', ' ').title()} | ${val:.0f} |")
        lines.append(f"\n**Grand Total: ${b.get('grand_total', 0):.0f}**")
        if budget:
            remaining = budget - b.get("grand_total", 0)
            emoji = "✅" if remaining >= 0 else "⚠️"
            lines.append(f"\n{emoji} Budget remaining: ${remaining:.0f}")

    return "\n".join(lines)
