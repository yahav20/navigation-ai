"""
schemas.py — Pydantic contracts between Planner → Executor → Replanner.
Import from here only; never redefine these in other files.
"""
from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Mode
# ---------------------------------------------------------------------------

ItineraryMode = Literal["with_travel_data", "standalone", "update"]

# ---------------------------------------------------------------------------
# Planner output
# ---------------------------------------------------------------------------

class PlanStep(BaseModel):
    """A single execution step in the itinerary plan."""
    model_config = ConfigDict(extra="forbid")

    step_id: int = Field(
        description="The sequential ID of the step."
    )
    step_type: Literal[
        "fetch_activities",
        "fetch_weather",
        "fetch_avg_prices",
        "fetch_min_prices",
        "fetch_special_events",
        "switch_travel_options",
        "build_day_schedule",
        "verify_budget",
        "update_day_schedule",
        "apply_global_preference",
    ] = Field(
        description="The exact type of step to execute."
    )
    description: str = Field(
        description="A short human-readable description of this step."
    )
    day: Optional[int] = Field(
        default=None,
        description="The specific day number — required ONLY for build_day_schedule steps."
    )


class SuggestedAdjustments(BaseModel):
    """
    Constraint relaxations the Planner may suggest when replanning.
    Supported: trip_days (reduce length) or total_budget (raise cap).
    """
    model_config = ConfigDict(extra="forbid")

    trip_days: Optional[int] = Field(
        default=None,
        description="Reduce trip to this many days if not enough activities."
    )
    total_budget: Optional[float] = Field(
        default=None,
        description="Raise budget ceiling to this value if costs are too high."
    )


class ExecutionPlan(BaseModel):
    """The full execution plan produced by the Planner."""
    model_config = ConfigDict(extra="forbid")

    destination: str
    origin: str
    total_days: int
    steps: List[PlanStep]
    retry_count: int = 0
    suggested_adjustments: Optional[SuggestedAdjustments] = Field(
        default=None,
        description=(
            "Use to relax constraints on replan. "
            "Set trip_days to reduce length or total_budget to raise the cap."
        )
    )


# ---------------------------------------------------------------------------
# Step execution result — stored in itinerary_plan["step_results"]
# ---------------------------------------------------------------------------

class StepExecutionResult(BaseModel):
    """Structured result written by the Executor for each completed step."""
    status: Literal["success", "failed", "fallback_used"] = Field(
        description="Outcome of the step."
    )
    data: Any = Field(
        default=None,
        description="Structured data returned by the tool(s). Dict, list, or None."
    )
    error: Optional[str] = Field(
        default=None,
        description="Human-readable error message if the step failed."
    )
    replan_hint: Optional[str] = Field(
        default=None,
        description="Actionable advice for the Planner if this step needs replanning."
    )
    trace: Dict[str, Any] = Field(
        default_factory=dict,
        description="Reasoning trace: thought, action, args, observation, reflection."
    )


# ---------------------------------------------------------------------------
# Day slot — individual time slot in a built day schedule
# ---------------------------------------------------------------------------

class DaySlot(BaseModel):
    time: str
    end_time: str
    duration_minutes: int
    slot_type: Literal["activity", "meal", "rest", "transport", "checkin"]
    name: str
    description: str = ""
    estimated_cost: float = 0.0
    lat: Optional[float] = None
    lng: Optional[float] = None
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Special event — date-specific events found via timeout.com (Tavily)
# ---------------------------------------------------------------------------

class SpecialEvent(BaseModel):
    """A date-specific event extracted from timeout.com and scheduled into itinerary days."""

    name: str
    description: str
    applicable_dates: List[str]       # ["YYYY-MM-DD", ...] dates when event runs
    time_start: str                   # "HH:MM" or "" (unknown/flexible)
    time_end: str                     # "HH:MM" or ""
    cost_usd: float                   # per-person USD estimate (includes snacks for free events)
    suggested_visit_hours: float      # realistic visit duration, not full event length
    is_evening_only: bool             # True when event only makes sense after 18:00
    location_hint: str                # venue/area name for geocoding fallback
    lat: Optional[float] = None       # populated by geocoding after extraction
    lng: Optional[float] = None


# ---------------------------------------------------------------------------
# Itinerary update instruction — extracted from user modification requests
# ---------------------------------------------------------------------------

class DayUpdateInstruction(BaseModel):
    """Structured instruction extracted from a user request to modify their built itinerary."""

    update_type: Literal[
        "exclude_activity",
        "replace_activity",
        "add_activity",
        "exclude_restaurant",
        "replace_restaurant",
        "global_preference",
    ] = Field(description="The type of change the user is requesting.")

    scope: Literal["single_day", "any_day", "all_days"] = Field(
        description=(
            "single_day: user named a specific day. "
            "any_day: user doesn't care which day. "
            "all_days: change applies to every day (e.g. 'no breakfast every day')."
        )
    )

    day: Optional[int] = Field(
        default=None,
        description="Day number (1, 2, 3…) when scope is single_day. Null otherwise.",
    )

    target_name: Optional[str] = Field(
        default=None,
        description="Name of the activity or restaurant to remove / replace. Null if not applicable.",
    )

    replacement_name: Optional[str] = Field(
        default=None,
        description="Explicit name the user wants added in place of the target, or just added. Null if not specified.",
    )

    preference_hint: Optional[str] = Field(
        default=None,
        description="Any extra user context ('I like football', 'something cheaper', 'outdoor').",
    )

    excluded_slot_types: List[str] = Field(
        default_factory=list,
        description=(
            "Slot types to remove from ALL days. "
            "Use ['breakfast'] for 'no breakfast', ['meal'] for all meals, etc. "
            "Only populated for global_preference scope."
        ),
    )
