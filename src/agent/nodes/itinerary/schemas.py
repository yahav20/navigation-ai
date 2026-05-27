"""
schemas.py — Pydantic contracts between Planner → Executor → Observer.
Import from here only; never redefine these in other files.
"""
from __future__ import annotations
from typing import Any, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field

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
        "fetch_flights",
        "fetch_return_flights",
        "fetch_hotels",
        "fetch_activities",
        "fetch_weather",
        "build_day_schedule",
        "verify_budget",
    ] = Field(
        description="The exact type of step to execute."
    )
    description: str = Field(
        description="A short, human-readable description of what this step does (e.g., 'Fetch hotels in Paris' or 'Build schedule for Day 2')."
    )
    day: Optional[int] = Field(
        default=None,
        description="The specific day number, required ONLY if step_type is 'build_day_schedule'."
    )


class SuggestedAdjustments(BaseModel):
    """
    Constraint relaxations the Planner may suggest when replanning fails.
    Supported keys: trip_days (reduce trip length) or total_budget (raise budget cap).
    """
    model_config = ConfigDict(extra="forbid")

    trip_days: Optional[int] = Field(
        default=None,
        description="Reduce the trip to this many days if there are not enough activities."
    )
    total_budget: Optional[float] = Field(
        default=None,
        description="Raise the budget ceiling to this value if flights/hotels are too expensive."
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
            "Use this to relax constraints if replanning fails. "
            "Set trip_days to reduce trip length or total_budget to raise the budget cap."
        )
    )



class StepExecutionResult(BaseModel):
    """The strict output schema demanded by the Observer."""
    status: Literal["success", "failed", "retrying", "fallback_used"] = Field(
        description="The outcome of the execution step."
    )
    data: dict | list = Field(
        default_factory=dict, 
        description="The structured data returned by the tool(s) used. Can be dict, list, or JSON string."
    )
    error: Optional[str] = Field(
        default=None, 
        description="A clear, human-readable error message if the step failed."
    )
    replan_hint: Optional[str] = Field(
        default=None, 
        description="Actionable advice for the Planner if this step needs to be replanned."
    )
    trace: str = Field(
        default="", 
        description="A brief summary of the reasoning and tools called during execution."
    )


# ---------------------------------------------------------------------------
# Observer output — either done or re-plan
# ---------------------------------------------------------------------------

class FinalResponse(BaseModel):
    status: Literal["complete"] = "complete"
    markdown: str


class RevisedPlan(BaseModel):
    status: Literal["replan"] = "replan"
    reason: str
    remaining_steps: List[PlanStep]
    adjustments: dict = Field(default_factory=dict)


class ObserverOutput(BaseModel):
    result: FinalResponse | RevisedPlan


# ---------------------------------------------------------------------------
# Day slot
# ---------------------------------------------------------------------------

class DaySlot(BaseModel):
    time: str
    duration_minutes: int
    slot_type: Literal["activity", "meal", "rest", "transport"]
    name: str
    description: str
    estimated_cost: float = 0.0
    lat: Optional[float] = None
    lng: Optional[float] = None
    notes: Optional[str] = None