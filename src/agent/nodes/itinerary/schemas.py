"""
schemas.py — Pydantic contracts between Planner → Executor → Observer.
Import from here only; never redefine these in other files.
"""
from __future__ import annotations
from typing import List, Literal, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Planner output
# ---------------------------------------------------------------------------

class PlanStep(BaseModel):
    step_id: int
    step_type: Literal[
        "fetch_flights",
        "fetch_return_flights",
        "fetch_hotels",
        "fetch_activities",
        "fetch_weather",
        "build_day_schedule",
        "verify_budget",
    ]
    description: str         
    day: Optional[int] = None


class ExecutionPlan(BaseModel):
    destination: str
    origin: str
    total_days: int
    steps: List[PlanStep]
    retry_count: int = 0


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
