"""
Pydantic schemas shared across all itinerary nodes.
Import from here — never duplicate these in individual node files.
"""
from __future__ import annotations
from typing import List, Literal, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Plan & Execute — Planner output
# ---------------------------------------------------------------------------

class PlanStep(BaseModel):
    step_id: int
    step_type: Literal["select_flight", "select_hotel", "build_day", "verify_budget"]
    description: str                              # e.g. "Day 3: last day, depart at 18:00"
    depends_on: List[int] = Field(default_factory=list)


class ExecutionPlan(BaseModel):
    destination: str
    origin: str
    total_days: int
    steps: List[PlanStep]


# ---------------------------------------------------------------------------
# Executor output — individual day slots
# ---------------------------------------------------------------------------

class DaySlot(BaseModel):
    time: str                                     # "HH:MM"
    duration_minutes: int
    slot_type: Literal["activity", "meal", "rest", "transport"]
    name: str
    description: str
    estimated_cost: float = 0.0
    lat: Optional[float] = None
    lng: Optional[float] = None
    notes: Optional[str] = None


class BuiltDay(BaseModel):
    day: int
    theme: str
    slots: List[DaySlot]
    day_cost: float = 0.0