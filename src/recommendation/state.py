"""Typed state shared across nodes of the recommendation graph."""
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class RecommendationState(TypedDict):
    """Lightweight state for the high-level travel advisor graph."""

    messages: Annotated[list, add_messages]
    step_count: int
    summary: str
