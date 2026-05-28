"""
agent/nodes/itinerary/
======================
Plan & Execute itinerary sub-graph.

Public API:
    from agent.nodes.itinerary import (
        ItineraryEnrichmentNode,
        ItineraryPlannerNode,
        ItineraryExecutorNode,
        ItineraryReplannerNode,
        ItineraryFormatterNode,
    )

Edge functions in itinerary_edges.py:
    after_itinerary_planner
    after_itinerary_replanner

File layout:
    schemas.py              — Pydantic contracts (ExecutionPlan, DaySlot, etc.)
    itinerary_tools.py      — LangChain tools (search_activities, get_weather, etc.)
    itinerary_enrichment.py — ItineraryEnrichmentNode
    planner.py              — ItineraryPlannerNode
    executor.py             — ItineraryExecutorNode
    replanner.py            — ItineraryReplannerNode
    formatter.py            — ItineraryFormatterNode
    schedule_engine.py      — Pure-Python deterministic day scheduler
    activity_selector.py    — LLM-driven activity selection + clustering
"""

from agent.nodes.itinerary.itinerary_enrichment import ItineraryEnrichmentNode
from agent.nodes.itinerary.planner   import ItineraryPlannerNode
from agent.nodes.itinerary.executor  import ItineraryExecutorNode
from agent.nodes.itinerary.replanner import ItineraryReplannerNode
from agent.nodes.itinerary.formatter import ItineraryFormatterNode

__all__ = [
    "ItineraryEnrichmentNode",
    "ItineraryPlannerNode",
    "ItineraryExecutorNode",
    "ItineraryReplannerNode",
    "ItineraryFormatterNode",
]
