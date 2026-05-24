"""
agent/nodes/itinerary/
======================
Plan & Execute itinerary sub-graph.

Public API:
    from agent.nodes.itinerary import (
        ItineraryPlannerNode,
        ItineraryExecutorNode,
        ItineraryObserverNode,
        ItineraryFallbackNode,
    )

Edge functions live in agent/edge.py:
    after_itinerary_planner
    after_itinerary_observer
    after_itinerary_fallback

File layout:
    schemas.py          — Pydantic contracts (ExecutionPlan, ObserverOutput, DaySlot)
    itinerary_tools.py  — LangChain tools (search_flights, search_hotels, etc.)
    planner.py          — ItineraryPlannerNode
    executor.py         — ItineraryExecutorNode
    observer.py         — ItineraryObserverNode
    fallback.py         — ItineraryFallbackNode
"""

from agent.nodes.itinerary.planner  import ItineraryPlannerNode
from agent.nodes.itinerary.executor import ItineraryExecutorNode
from agent.nodes.itinerary.observer import ItineraryObserverNode
from agent.nodes.itinerary.fallback import ItineraryFallbackNode

__all__ = [
    "ItineraryPlannerNode",
    "ItineraryExecutorNode",
    "ItineraryObserverNode",
    "ItineraryFallbackNode",
]
