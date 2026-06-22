"""LangGraph factory for the Explore feature.

Registered in langgraph.json as "explore". Compiles to a single-node graph
that runs get_explore_snapshots() in parallel, formats the results, and stores
the 5 destination cards in state. No user input or conversation required.
"""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph


class ExploreState(TypedDict):
    cards: list[dict]


async def _explore_node(state: ExploreState) -> dict:
    from agent.explore.snapshot import get_explore_snapshots
    from agent.explore.formatter import format_explore_cards

    snapshots = await get_explore_snapshots()
    cards = format_explore_cards(snapshots)
    return {"cards": cards}


def make_explore_graph() -> CompiledStateGraph:
    """Graph factory referenced by langgraph.json — entry point for the Explore feature."""
    builder: StateGraph = StateGraph(ExploreState)
    builder.add_node("explore", _explore_node)
    builder.add_edge(START, "explore")
    builder.add_edge("explore", END)
    return builder.compile()
