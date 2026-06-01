"""Integration smoke test — real Groq LLM call, requires GROQ_API_KEY."""
import os

import pytest
from langchain_core.messages import HumanMessage


@pytest.mark.integration
def test_paris_3day_smoke():
    """Full pipeline: Tel Aviv -> Paris, 3-day itinerary, $1500."""
    from agent.graph import build_graph

    assert os.getenv("GROQ_API_KEY"), "GROQ_API_KEY not set in environment"

    graph = build_graph(provider="groq")

    # Use explicit "build itinerary" phrasing to trigger the itinerary planner path
    result = graph.invoke(
        {"messages": [HumanMessage(
            content="Build a 3-day day-by-day itinerary from Tel Aviv to Paris with a $1500 budget"
        )]},
        config={"configurable": {"thread_id": "ci-smoke-001"}},
    )

    messages = result.get("messages", [])
    assert messages, "Graph returned no messages"

    last_content = str(messages[-1].content) if hasattr(messages[-1], "content") else ""
    assert last_content.strip(), "Last message is empty"

    # Check that the response is travel-related (not a refusal or error)
    lower = last_content.lower()
    assert any(word in lower for word in ["paris", "itinerary", "day", "hotel", "flight"]), \
        f"Response does not look like a travel plan: {last_content[:200]}"

    # If the full itinerary planner ran, also validate its structured output
    plan = result.get("itinerary_plan", {})
    if plan.get("final_markdown"):
        assert "Paris" in plan["final_markdown"]
        assert "$" in plan["final_markdown"], "Budget table missing"
        step_results = plan.get("step_results", {})
        assert any(k.startswith("fetch_flights") for k in step_results), "flights step missing"
        assert any(k.startswith("fetch_hotels") for k in step_results), "hotels step missing"
