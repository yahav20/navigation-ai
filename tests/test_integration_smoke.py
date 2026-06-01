"""Integration smoke test — real Groq LLM call, requires GROQ_API_KEY."""
import os

import pytest
from langchain_core.messages import HumanMessage


@pytest.mark.integration
def test_paris_3day_smoke():
    """Full pipeline: Tel Aviv → Paris, 3 days, $1500."""
    from agent.graph import build_graph

    assert os.getenv("GROQ_API_KEY"), "GROQ_API_KEY not set in environment"

    graph = build_graph(provider="groq")
    result = graph.invoke(
        {"messages": [HumanMessage(content="Plan a 3 day trip from Tel Aviv to Paris with $1500")]},
        config={"configurable": {"thread_id": "ci-smoke-001"}},
    )

    plan = result.get("itinerary_plan", {})
    step_results = plan.get("step_results", {})

    assert plan.get("final_markdown"), "No markdown output produced"
    assert "Paris" in plan["final_markdown"]
    assert "$" in plan["final_markdown"], "Budget table missing"
    assert any(k.startswith("fetch_flights") for k in step_results), "flights step missing"
    assert any(k.startswith("fetch_hotels") for k in step_results), "hotels step missing"
    assert any(k.startswith("build_day_schedule") for k in step_results), "day schedule missing"
    assert any(k.startswith("verify_budget") for k in step_results), "budget verify missing"

    budget_key = next(k for k in step_results if k.startswith("verify_budget"))
    budget_data = step_results[budget_key].get("data", {})
    grand_total = budget_data.get("grand_total", 0)
    assert 0 < grand_total <= 1575, f"Grand total ${grand_total} out of expected range"  # noqa: PLR2004
