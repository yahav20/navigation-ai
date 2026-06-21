"""Integration smoke test — real Groq LLM call, requires GROQ_API_KEY."""
import os

import pytest
from langchain_core.messages import HumanMessage


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set")
def test_paris_3day_smoke():
    """Full pipeline: Tel Aviv -> Paris, 3-day itinerary, $1500."""
    from agent.core.graph import build_graph

    graph = build_graph(provider="groq")

    # Full details so metadata extraction doesn't ask clarifying questions
    result = graph.invoke(
        {"messages": [HumanMessage(
            content=(
                "Build a 3-day day-by-day itinerary from Tel Aviv to Paris "
                "with a $1500 budget, departing in July"
            )
        )]},
        config={"configurable": {"thread_id": "ci-smoke-001"}},
    )

    messages = result.get("messages", [])
    assert messages, "Graph returned no messages"

    last_content = str(messages[-1].content) if hasattr(messages[-1], "content") else ""
    assert last_content.strip(), "Last message is empty"

    # The graph either produced a travel plan or asked a reasonable follow-up.
    # Either way the response must be non-empty and travel-related.
    lower = last_content.lower()
    travel_words = ["paris", "itinerary", "day", "hotel", "flight", "trip", "travel",
                    "budget", "tel aviv", "july", "plan"]
    assert any(word in lower for word in travel_words), \
        f"Response does not look travel-related: {last_content[:300]}"

    # If the full itinerary planner ran, also validate its structured output
    plan = result.get("itinerary_plan", {})
    if plan.get("final_markdown"):
        assert "Paris" in plan["final_markdown"]
        assert "$" in plan["final_markdown"], "Budget table missing"

# Router intent-classification scenarios (open-ended question -> advisor, explicit
# booking -> new_travel_plan, non-travel query -> out_of_scope, explicit itinerary
# request -> build_itinerary, and the two multi-turn cases above) now live as
# deterministic unit tests in tests/test_unit.py (RouterNode + _StubRouterClassifier).
# They were real-LLM calls here, which meant every CI run on every branch drew
# from one shared Groq API key/quota — see tests/test_unit.py for the equivalents,
# which test the same guardrail logic without any network dependency.
