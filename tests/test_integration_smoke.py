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


# ---------------------------------------------------------------------------
# Router intent classification — real LLM (GROQ_API_KEY required)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set")
def test_router_classifies_open_ended_question_as_advisor():
    """Router: open-ended destination question must classify as advisor."""
    from agent.core.graph import build_graph
    graph = build_graph(provider="groq")
    result = graph.invoke(
        {"messages": [HumanMessage(content="Where should I travel in Europe in summer?")]},
        config={"configurable": {"thread_id": "ci-router-b1"}},
    )
    assert result.get("intent") == "advisor", f"Got intent: {result.get('intent')}"


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set")
def test_router_classifies_explicit_booking_request_as_new_travel_plan():
    """Router: explicit destination + booking signal must classify as new_travel_plan."""
    from agent.core.graph import build_graph
    graph = build_graph(provider="groq")
    result = graph.invoke(
        {"messages": [HumanMessage(
            content="I want to fly to Tokyo for 5 days with a $2000 budget from Tel Aviv"
        )]},
        config={"configurable": {"thread_id": "ci-router-b2"}},
    )
    assert result.get("intent") == "new_travel_plan", f"Got intent: {result.get('intent')}"


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set")
def test_router_classifies_non_travel_query_as_out_of_scope():
    """Router: math question with zero travel relevance must classify as out_of_scope."""
    from agent.core.graph import build_graph
    graph = build_graph(provider="groq")
    result = graph.invoke(
        {"messages": [HumanMessage(content="What is the square root of 144?")]},
        config={"configurable": {"thread_id": "ci-router-b3"}},
    )
    assert result.get("intent") == "out_of_scope", f"Got intent: {result.get('intent')}"


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set")
def test_router_classifies_explicit_itinerary_request_as_build_itinerary():
    """Router: explicit day-by-day request with all details must classify as build_itinerary."""
    from agent.core.graph import build_graph
    graph = build_graph(provider="groq")
    result = graph.invoke(
        {"messages": [HumanMessage(
            content="Build a 3-day day-by-day itinerary from Tel Aviv to Rome with a $1500 budget"
        )]},
        config={"configurable": {"thread_id": "ci-router-b6"}},
    )
    assert result.get("intent") == "build_itinerary", f"Got intent: {result.get('intent')}"


# ---------------------------------------------------------------------------
# Multi-turn routing — state carried across two graph invocations
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set")
def test_router_multiturn_advisor_then_commit_transitions_to_new_travel_plan():
    """Multi-turn: open advisor question followed by explicit destination commitment.

    Turn 1 must route to advisor.
    Turn 2 must route to new_travel_plan — the router sees the advisor history and
    the TRANSITION RULE in the prompt detects the commitment signal.
    """
    from agent.core.graph import build_graph
    graph = build_graph(provider="groq")
    config = {"configurable": {"thread_id": "ci-router-b4"}}

    result1 = graph.invoke(
        {"messages": [HumanMessage(content="What are popular tourist cities in Japan?")]},
        config=config,
    )
    assert result1.get("intent") == "advisor", (
        f"Turn 1 expected advisor, got: {result1.get('intent')}"
    )

    result2 = graph.invoke(
        {"messages": [HumanMessage(content="Let's go to Tokyo - I'm ready to plan this trip")]},
        config=config,
    )
    assert result2.get("intent") == "new_travel_plan", (
        f"Turn 2 expected new_travel_plan, got: {result2.get('intent')}"
    )


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set")
def test_router_multiturn_enrichment_short_answer_preserved():
    """Multi-turn: short enrichment answer ('5 days') must stay new_travel_plan.

    Turn 1 triggers new_travel_plan + enrichment questions (missing days/budget).
    Turn 2 is a bare short reply — Guardrail 0 must preserve the active intent even
    if the real LLM is confused by the terse answer.
    """
    from agent.core.graph import build_graph
    graph = build_graph(provider="groq")
    config = {"configurable": {"thread_id": "ci-router-b5"}}

    result1 = graph.invoke(
        {"messages": [HumanMessage(content="I want to travel to Paris from Tel Aviv")]},
        config=config,
    )
    assert result1.get("intent") == "new_travel_plan", (
        f"Turn 1 expected new_travel_plan, got: {result1.get('intent')}"
    )

    result2 = graph.invoke(
        {"messages": [HumanMessage(content="5 days")]},
        config=config,
    )
    assert result2.get("intent") == "new_travel_plan", (
        f"Turn 2 expected new_travel_plan (Guardrail 0), got: {result2.get('intent')}"
    )
