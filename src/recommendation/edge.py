"""Routing edges for the recommendation graph."""
from recommendation.state import RecommendationState

MAX_STEPS = 4


def rec_should_continue(state: RecommendationState) -> str:
    """Route to tools if the agent issued tool calls, otherwise send to formatter.

    MAX_STEPS is intentionally low — recommendation queries should rarely need more
    than 3 tool turns. A high step count is usually a symptom of the agent over-tooling
    on broad questions (e.g. calling get_city_overview for every city in a list).
    """
    last_message = state["messages"][-1]
    step_count = state.get("step_count", 0)

    if step_count >= MAX_STEPS:
        return "rec_formatter"

    if getattr(last_message, "tool_calls", None):
        return "tools"

    return "rec_formatter"
