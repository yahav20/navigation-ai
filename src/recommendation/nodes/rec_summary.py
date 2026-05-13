"""Rolling summary node for the recommendation agent."""
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import RemoveMessage

from recommendation.state import RecommendationState

MIN_MESSAGES_TO_SUMMARIZE = 3

_SUMMARY_PROMPT = """You are a memory module for a travel advisor agent.
You track PERSISTENT user facts only — NOT topics, vibes, or destinations from this turn.

EXISTING MEMORY:
{existing_summary}

ONLY add a fact to memory if ALL of these are true:
- The user EXPLICITLY stated it (not inferred, not the agent's recommendation)
- It is one of the four allowed fact types:
    1. Home city / origin city  (e.g. "I'm in Tel Aviv", "I'm based in Israel")
    2. Numeric budget ceiling   (e.g. "my budget is $1000")
    3. Numeric flight cap       (e.g. "I hate flights over 4 hours")
    4. Travel companions        (e.g. "with my kids aged 7 and 10")

DO NOT include in memory (these change per question — they are NOT persistent):
- The vibe/style they asked about this turn  (romantic, beach, nature, family, foodie, etc.)
- Destinations the agent recommended or that came up in tool results
- The specific question being asked

If the new conversation contains none of the four allowed fact types, return the existing
memory UNCHANGED — do not add anything new.

OUTPUT FORMAT: Return only the updated memory text — at most 4 short bullets.
Example:
- Home city: Tel Aviv
- Budget: $1000
- Travel companions: kids aged 7 and 10
"""


class RecSummaryNode:
    """Maintain a rolling summary and prune message history to avoid context bloat."""

    def __init__(self, model: BaseChatModel) -> None:
        self.model = model

    def __call__(self, state: RecommendationState) -> dict:
        messages = list(state.get("messages", []))
        existing_summary = state.get("summary", "")

        # Defensive: drop any trailing agent message with unresolved tool_calls.
        # The formatter should have already emitted a RemoveMessage for it, but guard
        # here in case the state hasn't been reduced yet when this node runs.
        if messages and getattr(messages[-1], "tool_calls", None):
            messages = messages[:-1]

        if len(messages) < MIN_MESSAGES_TO_SUMMARIZE:
            return {}

        prompt = _SUMMARY_PROMPT.format(
            existing_summary=existing_summary or "No previous memory.",
        )

        response = self.model.invoke([
            {"role": "system", "content": prompt},
            *messages,
        ])

        new_summary = response.content

        messages_to_keep = set()
        for m in reversed(messages):
            if m.type == "human":
                messages_to_keep.add(m.id)
                break
        if messages:
            messages_to_keep.add(messages[-1].id)

        delete_commands = [
            RemoveMessage(id=m.id)
            for m in messages
            if m.id is not None and m.id not in messages_to_keep
        ]

        return {
            "summary": new_summary,
            "messages": delete_commands,
        }
