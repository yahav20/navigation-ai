"""Formatter for general chat responses — passes LLM prose through to the user."""
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from agent.state import AgentState

_SYSTEM_PROMPT = """You are Atlas, a friendly travel assistant.
The agent has collected some information. Rewrite it as a warm, concise reply to the user.
- Use natural conversational prose (no bullet walls unless a list genuinely helps)
- Keep it short — 3-6 sentences max unless the user asked for detail
- End with a light invitation to plan a trip if appropriate
- Do NOT use "DATA COLLECTED" or "READY FOR FORMATTING" — that format is forbidden here
"""


class GeneralChatFormatterNode:
    """Format general chat agent output into a friendly conversational response."""

    def __init__(self, extraction_model: BaseChatModel) -> None:
        self.model = extraction_model

    def __call__(self, state: AgentState) -> dict:
        messages = list(state.get("messages", []))

        # Find the last AI message — that's the general_chat node's raw output
        last_ai = next(
            (m for m in reversed(messages) if getattr(m, "type", "") == "ai"),
            None,
        )

        # If the response has no tool calls it's already conversational prose — pass through
        if last_ai and not getattr(last_ai, "tool_calls", None):
            return {}

        # If there were tool calls, the last AI message is a data summary — reformat it
        response = self.model.invoke([
            {"role": "system", "content": _SYSTEM_PROMPT},
            *messages,
        ])

        return {"messages": [response]}
