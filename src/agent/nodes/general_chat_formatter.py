"""Formatter for general chat responses — always produces clean user-friendly prose."""
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from agent.state import AgentState

_SYSTEM_PROMPT = """You are Atlas, a friendly AI travel assistant.

Your job is to turn the agent's raw response into a clean, warm reply for the user.

RULES:
- Write in natural conversational prose — no JSON, no dicts, no lists of tuples
- Keep it concise: 2-5 sentences unless the user asked for detail
- If there is useful travel info in the raw response, include it naturally in your reply
- End with a gentle invitation to plan a trip if it fits naturally
- Never output "DATA COLLECTED", "READY FOR FORMATTING", raw Python objects, or JSON
- Never mention that you are reformatting anything — just reply naturally
"""


def _extract_text(content) -> str:
    """Extract plain text from any content format Gemini might return."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
        return " ".join(parts)
    return str(content)


class GeneralChatFormatterNode:
    """Format general chat agent output into a friendly conversational response."""

    def __init__(self, extraction_model: BaseChatModel) -> None:
        self.model = extraction_model

    def __call__(self, state: AgentState) -> dict:
        messages = list(state.get("messages", []))

        # Find the last AI message and extract its text content
        last_ai = next(
            (m for m in reversed(messages) if getattr(m, "type", "") == "ai"),
            None,
        )

        # If already clean prose (no tool calls, plain string content) — still reformat
        # to guarantee consistent friendly tone regardless of model output format
        raw_text = _extract_text(last_ai.content) if last_ai else ""

        response = self.model.invoke([
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Raw agent response:\n{raw_text}\n\nRewrite this as a friendly reply to the user."},
        ])

        return {"messages": [response]}
