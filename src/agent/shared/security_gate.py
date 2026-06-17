"""Security gate to validate inputs before LLM invocation."""
from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, HumanMessage

from agent.core.state import AgentState
from security import coerce_to_text, sanitize_message, validate_input

if TYPE_CHECKING:
    from agent.shared.injection_guard import InjectionGuard

_BLOCKED_MSG = "I'm not able to process that request. Please ask me about travel planning!"


def make_security_gate_node(guard: "InjectionGuard"):
    """Return a security gate node that validates, sanitizes, and LLM-guards each user message."""

    def security_gate_node(state: AgentState) -> dict:
        messages = state.get("messages", [])
        if not messages:
            return {}

        last_msg = messages[-1]

        if getattr(last_msg, "type", "") != "human":
            return {}

        session_id = state.get("session_id", "unknown")

        try:
            text = validate_input(last_msg.content)
        except ValueError as e:
            return {"messages": [AIMessage(content=str(e), name="security_gate")]}

        # Strip embedded injection clauses while keeping the travel question
        sanitized = sanitize_message(text, session_id=session_id)

        # LLM semantic guard — catches creative bypasses that regex misses
        if guard.is_injection(sanitized, session_id=session_id):
            return {"messages": [AIMessage(content=_BLOCKED_MSG, name="security_gate")]}

        # Normalize content type AND replace with sanitized text if anything changed
        if sanitized != text or not isinstance(last_msg.content, str):
            msg_id = getattr(last_msg, "id", None)
            if msg_id:
                return {"messages": [HumanMessage(content=sanitized, id=msg_id)]}

        return {}

    return security_gate_node
