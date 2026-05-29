"""Security gate to validate inputs before LLM invocation."""
from langchain_core.messages import AIMessage, HumanMessage
from agent.state import AgentState
from security import coerce_to_text, validate_input

def security_gate_node(state: AgentState) -> dict:
    """Run regex and length security checks before hitting any LLM."""
    messages = state.get("messages", [])
    if not messages:
        return {}

    last_msg = messages[-1]

    # Check only human messages
    if getattr(last_msg, "type", "") != "human":
        return {}

    try:
        validate_input(last_msg.content)
    except ValueError as e:
        # Invalid/malicious input detected
        return {
            "messages": [AIMessage(content=str(e), name="security_gate")]
        }

    # Chat UIs (e.g. agent-chat-ui) send content as a list of content blocks,
    # but downstream nodes assume a plain string (e.g. `router` calls
    # `.content.lower()`). Normalize to text in-place — replacing the message by
    # id — so the rest of the graph behaves exactly like the string-based CLI.
    if not isinstance(last_msg.content, str) and getattr(last_msg, "id", None):
        return {"messages": [HumanMessage(content=coerce_to_text(last_msg.content), id=last_msg.id)]}

    return {}
