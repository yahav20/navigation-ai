"""Security gate to validate inputs before LLM invocation."""
from langchain_core.messages import AIMessage, HumanMessage
from agent.core.state import AgentState
from security import coerce_to_text, validate_input, sanitize_message

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
        text = validate_input(last_msg.content)
    except ValueError as e:
        return {
            "messages": [AIMessage(content=str(e), name="security_gate")]
        }

    # Strip any embedded injection clauses from the validated text
    sanitized = sanitize_message(text)

    # Normalize content type AND replace with sanitized text if anything changed
    if sanitized != text or not isinstance(last_msg.content, str):
        msg_id = getattr(last_msg, "id", None)
        if msg_id:
            return {"messages": [HumanMessage(content=sanitized, id=msg_id)]}

    return {}
