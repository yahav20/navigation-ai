"""Security gate to validate inputs before LLM invocation."""
from langchain_core.messages import AIMessage
from agent.core.state import AgentState
from security import validate_input

def security_gate_node(state: AgentState) -> dict:
    """Run regex and length security checks before hitting any LLM."""
    messages = state.get("messages", [])
    if not messages:
        return {}

    last_msg = messages[-1]

    if getattr(last_msg, "type", "") != "human":
        return {}

    try:
        validate_input(last_msg.content)
        return {}

    except ValueError as e:
        return {
            "messages": [AIMessage(content=str(e), name="security_gate")]
        }
