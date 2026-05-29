"""Conversation summary node for the travel agent."""
from langchain_core.language_models import BaseChatModel

from agent.llm import silent
from agent.state import AgentState

# A full exchange (the user's message + the agent's reply) is the smallest unit
# worth folding into memory. Fewer new messages than this and we wait.
MIN_MESSAGES_TO_SUMMARIZE = 2


class SummaryNode:
    """Maintain a rolling conversation summary across turns."""

    def __init__(self, extraction_model: BaseChatModel) -> None:
        """Store the chat model used to generate summaries."""
        self.extraction_model = extraction_model

    def __call__(self, state: AgentState) -> dict:
        """Fold only the messages added since the last summary into memory.

        We track a watermark (`last_summarized_index`) rather than pruning the
        transcript: the full history stays in state (so the chat UI never
        collapses), but each turn the LLM only sees the *new* cycle — the
        rolling `summary` string carries everything older. This restores the
        old per-cycle scope without the destructive RemoveMessage pruning.
        """
        messages = state.get("messages", [])
        existing_summary = state.get("summary", "")

        # How many messages are already captured in `summary`. Clamp in case the
        # history shrank since (e.g. a RemoveMessage emitted by another node).
        start = min(state.get("last_summarized_index", 0), len(messages))
        new_messages = messages[start:]

        # Only summarize once a fresh exchange has accumulated. When we skip we
        # leave the watermark untouched so these messages are folded in later.
        if len(new_messages) < MIN_MESSAGES_TO_SUMMARIZE:
            return {}

        summary_prompt = f"""
            You are a memory management module for a travel agent.
            Your task is to maintain a concise "World State" summary.

            EXISTING MEMORY:
            {existing_summary or "No previous memory."}

            NEW CONVERSATION SEGMENT:
            Analyze the recent messages and update the memory.
            Ensure you keep track of:
            1. Origin city
            2. Destination city
            3. Total budget (and currency)
            4. Any specific preferences or constraints mentioned.
            5. CRITICAL: If the system found specific flights or hotels (e.g., from tool responses),
               summarize their details (airline, flight numbers, hotel names, prices) so the agent
               remembers them for the next turn.
            6. CRITICAL: If the last turn was a destination advisor (the agent recommended
               cities to visit), record the recommended cities by name so the user can refer back
               to them next turn (e.g. "sounds good, let's go there", "book the first one",
               "plan a trip to that city"). This enables the planning agent to resolve the
               destination on the next turn.

            Return ONLY the updated summary text.
            """

        response = silent(self.extraction_model).invoke([
            {"role": "system", "content": summary_prompt},
            *new_messages,
        ])

        # Advance the watermark to the full history length: everything up to here
        # is now reflected in `summary`. We deliberately do NOT prune the stored
        # messages — that would erase them from any client rendering live graph
        # state (e.g. agent-chat-ui) and collapse the visible conversation.
        return {
            "summary": response.content,
            "last_summarized_index": len(messages),
            "is_adjustment": False,
        }
