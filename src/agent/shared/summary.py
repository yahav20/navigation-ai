"""Conversation summary node for the travel agent."""
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import RemoveMessage

from agent.core.state import AgentState

MIN_MESSAGES_TO_SUMMARIZE = 3


class SummaryNode:
    """Maintain a rolling conversation summary across turns."""

    def __init__(self, extraction_model: BaseChatModel) -> None:
        """Store the chat model used to generate summaries."""
        self.extraction_model = extraction_model

    def __call__(self, state: AgentState) -> dict:
        """Summarize the current state and prune the message history."""
        messages = state.get("messages", [])
        existing_summary = state.get("summary", "")

        if len(messages) < MIN_MESSAGES_TO_SUMMARIZE:
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

        response = self.extraction_model.invoke([
            {"role": "system", "content": summary_prompt},
            *messages,
        ])

        new_summary = response.content

        messages_to_keep_ids = set()

        for m in reversed(messages):
            if m.type == "human":
                messages_to_keep_ids.add(m.id)
                break

        if messages:
            messages_to_keep_ids.add(messages[-1].id)

        delete_commands = [
            RemoveMessage(id=m.id)
            for m in messages
            if m.id is not None and m.id not in messages_to_keep_ids
        ]

        return {
            "summary": new_summary,
            "messages": delete_commands,
            "is_adjustment": False,
        }
