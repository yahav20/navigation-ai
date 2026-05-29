"""Conversation summary node for the travel agent."""
from langchain_core.language_models import BaseChatModel

from agent.llm import silent
from agent.state import AgentState

MIN_MESSAGES_TO_SUMMARIZE = 3
# Cap how many recent messages feed the summary LLM so cost stays bounded even
# though we no longer prune the stored history.
SUMMARY_CONTEXT_WINDOW = 12


class SummaryNode:
    """Maintain a rolling conversation summary across turns."""

    def __init__(self, extraction_model: BaseChatModel) -> None:
        """Store the chat model used to generate summaries."""
        self.extraction_model = extraction_model

    def __call__(self, state: AgentState) -> dict:
        """Update the rolling conversation summary (without pruning history)."""
        messages = state.get("messages", [])
        existing_summary = state.get("summary", "")

        # We only want to summarize if there has been actual progress
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

        response = silent(self.extraction_model).invoke([
            {"role": "system", "content": summary_prompt},
            *messages[-SUMMARY_CONTEXT_WINDOW:],
        ])

        new_summary = response.content

        # IMPORTANT: do NOT prune the message history here. The rolling `summary`
        # string above is what carries memory across turns (nodes inject it into
        # their prompts; e.g. metadata/general_chat). Deleting messages from
        # state would also erase them from any client that renders the live graph
        # state — like agent-chat-ui — making the visible conversation collapse
        # at the end of every turn. Per-turn LLM context is already bounded where
        # it's consumed (e.g. metadata uses messages[-10:]) and by the window above.
        return {
            "summary": new_summary,
            "is_adjustment": False,
        }
