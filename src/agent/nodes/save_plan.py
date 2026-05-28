"""HITL node: ask the user whether to save the rendered travel plan to disk."""
from datetime import datetime
from pathlib import Path

from langchain_core.messages import AIMessage
from langgraph.types import interrupt

from agent.state import AgentState

OUT_DIR = Path(__file__).resolve().parents[3] / "out"


def _latest_formatter_markdown(messages: list) -> str:
    for msg in reversed(messages):
        if getattr(msg, "name", "") == "formatter_output":
            return str(msg.content or "")
    return ""


class SavePlanPromptNode:
    """Pause after the formatter and ask the user whether to save the plan as Markdown."""

    def __call__(self, state: AgentState) -> dict:
        markdown = _latest_formatter_markdown(state.get("messages", []))
        if not markdown:
            return {}

        # NOTE: interrupt() raises; everything after it only runs on resume.
        # The node re-runs from the top when resumed, so the markdown read
        # above is idempotent and re-extracts the same content.
        answer = interrupt({
            "question": "Save this travel plan to a file? (yes/no)",
            "default_filename": f"plan-{datetime.now():%Y%m%d-%H%M%S}.md",
        })

        if not isinstance(answer, str) or not answer.strip().lower().startswith("y"):
            return {"messages": [AIMessage(content="📁 Plan not saved.", name="save_plan")]}

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / f"plan-{datetime.now():%Y%m%d-%H%M%S}.md"
        path.write_text(markdown, encoding="utf-8")
        return {"messages": [AIMessage(content=f"💾 Plan saved to `{path}`.", name="save_plan")]}
