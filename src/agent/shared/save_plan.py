"""HITL node: ask the user whether to save the rendered travel plan to disk."""
from pathlib import Path

from langchain_core.messages import AIMessage
from langgraph.types import interrupt

from agent.core.state import AgentState

OUT_DIR = Path(__file__).resolve().parents[3] / "out"


def _latest_formatter_markdown(messages: list) -> str:
    for msg in reversed(messages):
        if getattr(msg, "name", "") == "formatter_output":
            return str(msg.content or "")
    return ""


def _plan_filename(state: AgentState) -> str:
    """Deterministic filename so the name shown in the prompt matches the saved
    file even though the node re-runs from the top on resume."""
    dest = str(state.get("destination_city") or "").strip().lower().replace(" ", "-")
    start = str(state.get("trip_start") or "").strip()
    stem = "-".join(p for p in ("plan", dest or "trip", start) if p)
    return f"{stem}.md"


def _parse_decision(answer: object) -> tuple[bool, str | None]:
    """Resolve a resume payload into (save?, edited_filename).

    Handles both shapes:
    - agent-chat-ui Agent Inbox: ``{"decisions": [{"type": "approve"|"reject"|"edit", ...}]}``
    - CLI / plain string: ``"yes"`` / ``"no"``
    """
    if isinstance(answer, dict) and isinstance(answer.get("decisions"), list):
        decisions = answer["decisions"]
        if not decisions or not isinstance(decisions[0], dict):
            return False, None
        decision = decisions[0]
        dtype = decision.get("type")
        if dtype == "approve":
            return True, None
        if dtype == "edit":
            args = (decision.get("edited_action") or {}).get("args") or {}
            edited = args.get("filename") if isinstance(args, dict) else None
            return True, edited
        return False, None  # reject / unknown

    if isinstance(answer, str):
        return answer.strip().lower().startswith("y"), None

    return False, None


class SavePlanPromptNode:
    """Pause after the formatter and ask the user whether to save the plan as Markdown."""

    def __call__(self, state: AgentState) -> dict:
        markdown = _latest_formatter_markdown(state.get("messages", []))
        if not markdown:
            return {}

        filename = _plan_filename(state)

        # Agent Inbox `HITLRequest` schema — agent-chat-ui renders native
        # Approve / Reject buttons for this and resumes with
        # {"decisions": [{"type": "approve"|"reject", ...}]}. The CLI resumes
        # with a plain "yes"/"no" string; both are handled by _parse_decision.
        # NOTE: interrupt() raises and the node re-runs from the top on resume,
        # so everything above must be idempotent (it is).
        answer = interrupt({
            "action_requests": [
                {
                    "name": "save_plan",
                    "args": {"filename": filename},
                    "description": f"Save this travel plan to `out/{filename}`?",
                }
            ],
            "review_configs": [
                {
                    "action_name": "save_plan",
                    "allowed_decisions": ["approve", "reject"],
                }
            ],
        })

        save, edited_name = _parse_decision(answer)
        if not save:
            return {"messages": [AIMessage(content="📁 Plan not saved.", name="save_plan")]}

        # Sanitize: strip any directory components from a (possibly edited) name.
        safe_name = Path(edited_name or filename).name or filename
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / safe_name
        path.write_text(markdown, encoding="utf-8")
        return {"messages": [AIMessage(content=f"💾 Plan saved to `{path}`.", name="save_plan")]}
