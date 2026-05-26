"""Executor node — runs ONE planned tool call and accumulates results for the current turn."""
import json
from typing import Any

from agent.state import AgentState
from tools.advisor_tools import advisor_tools

_tool_map = {t.name: t for t in advisor_tools}

_DISCOVERY_TOOLS = frozenset({"find_destinations_by_tag", "find_destinations_by_vibe"})

_FALLBACK: dict[str, tuple[str, dict]] = {
    "find_destinations_by_tag":  ("find_destinations_by_vibe", {"category": "Sightseeing"}),
    "find_destinations_by_vibe": ("find_destinations_by_tag",  {"tag": "city-break"}),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_empty(result: Any) -> bool:
    """Return True when a tool result carries no usable data."""
    if isinstance(result, list):
        return not result or (len(result) == 1 and "message" in result[0])
    if isinstance(result, dict):
        return "message" in result
    return False


def _run_step(tool_name: str, args: dict) -> Any:
    """Invoke a single tool, returning the result or an error dict."""
    tool = _tool_map.get(tool_name)
    if tool is None:
        return {"message": f"Unknown tool '{tool_name}', skipped."}
    try:
        return tool.invoke(args)
    except Exception as exc:  # noqa: BLE001
        return {"message": f"Error calling {tool_name}: {exc}"}


def format_tool_result(tool_name: str, args: dict, result: Any) -> str:
    """Render a single tool result as a labeled text block (shared with replanner)."""
    args_str = ", ".join(f'{k}="{v}"' for k, v in args.items())
    header = f"[{tool_name}({args_str})]"

    if isinstance(result, list):
        if not result:
            return f"{header}\n- No results found."
        if len(result) == 1 and "message" in result[0]:
            return f"{header}\n- {result[0]['message']}"
        lines = [header]
        for item in result:
            if isinstance(item, dict):
                parts = []
                for k, v in item.items():
                    parts.append(f"{k}: {json.dumps(v) if isinstance(v, (dict, list)) else v}")
                lines.append("- " + " | ".join(parts))
            else:
                lines.append(f"- {item}")
        return "\n".join(lines)

    if isinstance(result, dict):
        if "message" in result:
            return f"{header}\n- {result['message']}"
        lines = [header]
        for k, v in result.items():
            if isinstance(v, dict):
                for sk, sv in v.items():
                    lines.append(
                        f"- {k} / {sk}: {', '.join(str(x) for x in sv)}"
                        if isinstance(sv, list)
                        else f"- {k} / {sk}: {sv}"
                    )
            elif isinstance(v, list):
                if v and isinstance(v[0], dict):
                    for sub in v:
                        lines.append("- " + " | ".join(f"{sk}: {sv}" for sk, sv in sub.items()))
                else:
                    lines.append(f"- {k}: {', '.join(str(x) for x in v)}")
            else:
                lines.append(f"- {k}: {v}")
        return "\n".join(lines)

    return f"{header}\n- {result}"


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class AdvisorExecutorNode:
    """Execute the first planned tool call and append the result to accumulated state."""

    def __call__(self, state: AgentState) -> dict:
        plan = state.get("advisor_plan") or []
        past_results = list(state.get("advisor_last_tool_results") or [])

        if not plan:
            return {"advisor_last_tool_results": past_results}

        step = plan[0]
        tool_name = step["tool_name"]
        args = step.get("args", {})

        # Guard: skip if this exact call was already executed this turn
        already_ran = any(
            r["tool_name"] == tool_name and r["args"] == args
            for r in past_results
        )
        if already_ran:
            print(f"\n[Executor] Skipping duplicate call: {tool_name}({args})")
            return {"advisor_last_tool_results": past_results}

        print(f"\n[Executor] Executing: {tool_name}({args})")
        result = _run_step(tool_name, args)

        # Fallback: if this step returned empty and it's a discovery tool, try the sibling once.
        # When fallback succeeds, store the result under the ORIGINAL tool's name/args so
        # the plan tool count stays accurate (the fallback is a transparent retry, not a new step).
        if _is_empty(result) and tool_name in _DISCOVERY_TOOLS:
            fb_name, fb_args = _FALLBACK[tool_name]
            fb_already_ran = any(
                r["tool_name"] == fb_name and r["args"] == fb_args
                for r in past_results
            )
            if not fb_already_ran:
                fb_result = _run_step(fb_name, fb_args)
                if not _is_empty(fb_result):
                    print(f"\n[Executor] Fallback triggered: {fb_name}({fb_args})")
                    past_results.append({"tool_name": tool_name, "args": args, "result": fb_result})
                    return {"advisor_last_tool_results": past_results}

        past_results.append({"tool_name": tool_name, "args": args, "result": result})
        return {"advisor_last_tool_results": past_results}
