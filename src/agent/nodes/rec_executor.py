"""Executor node — runs the planned tool calls and builds the DATA COLLECTED block."""
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langchain_core.messages import AIMessage

from agent.state import AgentState
from tools.rec_tools import rec_tools

_tool_map = {t.name: t for t in rec_tools}

# Tools where an empty result warrants a fallback attempt
_DISCOVERY_TOOLS = frozenset({"find_destinations_by_tag", "find_destinations_by_vibe"})

# When a discovery tool returns nothing, try the sibling tool with a safe default arg
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


def _format_result(tool_name: str, args: dict, result: Any) -> str:
    """Render a single tool result as a labeled text block."""
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

class RecExecutorNode:
    """Execute planned tool calls in parallel and emit a DATA COLLECTED block."""

    def __call__(self, state: AgentState) -> dict:
        plan = state.get("rec_plan") or []

        if not plan:
            content = "DATA COLLECTED:\n- No data gathered (plan was empty).\nREADY FOR FORMATTING."
            return {
                "messages": [AIMessage(content=content)],
                "rec_last_tool_results": [],
            }

        # --- Run all steps in parallel (all are independent DB reads) ---
        with ThreadPoolExecutor(max_workers=len(plan)) as executor:
            raw_results = list(executor.map(
                lambda step: _run_step(step["tool_name"], step.get("args", {})),
                plan,
            ))

        # --- Fallback: if every result is empty and the plan had a discovery tool,
        #     try the paired fallback tool once ---
        all_empty = all(_is_empty(r) for r in raw_results)
        if all_empty:
            for step in plan:
                tool_name = step["tool_name"]
                if tool_name in _FALLBACK:
                    fb_name, fb_args = _FALLBACK[tool_name]
                    fb_result = _run_step(fb_name, fb_args)
                    if not _is_empty(fb_result):
                        # Prepend the fallback block before the (empty) originals
                        fallback_block = (
                            f"[no results for original query — showing general options via {fb_name}]\n"
                            + "\n".join(
                                _format_result(fb_name, fb_args, fb_result).split("\n")[1:]
                            )
                        )
                        blocks = [fallback_block] + [
                            _format_result(plan[i]["tool_name"], plan[i].get("args", {}), raw_results[i])
                            for i in range(len(plan))
                        ]
                        tool_results = [
                            {"tool_name": fb_name, "args": fb_args, "result": fb_result},
                            *[
                                {"tool_name": plan[i]["tool_name"], "args": plan[i].get("args", {}), "result": raw_results[i]}
                                for i in range(len(plan))
                            ],
                        ]
                        body = "\n\n".join(blocks)
                        content = f"DATA COLLECTED:\n{body}\nREADY FOR FORMATTING."
                        return {
                            "messages": [AIMessage(content=content)],
                            "rec_last_tool_results": tool_results,
                        }
                    break  # only try one fallback

        # --- Normal path: format results in plan order ---
        blocks = [
            _format_result(step["tool_name"], step.get("args", {}), result)
            for step, result in zip(plan, raw_results)
        ]
        tool_results = [
            {"tool_name": step["tool_name"], "args": step.get("args", {}), "result": result}
            for step, result in zip(plan, raw_results)
        ]

        body = "\n\n".join(blocks)
        content = f"DATA COLLECTED:\n{body}\nREADY FOR FORMATTING."
        return {
            "messages": [AIMessage(content=content)],
            "rec_last_tool_results": tool_results,
        }
