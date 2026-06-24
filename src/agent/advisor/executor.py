"""Executor node — runs ONE planned tool call and accumulates results for the current turn."""
import calendar
import json
from datetime import date
from typing import Any

from agent.core.state import AgentState
from security import scan_tool_output
from tools.destinations import advisor_tools
from tools.flights import fetch_flights
from ui import render_node, render_node_status

_tool_map = {t.name: t for t in advisor_tools} | {"fetch_flights": fetch_flights}

_MONTH_NUMS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}


def _month_to_iso(month: str | None) -> str | None:
    """Convert a 'Month YYYY' (or bare 'Month') concert/event filter into 'YYYY-MM'.

    With an explicit year, that year is used. With a bare month name, it resolves
    to the next future occurrence (this year if still ahead, otherwise next year) —
    matching the metadata extractor's FUTURE-MONTH rule. Returns None if unparseable.
    """
    if not month:
        return None
    parts = month.strip().split()
    if not parts:
        return None
    num = _MONTH_NUMS.get(parts[0].lower())
    if not num:
        return None

    if len(parts) >= 2 and parts[1].isdigit() and len(parts[1]) == 4:
        return f"{int(parts[1]):04d}-{num:02d}"

    today = date.today()
    year = today.year if num >= today.month else today.year + 1
    return f"{year:04d}-{num:02d}"


def extract_trip_total_usd(state: AgentState) -> float | None:
    """Deterministically read the planned trip's total USD cost from state.

    Reads the latest verify_budget result inside itinerary_plan.step_results and
    prefers the group total (what the whole party pays) over the per-person total.
    Returns None when no trip has been planned (or no usable total is present).
    """
    plan_state = state.get("itinerary_plan") or {}
    results = plan_state.get("step_results") or {}
    budget_key = next((k for k in results if k.startswith("verify_budget")), None)
    if not budget_key:
        return None

    entry = results[budget_key]
    inner = entry.get("data", entry) if isinstance(entry, dict) else {}
    if not isinstance(inner, dict):
        return None

    total = inner.get("group_grand_total") or inner.get("grand_total")
    try:
        total = float(total)
    except (TypeError, ValueError):
        return None
    return total if total > 0 else None


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
        # Lazy rebuild in case the server imported this module before a new tool was added
        from tools.destinations import advisor_tools as _latest  # noqa: PLC0415
        refreshed = {t.name: t for t in _latest}
        _tool_map.update(refreshed)
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


class AdvisorExecutorNode:
    """Execute the first planned tool call and append the result to accumulated state."""

    def __call__(self, state: AgentState) -> dict:
        render_node("advisor_executor")
        plan = state.get("advisor_plan") or []
        past_results = list(state.get("advisor_last_tool_results") or [])

        if not plan:
            return {"advisor_last_tool_results": past_results}

        step = plan[0]
        tool_name = step["tool_name"]
        args = step.get("args", {})

        # convert_trip_cost needs the planned trip's USD total, which lives in state
        # rather than in the planner's args. Inject it deterministically here.
        if tool_name == "convert_trip_cost":
            args = {**args, "amount_usd": extract_trip_total_usd(state)}

        already_ran = any(
            r["tool_name"] == tool_name and r["args"] == args
            for r in past_results
        )
        if already_ran:
            render_node_status(f"[Executor] Skipping duplicate call: {tool_name}({args})")
            return {"advisor_last_tool_results": past_results}

        render_node_status(f"[Executor] Executing: {tool_name}({args})")
        result = _run_step(tool_name, args)
        result = scan_tool_output(result, source=tool_name, session_id=state.get("session_id", "unknown"))

        past_results.append({"tool_name": tool_name, "args": args, "result": result})

        update: dict = {"advisor_last_tool_results": past_results}
        # Persist concert search args so the planner can carry them forward on genre refinements
        if tool_name == "search_concerts":
            update["advisor_last_concert_search"] = {
                k: args[k] for k in ("city", "month", "genre") if k in args
            }
        # Carry the searched month forward as the trip_start so that when the user
        # transitions from "concerts/events in August" to actually planning the trip,
        # the planning flow already knows the month and doesn't re-ask for a date.
        if tool_name in ("search_concerts", "search_special_events"):
            iso_month = _month_to_iso(args.get("month"))
            if iso_month:
                current = state.get("trip_start")
                # Don't clobber a more specific same-month date already in state.
                if not current or current[:7] != iso_month:
                    update["trip_start"] = iso_month
        return update
