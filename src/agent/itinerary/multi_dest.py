"""
Multi-destination orchestration for long trips.
================================================
When a trip is longer than the destination's recommended maximum stay, we stop
building one over-long single-city itinerary and instead plan a *round route* of
several nearby cities (fly into the anchor city, drive between the rest, fly home
from the anchor). Each city then gets its own full day-by-day itinerary.

The whole thing is driven by ONE list — ``trip_segments`` — so there is a single
code path for both cases:

  • single destination → ``trip_segments`` has one element
  • multi destination  → ``trip_segments`` has N elements

Flow (all on the existing in-graph itinerary nodes, so HITL interrupts and the
single-destination output are unchanged):

  segment_planner → [multi_flight] → leg_dispatch → itinerary_planner → executor
      ⇄ replanner → critic → leg_collect → (loop to leg_dispatch | trip_formatter)

The legs run *sequentially* (not via parallel Send) on purpose: the critic's
budget HITL uses ``interrupt()``, which only works under the graph checkpointer.

Nodes
-----
SegmentPlannerNode  Decides single vs. multi and builds ``trip_segments``.
LegDispatchNode     Loads the current segment into the per-leg itinerary state.
LegCollectNode      Captures a finished leg and advances / finishes the loop.
TripFormatterNode   Renders the trip. Single leg → delegates to the existing
                    ItineraryFormatterNode (byte-for-byte identical to before);
                    multi leg → stitches the legs together with drive transitions.
"""
from __future__ import annotations

import uuid
from typing import List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt
from pydantic import BaseModel, ConfigDict, Field

from agent.core.state import AgentState
from agent.itinerary.formatter import (
    ItineraryFormatterNode,
    _build_ui_props,
    _fmt_flight_line,
    _unwrap_result,
)
from security import validate_city
from tools.dependencies import data_provider

# Hard ceiling on cities in a multi-destination route — keeps the sequential
# leg loop well within the graph recursion limit and the output readable.
MAX_SEGMENTS = 4
# How many alternative routes to propose to the user for selection.
MAX_ROUTE_OPTIONS = 3
# Default recommended max stay when a city has no duration data in the DB.
_DEFAULT_MAX_DAYS = 5


# ---------------------------------------------------------------------------
# LLM split contract
# ---------------------------------------------------------------------------

class _CitySegment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    city: str = Field(description="City name, exactly as given in the candidate list.")
    days: int = Field(description="Whole days to spend in this city (>= 1).")


class _RouteOption(BaseModel):
    model_config = ConfigDict(extra="forbid")
    segments: List[_CitySegment] = Field(
        description=(
            "Ordered round route. The FIRST city must be the anchor city (the "
            "entry/exit airport); the remaining cities are reached by car. Days "
            "across all cities must sum to the total trip length."
        )
    )


class _MultiCityRoutes(BaseModel):
    model_config = ConfigDict(extra="forbid")
    routes: List[_RouteOption] = Field(
        description=(
            "Two or three DISTINCT alternative round routes for the same trip. "
            "Each route is a full plan on its own; they should differ from one "
            "another (different city mix and/or day split), not be minor variations."
        )
    )


_SPLIT_SYSTEM = """You are a trip routing planner.

The traveller wants a {total} day trip but {anchor} is best enjoyed in about
{max_days} days. Instead of over-staying one city, design ROAD TRIPS {region}:
fly into {anchor}, drive between nearby cities, and fly home from {anchor}.

Propose {n_routes} DISTINCT alternative routes so the traveller can choose.

Rules for EVERY route:
- The FIRST segment MUST be {anchor} (it is the entry and exit airport).
- Pick well-known, real tourist cities {region} that travellers actually visit,
  within reasonable driving distance of each other (e.g. for Italy: Rome,
  Florence, Venice, Naples, Milan, Bologna).
- Use between 2 and {max_segments} cities total (including {anchor}).
- Give each city a sensible number of days for its size; the days across ALL
  cities in that route MUST sum to exactly {total}.
- Order the cities as an efficient driving loop that starts and ends at {anchor}.

Make the routes genuinely different from each other (vary the city mix and the
day allocation). Return the structured routes only."""


# ---------------------------------------------------------------------------
# SegmentPlannerNode
# ---------------------------------------------------------------------------

class SegmentPlannerNode:
    """Build ``trip_segments`` — one city for normal trips, a round route for long ones."""

    def __init__(self, llm: Optional[BaseChatModel] = None) -> None:
        self._llm = (
            llm.with_structured_output(_MultiCityRoutes, method="function_calling")
            if llm else None
        )

    def __call__(self, state: AgentState) -> dict:
        anchor = (state.get("destination_city") or "").strip()
        total  = int(state.get("trip_days") or 3)

        # Always-valid single-destination baseline. Returned unchanged whenever
        # the trip fits, the mode isn't standalone, or the split can't be built.
        base = {
            "trip_segments": [
                {"destination": anchor, "days": total, "order": 0, "drive_from_prev": None}
            ],
            "seg_index":            0,
            "total_trip_days":      total,
            "is_multi_destination": False,
            "itineraries":          [],
            "trip_total_budget":    float(state.get("total_budget") or 0),
        }

        # Multi-destination applies to standalone itineraries AND to booked
        # (with_travel_data) trips that far exceed the anchor's recommended stay.
        # For a booked trip we keep the round-trip flights into the anchor and
        # re-plan a hotel per city (the single booked hotel only ever covered the
        # one city, so it can't carry a multi-city route).
        if state.get("itinerary_mode") not in ("standalone", "with_travel_data") \
                or not anchor or not self._llm:
            return base

        max_days = self._recommended_max(anchor)
        if total <= max_days:
            return base  # trip fits the anchor — stay single

        routes = self._split(anchor, total, max_days)
        if not routes:
            return base  # couldn't build any real route — fall back to single

        # Propose the candidate routes to the user; route_select pauses for their
        # pick. trip_segments is seeded with the first route so state stays valid
        # if selection is somehow skipped, but the real choice is made downstream.
        return {
            "proposed_routes":      routes,
            "trip_segments":        routes[0],
            "seg_index":            0,
            "total_trip_days":      total,
            "is_multi_destination": True,
            "itineraries":          [],
            "trip_total_budget":    float(state.get("total_budget") or 0),
            "progress_log": [
                "🗺️ **MULTI-DESTINATION:** "
                f"{total} days exceeds {anchor}'s recommended {max_days} — "
                f"proposing {len(routes)} route option(s) for the user to choose"
            ],
        }

    # ── helpers ────────────────────────────────────────────────────────────

    def _recommended_max(self, city: str) -> int:
        try:
            rec = data_provider.get_recommended_duration(city)
        except Exception:
            return _DEFAULT_MAX_DAYS
        if isinstance(rec, dict) and rec.get("max_days"):
            return int(rec["max_days"])
        return _DEFAULT_MAX_DAYS

    def _split(self, anchor: str, total: int, max_days: int) -> list[list[dict]]:
        """Ask the LLM for several distinct round routes of real cities.

        Returns a list of routes (each a list of segment dicts), cleaned and
        de-duplicated. Activities, flights and hotels are all sourced from live
        APIs at runtime (the DB city/flight tables are sparse), so candidate
        cities are NOT constrained to a DB list — any real, well-known city in
        the country works.
        """
        try:
            country = data_provider.get_city_country(anchor)
        except Exception:
            country = None
        region = f"in {country}" if country else "near it"

        try:
            result = self._llm.invoke([
                SystemMessage(content=_SPLIT_SYSTEM.format(
                    total=total, anchor=anchor, max_days=max_days,
                    max_segments=MAX_SEGMENTS, region=region,
                    n_routes=MAX_ROUTE_OPTIONS,
                )),
                HumanMessage(content=f"Plan the {total}-day round trip from {anchor}."),
            ])
        except Exception:
            return []

        routes: list[list[dict]] = []
        seen_routes: set[tuple] = set()
        for option in result.routes:
            segments = self._clean_route(option.segments, anchor, total, max_days)
            if not segments:
                continue
            # De-dup identical routes (same ordered cities + day split).
            sig = tuple((s["destination"].lower(), s["days"]) for s in segments)
            if sig in seen_routes:
                continue
            seen_routes.add(sig)
            routes.append(segments)
            if len(routes) >= MAX_ROUTE_OPTIONS:
                break
        return routes

    def _clean_route(
        self,
        raw_segments: list,
        anchor: str,
        total: int,
        max_days: int,
    ) -> list[dict]:
        """Validate/normalize one LLM route into segment dicts, or [] if invalid."""
        # Validate city-name FORMAT only (defends against prompt-injection / junk),
        # force the anchor first, de-dupe, and cap the count.
        cleaned: list[tuple[str, int]] = []
        seen: set[str] = set()
        for seg in raw_segments:
            try:
                city = validate_city((seg.city or "").strip())
            except Exception:
                continue
            key = city.lower()
            if key not in seen and seg.days and seg.days > 0:
                cleaned.append((city, int(seg.days)))
                seen.add(key)
        if anchor.lower() not in seen:
            cleaned.insert(0, (anchor, max_days))
        else:
            cleaned.sort(key=lambda c: c[0].lower() != anchor.lower())  # anchor first
        cleaned = cleaned[:MAX_SEGMENTS]
        if len(cleaned) < 2:
            return []

        days = _normalize_days([d for _, d in cleaned], total)
        prev = None
        segments: list[dict] = []
        for order, ((city, _), d) in enumerate(zip(cleaned, days)):
            segments.append({
                "destination": city, "days": d, "order": order,
                "drive_from_prev": prev,
            })
            prev = city
        return segments


def _normalize_days(days: list[int], total: int) -> list[int]:
    """Scale/clamp a list of day counts so they sum to exactly ``total`` (each >= 1)."""
    n = len(days)
    days = [max(1, d) for d in days]
    s = sum(days)
    # Distribute the difference one day at a time, largest segments first.
    diff = total - s
    order = sorted(range(n), key=lambda i: days[i], reverse=True)
    i = 0
    while diff != 0:
        idx = order[i % n]
        if diff > 0:
            days[idx] += 1
            diff -= 1
        elif days[idx] > 1:
            days[idx] -= 1
            diff += 1
        else:
            # everything is at the floor of 1 and we still owe removals — stop
            if all(d <= 1 for d in days):
                break
        i += 1
    return days


# ---------------------------------------------------------------------------
# RouteSelectNode — HITL: let the user pick one of the proposed routes
# ---------------------------------------------------------------------------

def _parse_route_selection(value: str, num_routes: int) -> int:
    """Parse the resume value from the route_selection interrupt.

    Accepts:
      "auto" / ""   → first (recommended) route
      "route:N"     → explicit route index
    Clamps to a valid index; falls back to 0 on any parse error.
    """
    if not value or value == "auto":
        return 0
    try:
        idx = int(value.split(":")[1]) if ":" in value else int(value)
    except (IndexError, ValueError):
        return 0
    if num_routes <= 0:
        return 0
    return max(0, min(idx, num_routes - 1))


class RouteSelectNode:
    """Pause and ask the user which of the proposed round routes to build.

    Emits a ``{type: "route_selection", routes: [...]}`` interrupt that the web
    UI renders as a route picker. On resume it finalizes ``trip_segments`` to the
    chosen route. The LLM route generation already ran in ``segment_planner`` and
    is persisted in ``proposed_routes``, so re-running this node on resume is
    cheap and deterministic (it just reads state and applies the choice).
    """

    def __call__(self, state: AgentState) -> dict:
        routes = state.get("proposed_routes") or []
        if not routes:
            return {}  # nothing proposed — segment_planner already seeded trip_segments

        anchor = routes[0][0]["destination"]
        total  = int(state.get("total_trip_days")
                     or sum(int(s["days"]) for s in routes[0]))

        options = [
            {
                "label":      f"Option {i + 1}",
                "total_days": sum(int(s["days"]) for s in segs),
                "segments":   [
                    {
                        "destination":     s["destination"],
                        "days":            int(s["days"]),
                        "drive_from_prev": s.get("drive_from_prev"),
                    }
                    for s in segs
                ],
            }
            for i, segs in enumerate(routes)
        ]

        choice: str = interrupt({
            "type": "route_selection",
            "question": (
                f"Your {total}-day trip from **{anchor}** is longer than one city "
                "is best enjoyed for, so I've planned a few road-trip routes. "
                "Pick the one you'd like me to build a full itinerary for:"
            ),
            "anchor":     anchor,
            "total_days": total,
            "routes":     options,
        })

        idx    = _parse_route_selection(choice, len(routes))
        chosen = routes[idx]
        return {
            "trip_segments":        chosen,
            "seg_index":            0,
            "total_trip_days":      sum(int(s["days"]) for s in chosen),
            "is_multi_destination": True,
            "itineraries":          [],
            "progress_log": [
                "✅ **ROUTE SELECTED:** "
                + " → ".join(f"{s['destination']} ({s['days']}d)" for s in chosen)
            ],
        }


# ---------------------------------------------------------------------------
# LegDispatchNode — load the current segment into the per-leg itinerary state
# ---------------------------------------------------------------------------

class LegDispatchNode:
    """Prime the shared itinerary state for the segment at ``seg_index``.

    For a single-destination trip this just re-asserts the existing destination
    and days and clears the (empty) plan, so the downstream planner sees exactly
    what it would have seen before this feature existed.
    """

    def __call__(self, state: AgentState) -> dict:
        segments = state.get("trip_segments") or []
        idx      = int(state.get("seg_index") or 0)
        is_multi = bool(state.get("is_multi_destination"))
        if not segments:
            return {}
        seg = segments[min(idx, len(segments) - 1)]

        update: dict = {
            "destination_city":   seg["destination"],
            "trip_days":          int(seg["days"]),
            # Fresh per-leg plan + loop counters
            "itinerary_plan":     {},
            "current_step_index": 0,
            "itinerary_feasible": True,
            "critic_attempts":    0,
            "critic_action":      "pass",
            "replanner_action":   "continue",
        }
        if is_multi:
            # Multi legs: standalone, hotel-only pricing, no per-leg budget HITL.
            update["itinerary_mode"]             = "standalone"
            update["include_flight"]             = False
            update["total_budget"]               = 0.0
            update["use_min_prices_for_budget"]  = False
            update["switch_travel_triggered"]    = False
        else:
            # Single destination: leave mode/budget exactly as plan_check set them.
            update["include_flight"] = True
        return update


# ---------------------------------------------------------------------------
# LegCollectNode — capture a finished leg, then loop or finish
# ---------------------------------------------------------------------------

class LegCollectNode:
    """Append the just-built leg to ``itineraries`` and advance the loop counter."""

    def __call__(self, state: AgentState) -> dict:
        segments = state.get("trip_segments") or []
        idx      = int(state.get("seg_index") or 0)

        leg = {
            "order":          idx,
            "destination":    state.get("destination_city", ""),
            "days":           int(state.get("trip_days") or 0),
            "drive_from_prev": (segments[idx].get("drive_from_prev")
                                if idx < len(segments) else None),
            "itinerary_mode": state.get("itinerary_mode", "standalone"),
            "itinerary_plan": state.get("itinerary_plan", {}),
            "critic_action":  state.get("critic_action", "pass"),
            "feasible":       state.get("itinerary_feasible", True),
        }

        # Sequential loop → a plain list write accumulates safely. The first leg
        # (idx 0, or an update with no segments) resets any list from a prior turn.
        existing = [] if idx == 0 else list(state.get("itineraries") or [])
        return {
            "itineraries": existing + [leg],
            "seg_index":   idx + 1,
        }


# ---------------------------------------------------------------------------
# TripFormatterNode — single passthrough or multi stitch
# ---------------------------------------------------------------------------

class TripFormatterNode:
    """Render the trip. Single leg is delegated to the unchanged itinerary formatter."""

    def __init__(self, llm: Optional[BaseChatModel] = None) -> None:
        self._formatter = ItineraryFormatterNode(llm)

    def __call__(self, state: AgentState) -> dict:
        legs = [l for l in (state.get("itineraries") or [])]
        if len(legs) <= 1:
            # Single destination (or an itinerary update) — identical to before.
            return self._formatter(state)
        return self._render_multi(state, sorted(legs, key=lambda l: l["order"]))

    # ── multi-city render ────────────────────────────────────────────────────

    def _render_multi(self, state: AgentState, legs: list[dict]) -> dict:
        """Render a multi-city trip as TWO messages:

          1. a visible text OVERVIEW (flight + route outline + combined budget), and
          2. a viewer-only message carrying one ItineraryViewer PER city.

        Two messages are required because the frontend hides a message's text
        whenever a viewer is attached to it — so the connecting narrative (flight,
        drives, budget) must live on a message with no viewer of its own.
        """
        origin = state.get("current_city", "")
        anchor = legs[0]["destination"]
        total  = int(state.get("total_trip_days") or sum(l["days"] for l in legs))
        budget = float(state.get("trip_total_budget") or 0)

        # ── 1. Overview (visible text — no viewer attached) ──────────────────
        parts: list[str] = [
            f"# 🌍 Your {total}-Day Multi-City Trip",
            f"Fly into **{anchor}** from {origin}, drive the loop below, then fly "
            f"home from **{anchor}**. Each city's day-by-day plan is shown beneath this.",
        ]
        flight_total, flight_md = self._flight_section(state, origin, anchor)
        if flight_md:
            parts.append(flight_md)

        parts.append("## 🗺️ Your route")
        legs_total = 0.0
        for i, leg in enumerate(legs):
            icon = "✈️" if i == 0 else "🚗"
            hop  = "" if i == 0 else f" *(drive from {leg['drive_from_prev']})*"
            parts.append(
                f"{i + 1}. {icon} **{leg['destination']}** — "
                f"{leg['days']} day{'s' if leg['days'] != 1 else ''}{hop}"
            )
            legs_total += _leg_cost(leg.get("itinerary_plan") or {})
        parts.append(f"{len(legs) + 1}. 🔙 Drive back to **{anchor}** for the flight home.")

        grand = flight_total + legs_total
        parts.append("\n---")
        parts.append("## 💰 Trip Budget *(estimated — no bookings confirmed)*")
        parts.append(f"- ✈️ Round-trip flight: **~${flight_total:.0f}**")
        parts.append(f"- 🏨 Cities (hotels + activities): **~${legs_total:.0f}**")
        parts.append(f"\n**Estimated grand total: ~${grand:.0f}**")
        if budget:
            remaining = budget - grand
            emoji = "✅" if remaining >= 0 else "⚠️"
            parts.append(f"\n{emoji} Budget on file: ${budget:.0f} — "
                         f"{'remaining' if remaining >= 0 else 'over by'} ${abs(remaining):.0f}")

        overview_msg = AIMessage(content="\n\n".join(parts), name="trip_overview")

        # ── 2. One ItineraryViewer per city ──────────────────────────────────
        # The viewers share one explicit message id (a fresh AIMessage has id=None
        # until add_messages fills it in, which would break the frontend match and
        # show only the last city). This message carries no text of its own.
        plans_id = str(uuid.uuid4())
        ui_messages: list[dict] = []
        for leg in legs:
            try:
                leg_state = {
                    **state,
                    "destination_city": leg["destination"],
                    "trip_days":        leg["days"],
                    "itinerary_plan":   leg.get("itinerary_plan") or {},
                    "itinerary_mode":   "standalone",
                }
                ui_messages.append({
                    "type":     "ui",
                    "name":     "ItineraryViewer",
                    "props":    _build_ui_props(leg_state),
                    "metadata": {"message_id": plans_id},
                })
            except Exception:
                continue

        plans_msg = AIMessage(content="", name="itinerary_formatter", id=plans_id)

        return {"messages": [overview_msg, plans_msg], "ui": ui_messages}

    def _flight_section(self, state: AgentState, origin: str, anchor: str) -> tuple[float, str]:
        # Prefer the user's booked flights (with_travel_data path) so the overview
        # reflects what they actually chose; fall back to the cheapest searched
        # option for standalone multi trips that never had a selection.
        outbound = state.get("itinerary_selected_outbound_flight") or _cheapest(state.get("flight_options"))
        ret      = state.get("itinerary_selected_return_flight")   or _cheapest(state.get("return_flight_options"))
        if not outbound and not ret:
            return 0.0, ""
        lines = ["## 🛫 Flights"]
        total = 0.0
        if outbound:
            lines.append(_fmt_flight_line("✈️ **Outbound**", outbound, origin, anchor))
            total += float(outbound.get("price", 0) or 0)
        if ret:
            lines.append(_fmt_flight_line("🔙 **Return**", ret, anchor, origin))
            total += float(ret.get("price", 0) or ret.get("total_price", 0) or 0)
        return total, "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Routing edges
# ---------------------------------------------------------------------------

def after_segment_planner(state: AgentState) -> str:
    """Multi trips pause for the user to pick a route; single trips skip to the loop."""
    return "route_select" if state.get("is_multi_destination") else "leg_dispatch"


def after_route_select(state: AgentState) -> str:
    """After the user picks a route, search the entry-city flight — unless the trip
    already has booked flights (with_travel_data path), in which case we keep them
    and go straight to the leg loop."""
    return "leg_dispatch" if state.get("has_flights") else "multi_flight"


def after_leg_collect(state: AgentState) -> str:
    """Loop back for the next city, or render once every segment is built."""
    segments = state.get("trip_segments") or []
    if int(state.get("seg_index") or 0) < len(segments):
        return "leg_dispatch"
    return "trip_formatter"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _cheapest(flights) -> Optional[dict]:
    usable = [f for f in (flights or [])
              if isinstance(f, dict) and not f.get("message")
              and (f.get("price") or f.get("total_price"))]
    if not usable:
        return None
    return min(usable, key=lambda f: float(f.get("price") or f.get("total_price") or 9e9))


def _leg_cost(plan: dict) -> float:
    """Hotel + activity cost for one leg, read from its verify_budget result."""
    results = (plan or {}).get("step_results", {})
    key = next((k for k in results if k.startswith("verify_budget")), None)
    if not key:
        return 0.0
    data = _unwrap_result(results[key])
    return float(data.get("group_grand_total") or data.get("grand_total", 0) or 0)


def _demote_heading(md: str) -> str:
    """Drop a leg's own top-level '# ...' title so it nests under the city section."""
    lines = md.splitlines()
    out = [ln for ln in lines if not ln.lstrip().startswith("# ")]
    return "\n".join(out).strip() or md
