"""
PlanCheckNode — Human-in-the-Loop checkpoint before building an itinerary.

Checks whether the state already contains a travel plan (hotels + flights).
  has data  → resolves hotel/flight references; sets itinerary_mode="with_travel_data"
  no data   → uses LangGraph interrupt() to ask the user:
                yes  → redirect_to_travel  (user will plan flights/hotels first)
                no   → standalone          (build itinerary without bookings)

State fields written:
    itinerary_mode                     "with_travel_data" | "standalone" | "redirect_to_travel"
    itinerary_selected_hotel           raw hotel dict (lat/lng/price) for Executor
    itinerary_selected_outbound_flight raw outbound flight dict for Executor
    itinerary_selected_return_flight   raw return flight dict for Executor
"""
from __future__ import annotations

from typing import Optional

from langgraph.types import interrupt

from agent.core.state import AgentState
from tools.dependencies import data_provider


class PlanCheckNode:
    """HITL checkpoint: ensure travel data exists before building an itinerary."""

    def __call__(self, state: AgentState) -> dict:
        destination = state.get("destination_city") or ""
        travel_plan = state.get("travel_plan") or {}

        has_hotels  = bool(travel_plan.get("hotels"))
        has_flights = (
            bool(state.get("has_flights"))
            and bool(state.get("flight_options"))
            and bool(state.get("return_flight_options"))
        )

        dest_label = destination or "your destination"

        if has_hotels and has_flights:
            # ── Human-in-the-Loop: let user pick their preferred flight + hotel ─
            user_choice: str = interrupt({
                "type": "travel_selection",
                "question": (
                    f"Choose your preferred flight and hotel for the itinerary "
                    f"to **{dest_label}**"
                ),
                "flight_pairings": travel_plan.get("flight_pairings", []),
                "hotels": travel_plan.get("hotels", []),
            })

            pairings_list = travel_plan.get("flight_pairings", [])
            hotels_list   = travel_plan.get("hotels", [])
            if user_choice.strip() == "auto":
                flight_idx = _cheapest_pairing_idx(pairings_list)
                hotel_idx  = _cheapest_hotel_idx(hotels_list)
            else:
                flight_idx, hotel_idx = _parse_selection(
                    user_choice.strip(),
                    len(pairings_list),
                    len(hotels_list),
                )

            resolved = self._resolve_travel_data(
                state, travel_plan, destination, flight_idx, hotel_idx
            )
            resolved["itinerary_mode"] = "with_travel_data"
            return resolved

        # ── Human-in-the-Loop: present a choice widget to the user ───────
        user_choice: str = interrupt({
            "question": (
                f"I don't have a travel plan with flights and hotels for your trip to "
                f"**{dest_label}** yet.\n\n"
                "Would you like me to search for flights and hotels first, "
                "or build a standalone day-by-day itinerary without specific bookings?"
            ),
            "options": [
                ("yes", "Search for flights & hotels first"),
                ("no",  "Build a generic itinerary"),
            ],
        })

        if user_choice.strip().lower() in ("yes", "y"):
            # Switch intent to travel planning so after_enrichment routes to
            # flight_search instead of looping back to plan_check.
            # extract_metadata re-reads the conversation history (which already
            # contains dates / budget the user provided) so trip_start is set.
            return {
                "intent":        "new_travel_plan",
                "itinerary_mode": "redirect_to_travel",
            }

        return {"itinerary_mode": "standalone"}

    # ------------------------------------------------------------------
    # Resolve raw hotel / flight records that the Executor needs
    # ------------------------------------------------------------------

    def _resolve_travel_data(
        self,
        state: AgentState,
        travel_plan: dict,
        destination: str,
        flight_idx: int = 0,
        hotel_idx: int = 0,
    ) -> dict:
        result: dict = {}

        # ── Hotel ──────────────────────────────────────────────────────
        hotels_curated      = travel_plan.get("hotels", [])
        hotel_idx           = min(hotel_idx, len(hotels_curated) - 1) if hotels_curated else 0
        selected_hotel_name = hotels_curated[hotel_idx].get("name") if hotels_curated else None
        if selected_hotel_name and destination:
            try:
                raw_hotels = [
                    h for h in data_provider.fetch_hotels(destination)
                    if isinstance(h, dict) and not h.get("message")
                ]
                matched = next(
                    (h for h in raw_hotels
                     if h.get("name", "").lower() == selected_hotel_name.lower()),
                    raw_hotels[0] if raw_hotels else None,
                )
                if matched:
                    result["itinerary_selected_hotel"] = matched
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning(
                    "Hotel resolution failed for %r in %r: %s", selected_hotel_name, destination, exc
                )
        # Fallback: build a minimal hotel dict from the curated plan so Executor
        # always has a non-empty itinerary_selected_hotel (avoids defaulting to Paris coords).
        if "itinerary_selected_hotel" not in result and hotels_curated:
            result["itinerary_selected_hotel"] = hotels_curated[hotel_idx]

        # ── Outbound flight ────────────────────────────────────────────
        valid_outbound = [
            f for f in (state.get("flight_options") or [])
            if isinstance(f, dict) and not f.get("message")
        ]
        if valid_outbound:
            pairings       = travel_plan.get("flight_pairings", [])
            outbound_label = ""
            if pairings:
                pairing_idx    = min(flight_idx, len(pairings) - 1)
                ob             = pairings[pairing_idx].get("outbound") or {}
                outbound_label = ob.get("flight_number") or ob.get("label") or ""

            matched_ob = _match_flight(valid_outbound, outbound_label)
            result["itinerary_selected_outbound_flight"] = (
                matched_ob or _pick_best_outbound(valid_outbound)
            )

        # ── Return flight ──────────────────────────────────────────────
        valid_return = [
            f for f in (state.get("return_flight_options") or [])
            if isinstance(f, dict) and not f.get("message")
        ]
        if valid_return:
            valid_return.sort(key=lambda f: float(f.get("price", 9999)))
            result["itinerary_selected_return_flight"] = valid_return[0]

        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _match_flight(flights: list[dict], label: str) -> Optional[dict]:
    if not label:
        return None
    for f in flights:
        fn = str(f.get("flight_number", "")).strip()
        if fn and fn in label:
            return f
    return None


def _arrival_hour(flight: dict) -> int:
    arr = str(flight.get("arrival_time", "23:59"))
    if "T" in arr:
        arr = arr.split("T")[1]
    try:
        return int(arr[:5].split(":")[0])
    except (ValueError, IndexError):
        return 23


def _cheapest_pairing_idx(pairings: list[dict]) -> int:
    if not pairings:
        return 0
    return min(range(len(pairings)), key=lambda i: float(pairings[i].get("total_price", 9999)))


def _cheapest_hotel_idx(hotels: list[dict]) -> int:
    if not hotels:
        return 0
    return min(range(len(hotels)), key=lambda i: float(hotels[i].get("price_per_night", 9999)))


def _parse_selection(
    value: str,
    num_pairings: int,
    num_hotels: int,
) -> tuple[int, int]:
    """Parse user resume value from the travel_selection interrupt.

    Accepts:
      "auto"          → cheapest of each (index 0 after caller sorts by price)
      "flight:N,hotel:M" → explicit indices
    Falls back to (0, 0) on any parse error.
    """
    if value == "auto" or not value:
        return 0, 0
    try:
        parts = value.split(",")
        flight_idx = int(parts[0].split(":")[1])
        hotel_idx  = int(parts[1].split(":")[1])
        # Clamp to valid range
        flight_idx = max(0, min(flight_idx, num_pairings - 1)) if num_pairings else 0
        hotel_idx  = max(0, min(hotel_idx,  num_hotels  - 1)) if num_hotels  else 0
        return flight_idx, hotel_idx
    except (IndexError, ValueError):
        return 0, 0


def _pick_best_outbound(flights: list[dict]) -> dict:
    """Prefer flights arriving before 16:00 (max Day-1 sightseeing); fallback cheapest."""
    if not flights:
        return {}
    early = [f for f in flights if _arrival_hour(f) < 16]
    pool  = early if early else flights
    return min(pool, key=lambda f: float(f.get("price", 9999)))
