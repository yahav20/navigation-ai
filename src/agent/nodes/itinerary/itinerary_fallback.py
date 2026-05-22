"""
ItineraryFallbackNode — Plan & Execute, Fallback Handler

Called when feasibility check fails. Decides the recovery strategy:

  1. "no_flights"      → find culturally-similar nearby destination
  2. "budget_exceeded" → try ±3 days adjustment; if still over, bump budget by up to $500;
                         if still over, find cheaper nearby destination
  3. "no_hotels"       → widen hotel search (relax some preference filters),
                         or fall back to nearby destination

The node WRITES its decision back to state so the router can branch correctly.
It does NOT re-run the full planner — that is done by a subsequent call to
ItineraryPlannerNode with updated state values.
"""

from __future__ import annotations

import logging
import math
import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from agent.state import AgentState

# יבוא נקי של ספק הנתונים המרכזי של המערכת, ללא התממשקות ישירה ל-SQL!
from tools.dependencies import data_provider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class ItineraryFallbackNode:
    """
    Reads `itinerary_fallback_reason` from state and attempts recovery.

    Writes to state:
        destination_city          — may change to alternative city
        trip_days                 — may be adjusted ±3
        total_budget              — may be bumped by up to $500
        itinerary_fallback_action — what was actually done (for formatter)
        itinerary_fallback_alternatives — list of up to 3 alternative cities offered
    """

    CULTURAL_SIMILARITY_PROMPT = """You are a travel geography expert.
Given a destination city, suggest up to 3 alternative cities that are:
1. In the SAME or culturally adjacent region (e.g., Spain → France/Italy, not Morocco).
2. Reachable from the same origin.
3. Similar in travel vibe (beach, culture, city-break, etc.).

Return ONLY a JSON array of city name strings, e.g.: ["Lisbon", "Rome", "Athens"]
No explanation, no markdown.
"""

    def __init__(self, response_model: BaseChatModel, extraction_model: BaseChatModel = None) -> None:
        """Store the models used for recovery routing and alternative generation."""
        self.response_model = response_model
        self.extraction_model = extraction_model or response_model
        # מחקנו את ה-self.db לגמרי!

    # ------------------------------------------------------------------

    def __call__(self, state: AgentState) -> dict:
        reason = state.get("itinerary_fallback_reason", "unknown")
        destination = state.get("destination_city", "")
        origin = state.get("current_city", "")
        budget = state.get("total_budget", 0)
        trip_days = state.get("trip_days", 3)
        data_bundle = state.get("itinerary_data_bundle", {})

        logger.info("ItineraryFallbackNode: reason=%s destination=%s", reason, destination)

        if reason == "budget_exceeded":
            return self._handle_budget(state, data_bundle, budget, trip_days, destination, origin)

        if reason == "no_flights":
            return self._handle_no_flights(state, destination, origin)

        if reason == "no_hotels":
            return self._handle_no_hotels(state, destination, origin, budget, trip_days)

        # Unknown reason — offer alternatives by default
        return self._suggest_alternatives(destination, origin, budget, trip_days)

    # ------------------------------------------------------------------
    # Strategy 1: budget exceeded
    # ------------------------------------------------------------------

    def _handle_budget(
        self,
        state: AgentState,
        bundle: dict,
        budget: float,
        trip_days: int,
        destination: str,
        origin: str,
    ) -> dict:
        min_flight = bundle.get("flights", [{}])[0].get("price", 0) if bundle.get("flights") else 0
        min_hotel_night = bundle.get("hotels", [{}])[0].get("price_per_night", 0) if bundle.get("hotels") else 0

        # Try ±3 days
        for delta in [-1, -2, -3, 1, 2, 3]:
            new_days = max(1, trip_days + delta)
            cost = min_flight + min_hotel_night * new_days
            if budget and cost <= budget * 1.05:
                logger.info("Budget fix: adjusting days %d→%d", trip_days, new_days)
                return {
                    "trip_days": new_days,
                    "itinerary_fallback_action": f"adjusted_days_{new_days}",
                    "itinerary_feasible": True,   # retry planner
                    "itinerary_fallback_alternatives": [],
                }

        # Try +$500 budget bump
        bumped_budget = budget + 500
        cost_original = min_flight + min_hotel_night * trip_days
        if cost_original <= bumped_budget * 1.05:
            logger.info("Budget fix: bumping budget %.0f→%.0f", budget, bumped_budget)
            return {
                "total_budget": bumped_budget,
                "itinerary_fallback_action": "budget_bumped_500",
                "itinerary_feasible": True,
                "itinerary_fallback_alternatives": [],
            }

        # Nothing worked → suggest culturally similar cheaper destinations
        logger.info("Budget fix: suggesting alternative destinations")
        return self._suggest_alternatives(destination, origin, bumped_budget, trip_days)

    # ------------------------------------------------------------------
    # Strategy 2: no flights found
    # ------------------------------------------------------------------

    def _handle_no_flights(self, state: AgentState, destination: str, origin: str) -> dict:
        logger.info("No flights found — suggesting alternative destinations")
        return self._suggest_alternatives(
            destination, origin,
            state.get("total_budget", 0),
            state.get("trip_days", 3),
        )

    # ------------------------------------------------------------------
    # Strategy 3: no matching hotels
    # ------------------------------------------------------------------

    def _handle_no_hotels(
        self,
        state: AgentState,
        destination: str,
        origin: str,
        budget: float,
        trip_days: int,
    ) -> dict:
        # Try relaxing kosher-only constraint (but keep accessibility)
        relaxed_prefs = dict(state.get("user_preferences", {}))
        if relaxed_prefs.get("kosher"):
            relaxed_prefs.pop("kosher")
            
            # Use the global data_provider instead of direct DB instance
            raw_hotels = data_provider.fetch_hotels(destination) or []
            
            if raw_hotels:
                logger.info("No-hotel fix: relaxing kosher filter")
                return {
                    "user_preferences": relaxed_prefs,
                    "itinerary_fallback_action": "relaxed_kosher_for_hotels",
                    "itinerary_feasible": True,
                    "itinerary_fallback_alternatives": [],
                }

        # Otherwise suggest alternatives
        return self._suggest_alternatives(destination, origin, budget, trip_days)

    # ------------------------------------------------------------------
    # Core: culturally-similar alternative destinations (up to 3)
    # ------------------------------------------------------------------

    def _suggest_alternatives(
        self,
        destination: str,
        origin: str,
        budget: float,
        trip_days: int,
    ) -> dict:
        # Ask LLM for culturally similar cities (relying on its own geographical knowledge)
        prompt = f"""Destination: {destination}
Origin: {origin}
Trip days: {trip_days}
Budget: ${budget or 'flexible'}

Suggest up to 3 alternative cities."""

        messages = [
            SystemMessage(content=self.CULTURAL_SIMILARITY_PROMPT),
            HumanMessage(content=prompt),
        ]
        response = self.response_model.invoke(messages)
        raw = response.content.strip()

        # Strip markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip().rstrip("```").strip()

        try:
            alternatives = json.loads(raw)
            if not isinstance(alternatives, list):
                alternatives = []
        except Exception:
            alternatives = []

        # Validate: each alternative must exist in DB and have flights from origin
        validated = []
        for city in alternatives[:3]:
            # Clean validation using the global data_provider
            flights = data_provider.fetch_flights(origin, city) or []
            available = [f for f in flights if str(f.get("availability", "")).lower() == "available"]
            if available:
                validated.append(city)

        logger.info("Alternative destinations: %s", validated)

        return {
            "itinerary_fallback_action": "suggested_alternatives",
            "itinerary_fallback_alternatives": validated or alternatives[:3],
            "itinerary_feasible": False,  # formatter will show alternatives
            "alternative_destinations": validated or alternatives[:3],
        }