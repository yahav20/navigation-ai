# src/agent/nodes/itinerary/formatter.py
from __future__ import annotations
import logging
from langchain_core.messages import AIMessage
from agent.state import AgentState

logger = logging.getLogger(__name__)

class ItineraryFormatterNode:
    """
    Dedicated formatting node for the travel itinerary (Itinerary Formatter).
    Its role is to take the raw State data and turn it into formatted,
    interactive, and inviting text for the user, while handling all edge cases (success/alternatives/errors).
    """

    def __call__(self, state: AgentState) -> dict:
        plan_state = state.get("itinerary_plan", {})
        action = state.get("itinerary_fallback_action", "")
        feasible = state.get("itinerary_feasible", True)
        
        print("\n--- 📊 FORMATTING PHASE: Preparing User Response ---")

        # Case 1: The itinerary failed and the system proposed alternative destinations (Fallback Alternatives)
        if action == "suggested_alternatives" or not feasible:
            return self._format_fallback_response(state)

        # Case 2: The itinerary was completed successfully and we have a ready markdown from the Observer
        final_markdown = plan_state.get("final_markdown")
        if final_markdown:
            print("✨ Successfully formatted final itinerary markdown.")
            return {"messages": [AIMessage(content=final_markdown)]}

        # Case 3: Edge case/fallback - an existing plan that wasn't finalized
        return self._format_intermediate_response(state)

    def _format_fallback_response(self, state: AgentState) -> dict:
        """Formats a nice response for the user when the original destination is not feasible and suggests culturally close alternatives"""
        print("🗺️ Formatting alternative destinations view.")
        original_dest = state.get("destination_city", "the requested destination")
        origin = state.get("current_city", "")
        alternatives = state.get("alternative_destinations", [])
        reason = state.get("itinerary_fallback_reason", "budget constraints or flight availability")

        lines = [
            f"### 🗺️ Oops, we had to recalculate the route...",
            f"We couldn't build a complete travel itinerary for **{original_dest}** that meets all your conditions and budget (Reason: {reason}).",
            f"\nBut don't worry! We found **3 amazing alternative destinations** for you with a similar culture, atmosphere, and vibe, with direct flights from {origin}:\n"
        ]

        for i, city in enumerate(alternatives, 1):
            lines.append(f"**{i}. 📍 {city}**")
            lines.append(f"Flights available from {origin}, travel experience and cultural atmosphere very close to {original_dest}.\n")

        lines.append("---")
        lines.append("**Which destination sounds best to you?** Just tell me its name, and I'll build a perfect itinerary for it! 😊")
        
        formatted_text = "\n".join(lines)
        return {"messages": [AIMessage(content=formatted_text)]}

    def _format_intermediate_response(self, state: AgentState) -> dict:
        """If the graph stops in the middle or requires manual intervention, displays a formatted technical status"""
        plan_state = state.get("itinerary_plan", {})
        results = plan_state.get("step_results", {})
        current_index = state.get("current_step_index", 0)
        
        lines = [
            "### 🛠️ Itinerary Building Status",
            "The system is in the middle of a step-by-step planning process. Here are the steps completed so far:\n"
        ]
        
        for key, val in results.items():
            status = "❌ Error" if isinstance(val, dict) and "error" in val else "✅ Completed"
            if isinstance(val, dict) and val.get("reason") == "cached":
                status = "⏩ Skipped (Fetched from cache)"
            lines.append(f"- **{key}**: {status}")
            
        lines.append(f"\nNext step in line: number {current_index + 1}")
        
        formatted_text = "\n".join(lines)
        return {"messages": [AIMessage(content=formatted_text)]}