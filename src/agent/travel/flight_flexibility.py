"""HITL gate offering budget/city flexibility when no flights or alternatives exist."""
from langchain_core.messages import HumanMessage
from langgraph.types import interrupt

from agent.core.state import AgentState

MAX_FLEXIBILITY_ATTEMPTS = 2


class FlightFlexibilityGateNode:
    """Ask the user to relax their budget or change origin/destination as a last resort.

    Reached only when AlternativeDestinationNode found nothing — i.e. there is no
    route at all between the requested cities AND no reachable alternative fits
    the budget. Reuses AdjustmentsNode to apply whatever the user types (it already
    parses free-text budget/city changes), so this node only needs to ask.
    """

    def __call__(self, state: AgentState) -> dict:
        attempts = state.get("flexibility_attempts", 0)
        if attempts >= MAX_FLEXIBILITY_ATTEMPTS:
            return {"flexibility_action": "give_up"}

        origin = state.get("current_city") or "your origin"
        destination = state.get("destination_city") or "your destination"
        budget = state.get("total_budget")
        no_route = state.get("alternative_destinations_no_route", True)

        if no_route:
            # No reachable alternative cities from origin exist at all — this is not
            # a budget problem, so don't frame it as one.
            problem_line = (
                f"No flights from **{origin}** to **{destination}**, and no other "
                f"destinations reachable from **{origin}** were found in our database."
            )
        else:
            # Candidate cities existed but none had flights/hotels within budget.
            budget_line = (
                f" within your **${budget:.0f}** budget"
                if isinstance(budget, (int, float)) and budget
                else ""
            )
            problem_line = (
                f"No flights from **{origin}** to **{destination}**, and no reachable "
                f"alternative destinations{budget_line} were found."
            )

        choice: str = interrupt({
            "question": (
                f"{problem_line}\n\n"
                "Would you be open to adjusting your budget or your origin/destination?"
            ),
            "options": [
                ("flexible", "Yes, let me adjust something"),
                ("give_up", "No, stop here"),
            ],
        })

        if (choice or "").strip().lower() != "flexible":
            return {"flexibility_action": "give_up"}

        adjustment_text: str = interrupt({
            "question": (
                "What would you like to change? "
                "(e.g. 'raise my budget to $9000', 'change destination to Rome')"
            ),
            "options": [],
        })

        return {
            "flexibility_action": "flexible",
            "flexibility_attempts": attempts + 1,
            "alternative_destinations": [],
            "messages": [HumanMessage(content=adjustment_text)],
        }
