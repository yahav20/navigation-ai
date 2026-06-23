"""Build the LangGraph state graph for the travel agent."""
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.core.edge import (
    after_flight_search,
    after_enrichment,
    after_router,
    after_advisor_planner,
    after_travel_agent,
    after_travel_formatter,
    after_travel_confirmation,
    after_security_gate,
    after_alternative_destination,
    after_flight_flexibility_gate,
)
from agent.core.llm import get_generation_model, get_models
from agent.core.state import AgentState
from agent.shared.enrichment import EnrichmentNode
from agent.shared.general_chat import GeneralChatNode
from agent.shared.metadata import MetadataNode
from agent.shared.router import RouterNode
from agent.shared.save_plan import SavePlanPromptNode
from agent.travel.confirmation import TravelConfirmationNode
from agent.shared.security_gate import make_security_gate_node
from agent.shared.injection_guard import InjectionGuard
from agent.shared.summary import SummaryNode
from agent.travel.adjustments import AdjustmentsNode
from agent.travel.agent import TravelAgentNode
from agent.travel.alternatives import AlternativeDestinationNode, FormatterAlternativeNode
from agent.travel.flight_flexibility import FlightFlexibilityGateNode
from agent.travel.flight_search import FlightSearchNode
from agent.travel.formatter import FormatterNode
from agent.advisor.planner import AdvisorPlannerNode
from agent.advisor.executor import AdvisorExecutorNode
from agent.advisor.formatter import AdvisorFormatterNode
from agent.advisor.replanner import AdvisorReplannerNode
from agent.itinerary.plan_check import PlanCheckNode
from agent.itinerary.planner import ItineraryPlannerNode
from agent.itinerary.executor import ItineraryExecutorNode
from agent.itinerary.replanner import ItineraryReplannerNode
from agent.itinerary.critic import ItineraryCriticNode
from agent.itinerary.itinerary_edges import (
    after_plan_check,
    after_itinerary_planner,
    after_itinerary_replanner,
    after_itinerary_critic,
)
from agent.itinerary.multi_dest import (
    SegmentPlannerNode,
    RouteSelectNode,
    LegDispatchNode,
    LegCollectNode,
    TripFormatterNode,
    after_segment_planner,
    after_route_select,
    after_leg_collect,
)
from tools import general_chat_tools
from tools.destinations import advisor_tools

_all_chat_tools = advisor_tools + general_chat_tools


def _out_of_scope_node(state: AgentState) -> dict:
    from langchain_core.messages import AIMessage
    return {"messages": [AIMessage(
        content="I'm a travel assistant — I can only help with trips, destinations, and travel questions. "
                "Is there somewhere you'd like to explore? ✈️",
        name="out_of_scope",
    )]}


def _assemble_builder(provider: str = "google") -> StateGraph:
    """Assemble the travel-agent StateGraph (nodes + edges), uncompiled."""
    # 1. Create nodes for the travel planning path
    response_model, extraction_model = get_models(provider)
    # travel_agent + formatter are the user-facing generation nodes; on OpenAI
    # they run on gpt-5.4-mini (~2x faster than gpt-4o-mini at equal quality).
    generation_model = get_generation_model(provider)

    extract_metadata_node = MetadataNode(extraction_model)
    adjustments_node = AdjustmentsNode(extraction_model)
    enrichment_node = EnrichmentNode(extraction_model)
    flight_search_node = FlightSearchNode()
    travel_agent_node = TravelAgentNode(generation_model)
    summary_node = SummaryNode(extraction_model)
    formatter = FormatterNode()
    travel_confirmation_node = TravelConfirmationNode()
    alternative_destination_node = AlternativeDestinationNode(extraction_model)
    formatter_alternative = FormatterAlternativeNode()
    flight_flexibility_gate_node = FlightFlexibilityGateNode()
    router_node = RouterNode(extraction_model)
    # save_plan_prompt HITL disabled for now — not wired into the graph below.
    # save_plan_prompt_node = SavePlanPromptNode()

    #   -Itinerary
    plan_check_node           = PlanCheckNode()
    segment_planner_node      = SegmentPlannerNode(extraction_model)
    route_select_node         = RouteSelectNode()
    multi_flight_node         = FlightSearchNode()
    leg_dispatch_node         = LegDispatchNode()
    itinerary_planner_node    = ItineraryPlannerNode(response_model)
    itinerary_executor_node   = ItineraryExecutorNode(response_model)
    itinerary_replanner_node  = ItineraryReplannerNode(response_model)
    itinerary_critic_node     = ItineraryCriticNode()
    leg_collect_node          = LegCollectNode()
    trip_formatter_node       = TripFormatterNode(response_model)

    # 2. Create nodes for the advisor path (uses its own model)
    _, advisor_extraction_model = get_models(provider, mode="advisor")
    advisor_planner_node = AdvisorPlannerNode(advisor_extraction_model)
    advisor_executor_node = AdvisorExecutorNode()
    advisor_replanner_node = AdvisorReplannerNode(advisor_extraction_model)
    advisor_formatter_node = AdvisorFormatterNode(advisor_extraction_model)

    # 3. Build the graph
    builder = StateGraph(AgentState)

    # Travel planning nodes
    injection_guard = InjectionGuard(extraction_model)
    builder.add_node("security_gate", make_security_gate_node(injection_guard))
    builder.add_node("router", router_node)
    builder.add_node("extract_metadata", extract_metadata_node)
    builder.add_node("adjustments", adjustments_node)
    builder.add_node("enrichment", enrichment_node)
    builder.add_node("flight_search", flight_search_node)
    builder.add_node("travel_agent", travel_agent_node)
    builder.add_node("formatter", formatter)
    builder.add_node("travel_confirmation", travel_confirmation_node)
    # save_plan_prompt HITL disabled for now (see edges below); node not registered.
    # builder.add_node("save_plan_prompt", save_plan_prompt_node)
    builder.add_node("alternative_destination", alternative_destination_node)
    builder.add_node("formatter_alternative", formatter_alternative)
    builder.add_node("flight_flexibility_gate", flight_flexibility_gate_node)
    builder.add_node("summary", summary_node)

    # Itinerary nodes
    builder.add_node("plan_check",          plan_check_node)
    builder.add_node("segment_planner",     segment_planner_node)
    builder.add_node("route_select",        route_select_node)
    builder.add_node("multi_flight",        multi_flight_node)
    builder.add_node("leg_dispatch",        leg_dispatch_node)
    builder.add_node("itinerary_planner",   itinerary_planner_node)
    builder.add_node("itinerary_executor",  itinerary_executor_node)
    builder.add_node("itinerary_replanner", itinerary_replanner_node)
    builder.add_node("itinerary_critic",    itinerary_critic_node)
    builder.add_node("leg_collect",         leg_collect_node)
    builder.add_node("trip_formatter",      trip_formatter_node)

    # Advisor nodes
    builder.add_node("advisor_planner", advisor_planner_node)
    builder.add_node("advisor_executor", advisor_executor_node)
    builder.add_node("advisor_replanner", advisor_replanner_node)
    builder.add_node("advisor_formatter", advisor_formatter_node)

    # Out-of-scope rejection node (no LLM call — static message, goes straight to END)
    builder.add_node("out_of_scope", _out_of_scope_node)
    builder.add_edge("out_of_scope", END)

    # 5. Define edges — travel planning path
    builder.add_edge(START, "security_gate")

    builder.add_conditional_edges(
        "security_gate",
        after_security_gate,
        {
            "router": "router",
            "summary": "summary",
        }
    )

    builder.add_conditional_edges(
        "router",
        after_router,
        {
            "extract_metadata":  "extract_metadata",
            "adjustments":       "adjustments",
            "advisor_planner":   "advisor_planner",
            "itinerary_planner": "itinerary_planner",
            "plan_check":        "plan_check",
            "out_of_scope":      "out_of_scope",
            END: END,
        },
    )

    builder.add_edge("extract_metadata", "enrichment")
    builder.add_edge("adjustments", "enrichment")

    builder.add_conditional_edges(
        "enrichment",
        after_enrichment,
        {
            "flight_search": "flight_search",
            "plan_check":    "plan_check",
            END: END,
        }
    )

    builder.add_conditional_edges(
        "flight_search",
        after_flight_search,
        {
            "travel_agent": "travel_agent",
            "alternative_destination": "alternative_destination",
        },
    )
    builder.add_conditional_edges(
        "travel_agent",
        after_travel_agent,
        {
            "formatter": "formatter",
            "summary": "summary",
        },
    )
    builder.add_conditional_edges(
        "alternative_destination",
        after_alternative_destination,
        {
            "formatter_alternative": "formatter_alternative",
            "flight_flexibility_gate": "flight_flexibility_gate",
        },
    )
    builder.add_conditional_edges(
        "flight_flexibility_gate",
        after_flight_flexibility_gate,
        {
            "adjustments": "adjustments",
            "formatter_alternative": "formatter_alternative",
        },
    )
    builder.add_edge("formatter_alternative", "summary")

    # After the travel formatter renders flights + hotels, pause for user confirmation.
    # Only interrupts when a full plan (hotels + flights) exists; falls back to summary.
    builder.add_conditional_edges(
        "formatter",
        after_travel_formatter,
        {
            "travel_confirmation": "travel_confirmation",
            "summary": "summary",
        },
    )
    builder.add_conditional_edges(
        "travel_confirmation",
        after_travel_confirmation,
        {
            "plan_check": "plan_check",
            "summary": "summary",
        },
    )

    # -----   -Itinerary (Multi-destination round route → per-city Plan & Execute) -----
    # Single destination and multi destination share one path: plan_check decides
    # the mode, segment_planner builds the trip_segments list (1 city or N), and the
    # leg loop runs the same Plan-and-Execute nodes once per city.
    builder.add_conditional_edges(
        "plan_check",
        after_plan_check,
        {
            "segment_planner":  "segment_planner",
            "extract_metadata": "extract_metadata",
            "summary":          "summary",
        }
    )

    # segment_planner → (multi) pause for the user to pick a route → search the
    # entry-city flight once → leg loop.  Single trips skip straight to the loop.
    builder.add_conditional_edges(
        "segment_planner",
        after_segment_planner,
        {
            "route_select": "route_select",
            "leg_dispatch": "leg_dispatch",
        }
    )
    # Booked trips keep their flights and skip the re-search; standalone multi
    # trips search the entry-city flight once before the leg loop.
    builder.add_conditional_edges(
        "route_select",
        after_route_select,
        {
            "multi_flight": "multi_flight",
            "leg_dispatch": "leg_dispatch",
        }
    )
    builder.add_edge("multi_flight", "leg_dispatch")
    builder.add_edge("leg_dispatch", "itinerary_planner")

    builder.add_conditional_edges(
        "itinerary_planner",
        after_itinerary_planner,
        {
            "itinerary_executor": "itinerary_executor",
            "trip_formatter":     "trip_formatter",
        }
    )

    builder.add_edge("itinerary_executor",  "itinerary_replanner")

    builder.add_conditional_edges(
        "itinerary_replanner",
        after_itinerary_replanner,
        {
            "itinerary_executor": "itinerary_executor",
            "itinerary_planner":  "itinerary_planner",
            "itinerary_critic":   "itinerary_critic",
        }
    )

    builder.add_conditional_edges(
        "itinerary_critic",
        after_itinerary_critic,
        {
            "itinerary_planner": "itinerary_planner",
            "leg_collect":       "leg_collect",
        }
    )

    # leg_collect → next city (loop) or render the whole trip
    builder.add_conditional_edges(
        "leg_collect",
        after_leg_collect,
        {
            "leg_dispatch":   "leg_dispatch",
            "trip_formatter": "trip_formatter",
        }
    )

    builder.add_edge("trip_formatter", "summary")
    # ------------------------------------------------------

    builder.add_edge("summary", END)

    # 6. Define edges — advisor path (Plan-and-Execute loop)
    def _after_advisor_replan(state: AgentState) -> str:
        return "advisor_formatter" if not state.get("advisor_plan") else "advisor_executor"

    builder.add_conditional_edges("advisor_planner", after_advisor_planner)
    builder.add_edge("advisor_executor", "advisor_replanner")
    builder.add_conditional_edges("advisor_replanner", _after_advisor_replan)
    builder.add_edge("advisor_formatter", "summary")

    return builder


def build_graph(
    provider: str = "google",
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Compile the travel-agent graph for the CLI and tests.

    If no `checkpointer` is supplied, falls back to an in-memory saver so the
    graph still works for tests and one-shot invocations. Persistent runs
    should pass a durable checkpointer (e.g. `SqliteSaver`).
    """
    builder = _assemble_builder(provider)
    if checkpointer is None:
        checkpointer = MemorySaver(serde=JsonPlusSerializer())
    return builder.compile(checkpointer=checkpointer)


def make_graph() -> CompiledStateGraph:
    """Graph factory for the LangGraph API server (``langgraph dev`` / Platform).

    Referenced by ``langgraph.json``. Compiled WITHOUT a checkpointer — the
    server supplies its own persistence layer (threads/checkpoints), so a
    custom saver here would be ignored or conflict. Use ``build_graph`` for the
    CLI and tests instead.
    """
    from config.config import CHOSEN_PROVIDER

    return _assemble_builder(CHOSEN_PROVIDER).compile()
