"""Build the LangGraph state graph for the travel agent."""
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from agent.edge import (
    after_enrichment,
    after_flight_search,
    after_router,
    after_travel_agent,
    after_security_gate,
    after_alternative_destination,
    after_itinerary_planner,
    after_itinerary_executor,
    after_itinerary_observer,
    after_itinerary_fallback,
    chat_should_continue,
    rec_should_continue,
)
from agent.llm import get_models
from agent.nodes.adjustments import AdjustmentsNode
from agent.nodes.enrichment import EnrichmentNode
from agent.nodes.flight_search import FlightSearchNode
from agent.nodes.formatting import FormatterNode
from agent.nodes.metadata import MetadataNode
from agent.nodes.node_alternative import AlternativeDestinationNode, FormatterAlternativeNode
from agent.nodes.general_chat import GeneralChatNode
from agent.nodes.rec_agent import RecommendationAgentNode
from agent.nodes.rec_formatter import RecommendationFormatterNode
from agent.nodes.router import RouterNode
from agent.nodes.summary import SummaryNode
from agent.nodes.travel_agent import TravelAgentNode
from agent.nodes.security_gate import security_gate_node
from agent.nodes.itinerary.planner import ItineraryPlannerNode
from agent.nodes.itinerary.executor import ItineraryExecutorNode
from agent.nodes.itinerary.observer import ItineraryObserverNode
from agent.nodes.itinerary.itinerary_fallback import ItineraryFallbackNode
from agent.nodes.itinerary.itinerary_formatter import ItineraryFormatterNode
from agent.state import AgentState
from tools.rec_tools import rec_tools


def build_graph(
    provider: str = "google",
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:

    # ── Models ──────────────────────────────────────────────────────────
    response_model, extraction_model = get_models(provider)
    rec_model, rec_extraction_model   = get_models(provider, mode="recommendation")
    chat_model, _                     = get_models(provider, mode="recommendation")

    # ── Nodes ────────────────────────────────────────────────────────────
    builder = StateGraph(AgentState)

    # Security & routing
    builder.add_node("security_gate",    security_gate_node)
    builder.add_node("router",           RouterNode(extraction_model))

    # Standard travel planning
    builder.add_node("extract_metadata",        MetadataNode(extraction_model))
    builder.add_node("adjustments",             AdjustmentsNode(extraction_model))
    builder.add_node("enrichment",              EnrichmentNode(extraction_model))
    builder.add_node("flight_search",           FlightSearchNode())
    builder.add_node("travel_agent",            TravelAgentNode(response_model))
    builder.add_node("formatter",               FormatterNode(response_model))
    builder.add_node("alternative_destination", AlternativeDestinationNode(extraction_model))
    builder.add_node("formatter_alternative",   FormatterAlternativeNode(extraction_model))
    builder.add_node("summary",                 SummaryNode(extraction_model))

    # Itinerary — Plan & Execute
    builder.add_node("itinerary_planner",   ItineraryPlannerNode(response_model))
    builder.add_node("itinerary_executor",  ItineraryExecutorNode())         # no LLM
    builder.add_node("itinerary_observer",  ItineraryObserverNode())         # no LLM
    builder.add_node("itinerary_fallback",  ItineraryFallbackNode(response_model, extraction_model))
    builder.add_node("itinerary_formatter", ItineraryFormatterNode(response_model))

    # Recommendations
    builder.add_node("rec_agent",     RecommendationAgentNode(rec_model, rec_extraction_model))
    builder.add_node("rec_tools",     ToolNode(rec_tools))
    builder.add_node("rec_formatter", RecommendationFormatterNode(rec_extraction_model))

    # General chat
    builder.add_node("general_chat", GeneralChatNode(chat_model, extraction_model))
    builder.add_node("chat_tools",   ToolNode(rec_tools))

    # ── Edges ────────────────────────────────────────────────────────────

    # Entry
    builder.add_edge(START, "security_gate")
    builder.add_conditional_edges("security_gate", after_security_gate,
                                  {"router": "router", "summary": "summary"})

    # Router dispatch
    # NOTE: "update_itinerary" must be in the map — it routes to itinerary_planner or adjustments
    builder.add_conditional_edges(
        "router", after_router,
        {
            "extract_metadata":  "extract_metadata",
            "adjustments":       "adjustments",
            "itinerary_planner": "itinerary_planner",
            "rec_agent":         "rec_agent",
            "general_chat":      "general_chat",
            END:                 END,
        },
    )

    # Standard planning path
    builder.add_edge("extract_metadata", "enrichment")
    builder.add_edge("adjustments",      "enrichment")
    builder.add_conditional_edges("enrichment", after_enrichment,
                                  {"flight_search": "flight_search", END: END})
    builder.add_conditional_edges(
        "flight_search", after_flight_search,
        {
            "itinerary_planner":      "itinerary_planner",
            "travel_agent":           "travel_agent",
            "alternative_destination":"alternative_destination",
        },
    )
    builder.add_conditional_edges(
        "travel_agent", after_travel_agent,
        {
            "formatter":         "formatter",
            "itinerary_planner": "itinerary_planner",
            "summary":           "summary",
        },
    )
    builder.add_conditional_edges("alternative_destination", after_alternative_destination,
                                  {"formatter_alternative": "formatter_alternative"})
    builder.add_edge("formatter_alternative", "summary")
    builder.add_edge("formatter",             "summary")
    builder.add_edge("summary",               END)

    # ── Itinerary Plan & Execute sub-graph ───────────────────────────────
    #
    #  itinerary_planner
    #       ↓ (feasible)          ↓ (not feasible)
    #  itinerary_executor      itinerary_fallback
    #       ↓                       ↓ (retry)    ↓ (show alternatives)
    #  itinerary_observer  ←───────┘         itinerary_formatter → summary
    #       ↓ (ok)   ↓ (re-plan) ↓ (fallback)
    #    summary  planner      fallback
    #
    builder.add_conditional_edges(
        "itinerary_planner", after_itinerary_planner,
        {
            "itinerary_executor": "itinerary_executor",
            "itinerary_fallback": "itinerary_fallback",
        },
    )
    builder.add_conditional_edges(
        "itinerary_executor", after_itinerary_executor,
        {"itinerary_observer": "itinerary_observer"},
    )
    builder.add_conditional_edges(
        "itinerary_observer", after_itinerary_observer,
        {
            "itinerary_planner":  "itinerary_planner",   # re-plan
            "itinerary_fallback": "itinerary_fallback",  # unrecoverable
            "summary":            "summary",             # done
        },
    )
    builder.add_conditional_edges(
        "itinerary_fallback", after_itinerary_fallback,
        {
            "itinerary_planner":   "itinerary_planner",
            "itinerary_formatter": "itinerary_formatter",
        },
    )
    builder.add_edge("itinerary_formatter", "summary")

    # Recommendations
    builder.add_conditional_edges("rec_agent", rec_should_continue,
                                  {"rec_tools": "rec_tools", "rec_formatter": "rec_formatter"})
    builder.add_edge("rec_tools",     "rec_agent")
    builder.add_edge("rec_formatter", "summary")

    # General chat
    builder.add_conditional_edges("general_chat", chat_should_continue,
                                  {"chat_tools": "chat_tools", "summary": "summary"})
    builder.add_edge("chat_tools", "general_chat")

    # ── Compile ──────────────────────────────────────────────────────────
    if checkpointer is None:
        checkpointer = MemorySaver(serde=JsonPlusSerializer())
    return builder.compile(checkpointer=checkpointer)