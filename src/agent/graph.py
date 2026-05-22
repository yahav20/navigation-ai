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
from agent.nodes.itinerary.itinerary_planner import ItineraryPlannerNode
from agent.nodes.itinerary.itinerary_builder import ItineraryBuilderNode
from agent.nodes.itinerary.itinerary_fallback import ItineraryFallbackNode
from agent.nodes.itinerary.itinerary_formatter import ItineraryFormatterNode
from agent.state import AgentState
from tools.rec_tools import rec_tools


def build_graph(
    provider: str = "google",
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    # ------------------------------------------------------------------ #
    # 1. Instantiate models                                                #
    # ------------------------------------------------------------------ #
    response_model, extraction_model = get_models(provider)
    rec_model_with_tools, rec_extraction_model = get_models(provider, mode="recommendation")
    chat_model_with_tools, _ = get_models(provider, mode="recommendation")

    # ------------------------------------------------------------------ #
    # 2. Instantiate nodes                                                 #
    # ------------------------------------------------------------------ #

    # — Core travel planning —
    router_node               = RouterNode(extraction_model)
    extract_metadata_node     = MetadataNode(extraction_model)
    adjustments_node          = AdjustmentsNode(extraction_model)
    enrichment_node           = EnrichmentNode(extraction_model)
    flight_search_node        = FlightSearchNode()
    travel_agent_node         = TravelAgentNode(response_model)
    formatter_node            = FormatterNode(response_model)
    alternative_destination_node = AlternativeDestinationNode(extraction_model)
    formatter_alternative_node   = FormatterAlternativeNode(extraction_model)
    summary_node              = SummaryNode(extraction_model)

    # — Itinerary sub-graph —
    itinerary_planner_node    = ItineraryPlannerNode(response_model)
    itinerary_builder_node    = ItineraryBuilderNode()          # no LLM
    itinerary_fallback_node   = ItineraryFallbackNode(response_model, extraction_model)
    itinerary_formatter_node  = ItineraryFormatterNode(response_model)

    # — Recommendation path —
    rec_agent_node            = RecommendationAgentNode(rec_model_with_tools, rec_extraction_model)
    rec_formatter_node        = RecommendationFormatterNode(rec_extraction_model)

    # — General chat path —
    general_chat_node         = GeneralChatNode(chat_model_with_tools, extraction_model)

    # ------------------------------------------------------------------ #
    # 3. Register nodes                                                    #
    # ------------------------------------------------------------------ #
    builder = StateGraph(AgentState)

    builder.add_node("security_gate",           security_gate_node)
    builder.add_node("router",                  router_node)
    builder.add_node("extract_metadata",        extract_metadata_node)
    builder.add_node("adjustments",             adjustments_node)
    builder.add_node("enrichment",              enrichment_node)
    builder.add_node("flight_search",           flight_search_node)
    builder.add_node("travel_agent",            travel_agent_node)
    builder.add_node("formatter",               formatter_node)
    builder.add_node("alternative_destination", alternative_destination_node)
    builder.add_node("formatter_alternative",   formatter_alternative_node)
    builder.add_node("summary",                 summary_node)

    builder.add_node("itinerary_planner",       itinerary_planner_node)
    builder.add_node("itinerary_builder",       itinerary_builder_node)
    builder.add_node("itinerary_fallback",      itinerary_fallback_node)
    builder.add_node("itinerary_formatter",     itinerary_formatter_node)

    builder.add_node("rec_agent",               rec_agent_node)
    builder.add_node("rec_tools",               ToolNode(rec_tools))
    builder.add_node("rec_formatter",           rec_formatter_node)

    builder.add_node("general_chat",            general_chat_node)
    builder.add_node("chat_tools",              ToolNode(rec_tools))

    # ------------------------------------------------------------------ #
    # 4. Edges — entry & security                                          #
    # ------------------------------------------------------------------ #
    builder.add_edge(START, "security_gate")

    builder.add_conditional_edges(
        "security_gate",
        after_security_gate,
        {"router": "router", "summary": "summary"},
    )

    # ------------------------------------------------------------------ #
    # 5. Edges — router dispatch                                           #
    #                                                                      #
    # The router can send to itinerary_planner DIRECTLY when              #
    # flight_options are already in state (mid-conversation case).        #
    # Otherwise intent=itinerary goes via extract_metadata first.         #
    # ------------------------------------------------------------------ #
    builder.add_conditional_edges(
        "router",
        after_router,
        {
            "extract_metadata":   "extract_metadata",
            "adjustments":        "adjustments",
            "rec_agent":          "rec_agent",
            "itinerary_planner":  "itinerary_planner",   # mid-conversation shortcut
            "general_chat":       "general_chat",
            END:                  END,
        },
    )

    # ------------------------------------------------------------------ #
    # 6. Edges — standard travel planning path                             #
    # ------------------------------------------------------------------ #
    builder.add_edge("extract_metadata", "enrichment")
    builder.add_edge("adjustments",      "enrichment")

    builder.add_conditional_edges(
        "enrichment",
        after_enrichment,
        {"flight_search": "flight_search", END: END},
    )

    # flight_search is the GATE for itinerary:
    #   - no flights   → alternative_destination  (same as before)
    #   - flights + build_itinerary flag → itinerary_planner  ← NEW
    #   - flights only → travel_agent
    builder.add_conditional_edges(
        "flight_search",
        after_flight_search,
        {
            "travel_agent":        "travel_agent",
            "itinerary_planner":   "itinerary_planner",
            "alternative_destination": "alternative_destination",
        },
    )

    builder.add_conditional_edges(
        "travel_agent",
        after_travel_agent,
        {
            "formatter":          "formatter",
            "itinerary_planner":  "itinerary_planner",   # user asked for full trip mid-flow
            "summary":            "summary",
        },
    )

    # alternative_destination → formatter_alternative (unchanged)
    # BUT: on re-entry (next user message) the router handles the new intent.
    builder.add_conditional_edges(
        "alternative_destination",
        after_alternative_destination,
        {"formatter_alternative": "formatter_alternative"},
    )

    builder.add_edge("formatter_alternative", "summary")
    builder.add_edge("formatter",             "summary")
    builder.add_edge("summary",               END)

    # ------------------------------------------------------------------ #
    # 7. Edges — itinerary sub-graph                                       #
    # ------------------------------------------------------------------ #
    builder.add_conditional_edges(
        "itinerary_planner",
        after_itinerary_planner,
        {
            "itinerary_builder":  "itinerary_builder",
            "itinerary_fallback": "itinerary_fallback",
        },
    )

    builder.add_conditional_edges(
        "itinerary_fallback",
        after_itinerary_fallback,
        {
            "itinerary_planner":   "itinerary_planner",    # retry after adjustment
            "itinerary_formatter": "itinerary_formatter",  # show alternatives
        },
    )

    builder.add_edge("itinerary_builder",   "itinerary_formatter")
    builder.add_edge("itinerary_formatter", "summary")

    # ------------------------------------------------------------------ #
    # 8. Edges — recommendation path                                       #
    # ------------------------------------------------------------------ #
    builder.add_conditional_edges(
        "rec_agent",
        rec_should_continue,
        {"rec_tools": "rec_tools", "rec_formatter": "rec_formatter"},
    )
    builder.add_edge("rec_tools",      "rec_agent")
    builder.add_edge("rec_formatter",  "summary")

    # ------------------------------------------------------------------ #
    # 9. Edges — general chat path                                         #
    # ------------------------------------------------------------------ #
    builder.add_conditional_edges(
        "general_chat",
        chat_should_continue,
        {"chat_tools": "chat_tools", "summary": "summary"},
    )
    builder.add_edge("chat_tools", "general_chat")

    # ------------------------------------------------------------------ #
    # 10. Compile                                                           #
    # ------------------------------------------------------------------ #
    if checkpointer is None:
        checkpointer = MemorySaver(serde=JsonPlusSerializer())
    return builder.compile(checkpointer=checkpointer)