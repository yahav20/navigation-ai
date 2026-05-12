"""Build the LangGraph state graph for the travel agent."""
from agent.nodes.adjustments import AdjustmentsNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from agent.edge import after_enrichment, should_continue, after_router
from agent.llm import get_models
from agent.nodes.router import RouterNode
from agent.nodes.agent_core import AgentNode
from agent.nodes.enrichment import EnrichmentNode
from agent.nodes.formatting import FormatterNode
from agent.nodes.metadata import MetadataNode
from agent.nodes.node_alternative import (
    AlternativeDestinationNode,
    FormatterAlternativeNode,
)
from agent.nodes.summary import SummaryNode
from agent.state import AgentState
from tools import core_tools

# TODO: Replace with actual recommendations node
def dummy_recommendations_node(state: AgentState):
    from langchain_core.messages import AIMessage
    return {"messages": [AIMessage(content="I am the recommendations agent! My code is coming soon.")]}


def build_graph(provider: str = "google") -> CompiledStateGraph:
    """Build the graph using the specified model provider ('google' or 'groq')."""
    # 1. Create the nodes with the chosen model provider
    model_with_tools, extraction_model = get_models(provider)

    extract_metadata_node = MetadataNode(extraction_model)
    adjustments_node = AdjustmentsNode(extraction_model)
    enrichment_node = EnrichmentNode(extraction_model)
    call_model_node = AgentNode(model_with_tools)
    summary_node = SummaryNode(extraction_model)
    formatter = FormatterNode(extraction_model)
    alternative_destination_node = AlternativeDestinationNode(extraction_model)
    formatter_alternative = FormatterAlternativeNode(extraction_model)
    router_node = RouterNode(extraction_model)

    # 2. Build the standard graph
    builder = StateGraph(AgentState)

    builder.add_node("router", router_node)
    builder.add_node("recommendations", dummy_recommendations_node)
    builder.add_node("extract_metadata", extract_metadata_node)
    builder.add_node("adjustments", adjustments_node)
    builder.add_node("enrichment", enrichment_node)
    builder.add_node("agent", call_model_node)
    builder.add_node("tools", ToolNode(core_tools))
    builder.add_node("formatter", formatter)
    builder.add_node("alternative_destination", alternative_destination_node)
    builder.add_node("formatter_alternative", formatter_alternative)
    builder.add_node("summary", summary_node)

    # 3. Define the workflow edges
    builder.add_edge(START, "router")
    
    builder.add_conditional_edges(
        "router", 
        after_router, 
        {
            "extract_metadata": "extract_metadata", 
            "adjustments": "adjustments",
            "recommendations": "recommendations",
            END: END
        }
    )
    
    builder.add_edge("extract_metadata", "enrichment")
    builder.add_edge("adjustments", "enrichment")

    builder.add_edge("recommendations", "summary")
    builder.add_conditional_edges("enrichment", after_enrichment, {"agent": "agent", END: END})
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "formatter": "formatter",
            "alternative_destination": "alternative_destination",
        },
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("alternative_destination", "formatter_alternative")
    builder.add_edge("formatter_alternative", "summary")
    builder.add_edge("formatter", "summary")
    # The summary node marks the end of the processing cycle for the current turn
    builder.add_edge("summary", END)
    # Adding a checkpointer to save the agent's state across turns
    serializer = JsonPlusSerializer()
    memory = MemorySaver(serde=serializer)
    return builder.compile(checkpointer=memory)
