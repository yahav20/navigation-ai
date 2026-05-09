from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from agent.state import AgentState
from agent.edge import should_continue, after_enrichment
from agent.llm import get_models
from tools.tools import tools

from agent.nodes.metadata import MetadataNode
from agent.nodes.enrichment import EnrichmentNode
from agent.nodes.agent_core import AgentNode
from agent.nodes.summary import SummaryNode
from agent.nodes.formatting import FormatterNode 
from agent.nodes.node_alternative import AlternativeDestinationNode, FormatterAlternativeNode


def build_graph(provider: str = "google"):
    """
    Builds the graph using the specified model provider ('google' or 'groq').
    """
    # 1. Create the nodes with the chosen model provider
    model_with_tools, extraction_model = get_models(provider)
    
    extract_metadata_node = MetadataNode(extraction_model)
    enrichment_node = EnrichmentNode(extraction_model)
    call_model_node = AgentNode(model_with_tools)
    summary_node = SummaryNode(extraction_model)
    formatter = FormatterNode(extraction_model)
    alternative_destination_node = AlternativeDestinationNode(extraction_model)
    formatter_alternative = FormatterAlternativeNode(extraction_model)

    # 2. Build the standard graph
    builder = StateGraph(AgentState)

    builder.add_node("extract_metadata", extract_metadata_node)
    builder.add_node("enrichment", enrichment_node)
    builder.add_node("agent", call_model_node)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("formatter", formatter)
    builder.add_node("alternative_destination", alternative_destination_node)
    builder.add_node("formatter_alternative", formatter_alternative)
    builder.add_node("summary", summary_node)

    # 3. Define the workflow edges
    builder.add_edge(START, "extract_metadata")
    builder.add_edge("extract_metadata", "enrichment")
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