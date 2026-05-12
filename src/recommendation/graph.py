"""Build the LangGraph state graph for the recommendation agent."""
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from recommendation.edge import rec_should_continue
from recommendation.llm import get_rec_models
from recommendation.nodes.rec_agent import RecommendationAgentNode
from recommendation.nodes.rec_formatter import RecommendationFormatterNode
from recommendation.nodes.rec_summary import RecSummaryNode
from recommendation.state import RecommendationState
from recommendation.tools.rec_tools import rec_tools  # needed for ToolNode


def build_recommendation_graph(
    provider: str = "google",
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Build the recommendation graph.

    Accepts the same arguments as build_graph() so main.py needs only a single import swap.
    """
    model_with_tools, extraction_model = get_rec_models("openai")

    rec_agent_node = RecommendationAgentNode(model_with_tools)
    rec_formatter_node = RecommendationFormatterNode(extraction_model)
    rec_summary_node = RecSummaryNode(extraction_model)

    builder = StateGraph(RecommendationState)

    builder.add_node("rec_agent", rec_agent_node)
    builder.add_node("tools", ToolNode(rec_tools))
    builder.add_node("rec_formatter", rec_formatter_node)
    builder.add_node("rec_summary", rec_summary_node)

    builder.add_edge(START, "rec_agent")
    builder.add_conditional_edges(
        "rec_agent",
        rec_should_continue,
        {"tools": "tools", "rec_formatter": "rec_formatter"},
    )
    builder.add_edge("tools", "rec_agent")
    builder.add_edge("rec_formatter", "rec_summary")
    builder.add_edge("rec_summary", END)

    if checkpointer is None:
        checkpointer = MemorySaver(serde=JsonPlusSerializer())

    return builder.compile(checkpointer=checkpointer)
