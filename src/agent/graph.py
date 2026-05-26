"""Build the LangGraph state graph for the travel agent."""
from agent.nodes.adjustments import AdjustmentsNode
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from agent.edge import after_enrichment, after_router, should_continue
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
from agent.nodes.advisor_planner import AdvisorPlannerNode
from agent.nodes.advisor_executor import AdvisorExecutorNode
from agent.nodes.advisor_formatter import AdvisorFormatterNode
from agent.nodes.advisor_replanner import AdvisorReplannerNode
from agent.nodes.summary import SummaryNode
from agent.state import AgentState
from tools import core_tools


def build_graph(
    provider: str = "google",
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Build the graph using the specified model provider ('google' or 'groq').

    If no `checkpointer` is supplied, falls back to an in-memory saver so the
    graph still works for tests and one-shot invocations. Persistent runs
    should pass a durable checkpointer (e.g. `SqliteSaver`).
    """
    # 1. Create nodes for the travel planning path
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

    # 2. Create nodes for the advisor path (uses its own model)
    _, advisor_extraction_model = get_models(provider, mode="advisor")
    advisor_planner_node = AdvisorPlannerNode(advisor_extraction_model)
    advisor_executor_node = AdvisorExecutorNode()
    advisor_replanner_node = AdvisorReplannerNode(advisor_extraction_model)
    advisor_formatter_node = AdvisorFormatterNode(advisor_extraction_model)

    # 3. Build the graph
    builder = StateGraph(AgentState)

    # Travel planning nodes
    builder.add_node("router", router_node)
    builder.add_node("extract_metadata", extract_metadata_node)
    builder.add_node("adjustments", adjustments_node)
    builder.add_node("enrichment", enrichment_node)
    builder.add_node("agent", call_model_node)
    builder.add_node("tools", ToolNode(core_tools))
    builder.add_node("formatter", formatter)
    builder.add_node("alternative_destination", alternative_destination_node)
    builder.add_node("formatter_alternative", formatter_alternative)
    builder.add_node("summary", summary_node)

    # Advisor nodes
    builder.add_node("advisor_planner", advisor_planner_node)
    builder.add_node("advisor_executor", advisor_executor_node)
    builder.add_node("advisor_replanner", advisor_replanner_node)
    builder.add_node("advisor_formatter", advisor_formatter_node)

    # 4. Define edges — travel planning path
    builder.add_edge(START, "router")

    builder.add_conditional_edges(
        "router",
        after_router,
        {
            "extract_metadata": "extract_metadata",
            "adjustments": "adjustments",
            "advisor_planner": "advisor_planner",
            END: END,
        },
    )

    builder.add_edge("extract_metadata", "enrichment")
    builder.add_edge("adjustments", "enrichment")
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
    builder.add_edge("summary", END)

    # 5. Define edges — advisor path (Plan-and-Execute loop)
    def _after_advisor_replan(state: AgentState) -> str:
        return "advisor_formatter" if not state.get("advisor_plan") else "advisor_executor"

    builder.add_edge("advisor_planner", "advisor_executor")
    builder.add_edge("advisor_executor", "advisor_replanner")
    builder.add_conditional_edges("advisor_replanner", _after_advisor_replan)
    builder.add_edge("advisor_formatter", "summary")

    if checkpointer is None:
        checkpointer = MemorySaver(serde=JsonPlusSerializer())
    return builder.compile(checkpointer=checkpointer)
