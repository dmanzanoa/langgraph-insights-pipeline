"""
Graph builder utilities for composing the LangGraph pipelines.

This module defines helper functions for registering the various nodes
implemented in ``pipeline_nodes`` into a ``StateGraph``.  The pipelines are
segmented into three stages: the main insights pipeline, the monthly
tendencias pipeline and the per‑subproject pipeline.  A convenience
``build_app`` function assembles all stages into a single compiled app.
"""

from langgraph.graph import StateGraph, END  # type: ignore

from . import state
from . import pipeline_nodes as nodes


def add_insights_pipeline(graph: StateGraph) -> None:
    """Register nodes and edges for the main insights pipeline."""
    graph.add_node("load_data", nodes.node_load_data)
    graph.add_node("preprocess", nodes.node_preprocess)
    graph.add_node("summarize_conversations", nodes.node_summarize_conversations)
    graph.add_node("generate_insights", nodes.node_generate_insights)
    graph.add_node("validate_json", nodes.node_validate_json)
    graph.add_node("save_to_s3", nodes.node_save_to_s3)
    graph.add_node("save_fatal_error", nodes.node_save_fatal_error_to_s3)
    graph.set_entry_point("load_data")
    # conditional edge after load
    graph.add_conditional_edges(
        "load_data",
        nodes.route_after_load,
        {"skip": END, "continue": "preprocess"},
    )
    # linear edges
    graph.add_edge("preprocess", "summarize_conversations")
    graph.add_edge("summarize_conversations", "generate_insights")
    graph.add_edge("generate_insights", "validate_json")
    # retry router for insights
    def retry_router(state_dict: state.PipelineState) -> str:
        if state_dict.get("fatal_error"):
            return "fatal"
        if state_dict.get("is_valid_json"):
            return "valid"
        return "retry"
    graph.add_conditional_edges(
        "validate_json",
        retry_router,
        {
            "valid": "save_to_s3",
            "retry": "generate_insights",
            "fatal": "save_fatal_error",
        },
    )
    graph.add_edge("save_to_s3", END)
    graph.add_edge("save_fatal_error", END)


def add_tendencias_pipeline(graph: StateGraph) -> None:
    """Register nodes and edges for the monthly tendencias pipeline."""
    graph.add_node("generate_monthly_tendencias", nodes.node_generate_monthly_tendencias)
    graph.add_node("validate_tendencias_json", nodes.node_validate_tendencias_json)
    graph.add_node("save_tendencias_to_s3", nodes.node_save_tendencias_to_s3)
    # run tendencias after saving insights
    graph.add_edge("save_to_s3", "generate_monthly_tendencias")
    graph.add_edge("generate_monthly_tendencias", "validate_tendencias_json")
    # router for tendencias
    def tendencias_router(state_dict: state.PipelineState) -> str:
        if state_dict.get("fatal_error"):
            return "fatal"
        if state_dict.get("tendencias_valid"):
            return "valid"
        return "retry"
    graph.add_conditional_edges(
        "validate_tendencias_json",
        tendencias_router,
        {
            "valid": "save_tendencias_to_s3",
            "retry": "generate_monthly_tendencias",
            "fatal": "save_fatal_error",
        },
    )
    graph.add_edge("save_tendencias_to_s3", END)


def add_subproject_pipeline(graph: StateGraph) -> None:
    """Register nodes and edges for the per‑project insights pipeline."""
    graph.add_node("generate_insights_by_subproject", nodes.node_generate_insights_by_subproject)
    graph.add_node("validate_subproject_insights", nodes.node_validate_subproject_insights)
    graph.add_node("save_subproject_insights", nodes.node_save_subproject_insights_to_s3)
    # start after tendencias are saved
    graph.add_edge("save_tendencias_to_s3", "generate_insights_by_subproject")
    graph.add_edge("generate_insights_by_subproject", "validate_subproject_insights")
    # router
    def subproject_router(state_dict: state.PipelineState) -> str:
        if state_dict.get("fatal_error"):
            return "fatal"
        if state_dict.get("subproject_valid"):
            return "valid"
        return "retry"
    graph.add_conditional_edges(
        "validate_subproject_insights",
        subproject_router,
        {
            "valid": "save_subproject_insights",
            "retry": "generate_insights_by_subproject",
            "fatal": "save_fatal_error",
        },
    )
    graph.add_edge("save_subproject_insights", END)
    graph.add_edge("save_fatal_error", END)


def build_app() -> Any:
    """Construct and compile the LangGraph app with all pipelines."""
    graph = StateGraph(state.PipelineState)
    # register fatal error node globally
    graph.add_node("save_fatal_error", nodes.node_save_fatal_error_to_s3)
    add_insights_pipeline(graph)
    add_tendencias_pipeline(graph)
    add_subproject_pipeline(graph)
    app = graph.compile()
    return app


__all__ = [
    "add_insights_pipeline",
    "add_tendencias_pipeline",
    "add_subproject_pipeline",
    "build_app",
]