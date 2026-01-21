"""
Top‑level package for the LangGraph insights pipeline.

Importing this package has no side effects.  Individual modules such as
``config``, ``state``, ``preprocessing`` and ``graph_builder`` should be
imported explicitly by consumers.  A convenience function
``app.run_all_sources`` is provided in ``src/app.py`` to execute the
pipeline end to end.
"""

__all__ = [
    "config",
    "state",
    "prompts",
    "preprocessing",
    "aggregation",
    "data_loading",
    "summarization",
    "insights",
    "pipeline_nodes",
    "graph_builder",
    "app",
]