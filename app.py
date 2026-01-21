"""
CLI entry points for running the LangGraph insights pipeline.

This module exposes a ``run_all_sources`` function which iterates over
``config.DATA_SOURCES`` and executes the compiled pipeline for each label.
The output is logged to the console and any fatal errors are surfaced.
"""

from typing import Any

from . import config
from . import state
from . import graph_builder


def run_all_sources() -> None:
    """Run the pipeline once for each data source defined in the config."""
    app = graph_builder.build_app()
    for label in config.DATA_SOURCES.keys():
        initial_state: state.PipelineState = {"label": label}
        print("\n==============================")
        print(f"Running pipeline for label='{label}'")
        print("===============================\n")
        final_state = app.invoke(initial_state)
        print("✅ Final state keys:", list(final_state.keys()))


def main() -> None:
    """Script entry point when executed as ``python -m src.app``."""
    run_all_sources()


if __name__ == "__main__":  # pragma: no cover
    main()
