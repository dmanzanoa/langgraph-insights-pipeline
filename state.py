"""
State definitions and helper functions for the LangGraph insights pipeline.

The pipeline passes around a ``state`` dictionary between nodes.  This module
defines a type alias to document the expected shape of that dictionary and
provides helper utilities for constructing fatal error objects.  Keeping
these definitions in a separate module avoids circular dependencies and
makes it easier to update the state schema as the pipeline evolves.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict

# The state object is a plain dictionary with string keys.  Nodes will
# add and modify keys as they run.  The use of ``Dict[str, Any]`` makes
# it clear that arbitrary data may be stored in the state.  If you wish to
# introduce stronger typing you can replace this alias with a ``TypedDict``.
PipelineState = Dict[str, Any]


def build_fatal_error_object(
    *,
    label: str,
    stage: str,
    fatal_reason: str,
    attempts: int,
    context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Construct a structured fatal error object for CloudWatch and S3.

    The returned dictionary contains metadata describing where in the
    pipeline the failure occurred, the number of attempts made and a
    human‑readable reason.  A UTC timestamp is appended to aid
    debugging.  Additional context can be passed in via the ``context``
    argument and will be included verbatim.

    Args:
        label: Identifier for the current data source (e.g. ``"subsidio"``).
        stage: The pipeline node name where the error surfaced.
        fatal_reason: A human‑readable description of the failure.
        attempts: How many retries were attempted before giving up.
        context: Optional dictionary with extra debugging information.

    Returns:
        A dictionary ready to be serialised to JSON and persisted.
    """
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    return {
        "pipeline": "insights_pipeline",
        "label": label,
        "stage": stage,
        "fatal_reason": fatal_reason,
        "attempts": attempts,
        "timestamp": timestamp,
        "context": context or {},
    }

__all__ = ["PipelineState", "build_fatal_error_object"]