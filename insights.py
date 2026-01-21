"""
Core functions for generating and validating insights and monthly tendencias.

This module wraps calls to the Anthropic models running on AWS Bedrock and
provides utilities for parsing their responses into JSON.  It also includes
helpers for building structured dataframes from summarised conversations and
constructing retry prompts when the model output is invalid.

No stateful logic is kept here; the functions defined below operate on
plain Python objects.  Pipeline nodes should be implemented in
``pipeline_nodes.py`` so that they can be composed in different graphs.
"""

import json
import re
from datetime import datetime
from typing import Dict, Any, List, Tuple

import pandas as pd  # type: ignore

from . import config


# -----------------------------------------------------------------------------
# Model invocation helpers
# -----------------------------------------------------------------------------

def generate_insights(prompt: str) -> str:
    """Invoke the main Bedrock model to generate insights.

    The function sends the user‑supplied prompt to the model specified by
    ``config.MODEL_MAIN`` and extracts the first JSON object from the
    assistant's reply.  The JSON is returned as a pretty‑printed string.  A
    ``ValueError`` is raised if no JSON object can be found in the output.

    Args:
        prompt: A full prompt comprising a schema description and the
            structured input or conversation text.

    Returns:
        A string containing the JSON returned by the model, formatted with
        indentation for readability.
    """
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8000,
        "temperature": 0.3,
    }
    response = config.bedrock.invoke_model(
        modelId=config.MODEL_MAIN,
        body=json.dumps(payload).encode("utf-8"),
    )
    text = response["body"].read().decode("utf-8")
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response.")
    obj = json.loads(match.group(0))
    return json.dumps(obj, ensure_ascii=False, indent=2)


def generate_tendencias(prompt: str) -> str:
    """Invoke the main Bedrock model to generate monthly tendencias JSON.

    This function sends the provided prompt to the same model used for
    generating insights.  It does not perform any validation on the output; the
    caller should parse and validate the returned JSON via
    ``extract_and_validate_tendencias_json``.

    Args:
        prompt: A complete prompt including the schema definition and the
            structured data for a particular month.

    Returns:
        A pretty‑printed JSON string.
    """
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8000,
        "temperature": 0.3,
    }
    response = config.bedrock.invoke_model(
        modelId=config.MODEL_MAIN,
        body=json.dumps(payload).encode("utf-8"),
    )
    text = response["body"].read().decode("utf-8")
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object boundaries found in tendencias response.")
    obj = json.loads(match.group(0))
    return json.dumps(obj, ensure_ascii=False, indent=2)


# -----------------------------------------------------------------------------
# Parsing and validation helpers
# -----------------------------------------------------------------------------

def unwrap_bedrock_text(blob: Any) -> str:
    """Extract the assistant's text from a Bedrock response.

    The Bedrock runtime returns a nested structure containing the assistant's
    reply under ``content[0].text``.  Sometimes the caller may pass
    a stringified JSON object instead of the raw dict.  This function handles
    both cases and returns a plain string.  If the input is already a
    string it is returned unchanged.
    """
    # dict case
    if isinstance(blob, dict):
        content = blob.get("content", [])
        if content and isinstance(content, list):
            return content[0].get("text", "")
    # string case
    if isinstance(blob, str):
        try:
            parsed = json.loads(blob)
            if isinstance(parsed, dict) and "content" in parsed:
                return parsed["content"][0].get("text", "")
        except Exception:
            pass
        return blob
    return ""


def extract_and_validate_json(
    blob: Any, required_keys: List[str]
) -> Tuple[bool, Dict[str, Any] | None, Any]:
    """Extract and validate a JSON object from model output.

    The function attempts to locate the first JSON object within the text
    returned by the model.  It strips any Markdown fences (```json) and
    parses the JSON.  It then checks that all keys in ``required_keys`` are
    present at the top level of the object.  On success ``(True, parsed, None)``
    is returned.  On failure the third element of the tuple contains either a
    string describing the error or a dict with the missing keys.
    """
    try:
        text = unwrap_bedrock_text(blob)
        # remove ```json fences
        text = re.sub(r"```json|```", "", text).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return (False, None, "No JSON object found")
        candidate = text[start : end + 1]
        parsed = json.loads(candidate)
        missing = [k for k in required_keys if k not in parsed]
        if missing:
            return (False, None, {"type": "missing_keys", "missing": missing})
        return (True, parsed, None)
    except json.JSONDecodeError as exc:
        return (False, None, f"JSONDecodeError: {exc}")
    except Exception as exc:
        return (False, None, f"Unexpected error: {exc}")


def extract_and_validate_tendencias_json(
    blob: Any,
) -> Tuple[bool, Dict[str, Any] | None, Any]:
    """Extract and perform basic validation on a tendencias JSON response.

    The returned JSON must contain the top‑level keys ``tendencias_mensuales``
    (a list) and ``insight_global`` (a string).  If validation fails a
    descriptive error is returned.
    """
    try:
        text = unwrap_bedrock_text(blob)
        text = re.sub(r"```json|```", "", text).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return (False, None, "No JSON object boundaries found")
        candidate = text[start : end + 1]
        parsed = json.loads(candidate)
        # check root
        for key in ("tendencias_mensuales", "insight_global"):
            if key not in parsed:
                return (False, None, {"type": "missing_keys", "missing": [key]})
        if not isinstance(parsed.get("tendencias_mensuales"), list):
            return (False, None, "`tendencias_mensuales` must be a list")
        return (True, parsed, None)
    except json.JSONDecodeError as exc:
        return (False, None, f"JSONDecodeError: {exc}")
    except Exception as exc:
        return (False, None, f"Unexpected error: {exc}")


def build_strict_retry_prompt(
    schema_prompt: str, missing_keys: List[str] | None = None
) -> str:
    """Construct a retry prompt emphasising missing keys.

    When the model fails to produce valid JSON, this function can be used
    to build a new prompt that reminds the model of the required schema and
    highlights any keys that were missing in the previous attempt.
    """
    reminder = ""
    if missing_keys:
        lines = ["⚠️ FALTARON LAS SIGUIENTES KEYS OBLIGATORIAS:"]
        for key in missing_keys:
            lines.append(f"- {key}")
        reminder = "\n" + "\n".join(lines) + "\n\n"
    parts = [
        "❗ TU RESPUESTA ANTERIOR NO FUE UN JSON VÁLIDO.\n\n",
        reminder,
        "Debes devolver ÚNICAMENTE un JSON válido.\n",
        "Sin texto adicional, sin comentarios.\n\n",
        "Sigue EXACTAMENTE esta estructura:\n\n",
        schema_prompt,
        "\n\n",
        "Regenera el JSON COMPLETO desde cero.",
    ]
    return "".join(parts)


def build_strict_tendencias_prompt(
    base_schema_prompt: str, missing_keys: List[str] | None = None
) -> str:
    """Construct a retry prompt for tendencias with missing keys.

    This helper mirrors ``build_strict_retry_prompt`` but is tailored to the
    monthly tendencias format.  It reminds the model to return only JSON and
    lists any missing keys detected during validation.
    """
    reminder = ""
    if missing_keys:
        lines = ["⚠️ FALTARON LAS SIGUIENTES KEYS OBLIGATORIAS:"]
        for key in missing_keys:
            lines.append(f"- {key}")
        reminder = "\n" + "\n".join(lines) + "\n\n"
    parts = [
        "❗ TU RESPUESTA ANTERIOR NO FUE UN JSON VÁLIDO.\n\n",
        reminder,
        "Devuelve ÚNICAMENTE un JSON válido.\n",
        "Sin texto adicional.\n\n",
        "Sigue EXACTAMENTE esta estructura:\n\n",
        base_schema_prompt,
        "\n\n",
        "Regenera el JSON COMPLETO desde cero.",
    ]
    return "".join(parts)


# -----------------------------------------------------------------------------
# Dataframe helpers
# -----------------------------------------------------------------------------

def build_insights_dataframe(
    conversaciones: pd.DataFrame,
    df_raw: pd.DataFrame,
    client_col: str = "clientId",
    date_col: str = "createdAt",
    project_col: str = "subProjectInfo",
) -> pd.DataFrame:
    """Create a structured DataFrame from compressed conversation summaries.

    Each row in ``conversaciones`` contains a clientId, a potentially
    structured summary (a dict returned by the compression model) and the
    subProjectInfo if available.  This function merges the summarised data
    with metadata from ``df_raw`` (specifically the first ``createdAt`` and
    ``subProjectInfo`` per client) and expands the summary into individual
    columns.
    """
    # build basic meta: earliest date and project per client
    meta = (
        df_raw.sort_values(date_col)
        .groupby(client_col)
        .agg(
            createdAt=(date_col, "min"),
            subProjectInfo=(project_col, "first"),
        )
        .reset_index()
    )
    # drop duplicated project_col if present in conversaciones
    if project_col in conversaciones.columns:
        conversaciones = conversaciones.drop(columns=[project_col])
    conversaciones = conversaciones.merge(meta, on=client_col, how="left", validate="one_to_one")
    records = []
    for _, row in conversaciones.iterrows():
        summary = row.get("summary")
        # summary is expected to be a dict of compressed fields
        if not isinstance(summary, dict):
            summary = {}
        record = {
            client_col: row[client_col],
            "createdAt": row["createdAt"],
            "subProjectInfo": row["subProjectInfo"],
            # map compressed keys to human friendly names
            "sentimiento": summary.get("sentimiento_promedio", "desconocido"),
            "pain_point": summary.get("pain_point_del_cliente", "desconocido"),
            "perfil_cliente": summary.get("perfil_del_cliente", "desconocido"),
            "situacion_laboral": summary.get("situacion_laboral", "desconocido"),
            "ingreso": summary.get("ingreso", "desconocido"),
            "abandono": summary.get("abandono", "desconocido"),
            "solucion_bot": summary.get("solucion_bot", "desconocido"),
            "topicos": summary.get("topicos_consulta", "desconocido"),
            "atributo_valorado": summary.get("atributo_valorado", "desconocido"),
            "topico_valorado": summary.get("topico_valorado", "desconocido"),
            "etapa_funnel": summary.get("etapa_funnel", "desconocido"),
            "intencion_compra": summary.get("intencion_compra", "desconocido"),
            "capacidad_info": summary.get("capacidad_info", "desconocido"),
            "motivo_abandono": summary.get("motivo_abandono", "desconocido"),
            "tipo_consulta": summary.get("tipo_consulta", "desconocido"),
            "friccion_bot": summary.get("friccion_bot", "desconocido"),
            # cierre is not part of the compress spec; mark as unknown
            "cierre": summary.get("cierre", "desconocido"),
        }
        records.append(record)
    df_out = pd.DataFrame(records)
    df_out["createdAt"] = pd.to_datetime(df_out["createdAt"], errors="coerce")
    return df_out


__all__ = [
    "generate_insights",
    "generate_tendencias",
    "unwrap_bedrock_text",
    "extract_and_validate_json",
    "extract_and_validate_tendencias_json",
    "build_strict_retry_prompt",
    "build_strict_tendencias_prompt",
    "build_insights_dataframe",
]