"""
Pipeline node implementations for the LangGraph insights workflow.

Each function in this module operates on a mutable ``state`` dictionary and
returns an updated state.  Nodes are connected together in ``graph_builder``
using conditional routing based on flags stored in the state.  Functions
defined here are deliberately stateless so that they can be reused across
multiple pipelines.
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd  # type: ignore

from . import config
from . import state
from . import data_loading
from . import preprocessing
from . import summarization
from . import aggregation
from . import prompts
from . import insights


# -----------------------------------------------------------------------------
# Utility router
# -----------------------------------------------------------------------------

def route_after_load(pipeline_state: state.PipelineState) -> str:
    """Decide whether to continue after loading data or skip.

    If the ``skip`` flag has been set in ``pipeline_state`` the pipeline
    terminates early.  Otherwise the next node is executed.
    """
    if pipeline_state.get("skip"):
        return "skip"
    return "continue"


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def node_load_data(pipeline_state: state.PipelineState) -> state.PipelineState:
    """Load all parquet files for the given label into a DataFrame.

    The label is expected to be present in the state under the key
    ``"label"``.  The corresponding prefix is looked up in
    ``config.DATA_SOURCES`` and all parquet files are downloaded and
    concatenated.  If no data is found a skip flag is set so that the
    pipeline can short‑circuit.
    """
    label = pipeline_state["label"]
    prefix = config.DATA_SOURCES[label]
    print(f"\n🔹 Processing dataset: {label.upper()} (prefix={prefix})")
    df = data_loading.load_parquet_folder(prefix)
    pipeline_state["df"] = df
    if df.empty:
        print(f"⚠ No data found for {label}, skipping.")
        pipeline_state["skip"] = True
    else:
        pipeline_state["skip"] = False
    return pipeline_state


# -----------------------------------------------------------------------------
# Preprocessing
# -----------------------------------------------------------------------------

def node_preprocess(pipeline_state: state.PipelineState) -> state.PipelineState:
    """Standardise column names and assemble per‑client conversations.

    This function performs minimal cleaning: converting the ``createdAt``
    column to datetime, trimming whitespace from ``text`` and filtering out
    very short messages.  It then groups messages by client and concatenates
    them into a single conversation string using ``preprocessing.merge_full_conversation``.

    A prompt appropriate to the current label is stored in the state for use
    by downstream nodes.
    """
    label = pipeline_state["label"]
    df = pipeline_state["df"]
    # ensure datetime
    df["createdAt"] = pd.to_datetime(df["createdAt"], errors="coerce")
    # basic text clean
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 1]
    df = df.sort_values(["clientId", "createdAt"])
    # choose prompt based on label
    if label == "recomendador":
        prompt = prompts.PROMPT_RECOMENDADOR
    else:
        prompt = prompts.PROMPT_LIDZ
    # build conversation strings
    conversaciones = (
        df.groupby("clientId")
        .apply(preprocessing.merge_full_conversation)
        .reset_index(name="conversacion")
    )
    # attach project metadata if available
    meta = (
        df.groupby("clientId")
        .agg(subProjectInfo=("subProjectInfo", "first"))
        .reset_index()
    )
    conversaciones = conversaciones.merge(meta, on="clientId", how="left")
    pipeline_state["prompt"] = prompt
    pipeline_state["conversaciones"] = conversaciones
    pipeline_state["df"] = df
    total = conversaciones["clientId"].nunique()
    print(f"🧮 Total unique conversations: {total}")
    return pipeline_state


# -----------------------------------------------------------------------------
# Summarisation
# -----------------------------------------------------------------------------

def node_summarize_conversations(pipeline_state: state.PipelineState) -> state.PipelineState:
    """Compress each conversation into a structured summary.

    Conversations are processed concurrently using a small thread pool to
    maximise throughput.  Each summary is a dictionary with the fields
    defined in ``summarization.COMPRESS_SCHEMA_SPEC``.  The list of
    summaries is attached to the ``conversaciones`` dataframe under the
    ``summary`` column.
    """
    conversaciones: pd.DataFrame = pipeline_state["conversaciones"]
    summaries = []
    # process in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(summarization.compress_with_validation, row["conversacion"]): idx
            for idx, row in conversaciones.iterrows()
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                parsed = future.result()
            except Exception as exc:
                print(f"[ERROR] Compress failed idx={idx}: {exc}")
                parsed = {k: "desconocido" for k in summarization.COMPRESS_SCHEMA_SPEC}
            summaries.append((idx, parsed))
    summaries = sorted(summaries, key=lambda x: x[0])
    conversaciones = conversaciones.copy()
    conversaciones["summary"] = [s for _, s in summaries]
    pipeline_state["conversaciones"] = conversaciones
    print("🧠 Sample structured summary:")
    if not conversaciones.empty:
        print(conversaciones["summary"].iloc[0])
    return pipeline_state


# -----------------------------------------------------------------------------
# Insights generation and validation
# -----------------------------------------------------------------------------

def node_generate_insights(pipeline_state: state.PipelineState) -> state.PipelineState:
    """Create aggregated metrics and call the LLM to produce insights.

    The compressed conversation summaries are expanded into a structured
    dataframe via ``insights.build_insights_dataframe``.  High level
    aggregates are computed using ``aggregation.build_project_aggregates``
    and the dominant client language is extracted with
    ``aggregation.build_lenguaje_cliente_global``.  These aggregated
    features are serialised to JSON and appended to the base prompt.  If
    the state indicates this is a retry, a strict prompt is built with
    missing keys highlighted.
    """
    new_state: state.PipelineState = dict(pipeline_state)
    is_retry = new_state.get("attempts", 0) > 0
    prompt_schema = new_state["prompt"]
    conversaciones: pd.DataFrame = new_state["conversaciones"]
    df_raw: pd.DataFrame = new_state["df"]
    # build structured df
    insights_df = insights.build_insights_dataframe(conversaciones, df_raw)
    new_state["insights_df"] = insights_df
    # global aggregates
    aggregated = aggregation.build_project_aggregates(insights_df)
    lenguaje_cliente = aggregation.build_lenguaje_cliente_global(conversaciones)
    aggregated["lenguaje_cliente"] = lenguaje_cliente
    structured_input = json.dumps(aggregated, ensure_ascii=False, indent=2)
    # assemble final prompt
    if is_retry:
        strict_prompt = insights.build_strict_retry_prompt(
            schema_prompt=prompt_schema,
            missing_keys=new_state.get("missing_keys"),
        )
        final_prompt = strict_prompt + "\n\nDATOS_ESTRUCTURADOS:\n" + structured_input
    else:
        final_prompt = prompt_schema + "\n\nDATOS_ESTRUCTURADOS:\n" + structured_input
    # call LLM
    try:
        insights_json = insights.generate_insights(final_prompt)
    except Exception as exc:
        print(f"[ERROR] generate_insights call failed: {exc}")
        # propagate error: this will be handled by validate_json
        insights_json = "{}"
    new_state["full_text"] = final_prompt
    new_state["insights_json"] = insights_json
    return new_state


def node_validate_json(pipeline_state: state.PipelineState) -> state.PipelineState:
    """Validate the JSON returned by the insights model.

    Required keys differ depending on the current label.  If validation
    succeeds the parsed JSON is re‑serialised with indentation and stored
    back into the state.  Otherwise missing keys are recorded and a retry
    may be triggered.  After ``config.MAX_RETRIES`` attempts the pipeline
    enters a fatal error state.
    """
    new_state: state.PipelineState = dict(pipeline_state)
    attempts = new_state.get("attempts", 0) + 1
    new_state["attempts"] = attempts
    label = new_state["label"]
    # define required top level keys
    if label in ("subsidio", "no_subsidio"):
        required = [
            "resumen_general", "pain_points", "segmentacion_audiencias",
            "insights_producto", "momentos_abandono", "recomendaciones_estrategicas",
            "quick_wins", "kpis_recomendados", "analisis_funnel",
            "topicos_consulta", "conclusiones_y_focos", "oportunidad_principal",
        ]
    else:
        required = [
            "resumen_general", "preferencias_ubicacion", "caracteristicas_proyecto_prioritarias",
            "objeciones_comunes", "percepcion_marca_y_proyectos",
            "segmentos_cliente_detectados", "topicos_consulta", "analisis_funnel",
            "analisis_producto", "conclusiones_y_focos", "recomendaciones_comerciales",
        ]
    raw_output = new_state.get("insights_json", "")
    is_valid, parsed, error = insights.extract_and_validate_json(raw_output, required)
    if is_valid:
        new_state["insights_json"] = json.dumps(parsed, ensure_ascii=False, indent=2)
        new_state["is_valid_json"] = True
        return new_state
    # not valid
    print(f"❌ Insights JSON inválido (attempt {attempts}): {error}")
    new_state["is_valid_json"] = False
    if isinstance(error, dict) and error.get("type") == "missing_keys":
        new_state["missing_keys"] = error.get("missing")
    # fatal condition
    if attempts >= config.MAX_RETRIES:
        fatal_reason = f"Insights JSON inválido tras {config.MAX_RETRIES} intentos"
        err_obj = state.build_fatal_error_object(
            label=label,
            stage="validate_json",
            fatal_reason=fatal_reason,
            attempts=attempts,
            context={"missing_keys": new_state.get("missing_keys")},
        )
        # emit metric to cloudwatch
        config.cloudwatch.put_metric_data(
            Namespace="InsightsPipeline",
            MetricData=[{
                "MetricName": "FatalError",
                "Dimensions": [
                    {"Name": "Label", "Value": label},
                    {"Name": "Stage", "Value": "validate_json"},
                    {"Name": "Reason", "Value": fatal_reason[:255]},
                ],
                "Value": 1,
                "Unit": "Count",
            }],
        )
        new_state["fatal_error"] = True
        new_state["fatal_reason"] = fatal_reason
        new_state["fatal_error_obj"] = err_obj
    return new_state


def node_save_to_s3(pipeline_state: state.PipelineState) -> state.PipelineState:
    """Persist the generated insights JSON to S3.

    The S3 key is derived from the current label and stored under
    ``comercial/insights_graph_test/{label}/insights.json``.  The contents of
    ``insights_json`` are written verbatim.  If the key is missing an
    assertion will trigger.
    """
    assert "label" in pipeline_state, "label is missing in state before saving to S3"
    label = pipeline_state["label"]
    insights_json = pipeline_state["insights_json"]
    output_key = f"comercial/insights_graph_test/{label}/insights.json"
    config.s3.put_object(
        Bucket=config.OUTPUT_BUCKET,
        Key=output_key,
        Body=insights_json.encode("utf-8"),
        ContentType="application/json",
    )
    print(f"✅ Saved → s3://{config.OUTPUT_BUCKET}/{output_key}")
    return pipeline_state


# -----------------------------------------------------------------------------
# Monthly tendencias
# -----------------------------------------------------------------------------

def node_generate_monthly_tendencias(pipeline_state: state.PipelineState) -> state.PipelineState:
    """Generate aggregated monthly tendencias and call the LLM.

    For each month present in the ``insights_df`` dataframe the function
    computes aggregated metrics and language statistics.  These metrics are
    fed into the tendencias prompt and passed to the LLM.  Raw JSON strings
    keyed by month are stored in the state under ``tendencias``.  Missing or
    invalid months are skipped gracefully.
    """
    new_state: state.PipelineState = dict(pipeline_state)
    df = new_state["insights_df"].copy()
    conversaciones = new_state["conversaciones"].copy()
    label = new_state["label"]
    is_retry = new_state.get("tendencias_attempts", 0) > 0
    missing_keys = new_state.get("tendencias_missing_keys")
    # pick base prompt
    if label == "recomendador":
        prompt_base = prompts.PROMPT_TENDENCIAS_RECOMENDADOR
    else:
        prompt_base = prompts.PROMPT_TENDENCIAS_LIDZ
    # build prompt, possibly strict
    if is_retry:
        prompt_tendencias = insights.build_strict_tendencias_prompt(
            base_schema_prompt=prompt_base,
            missing_keys=missing_keys,
        )
    else:
        prompt_tendencias = prompt_base
    # ensure datetime and month grouping
    df["createdAt"] = pd.to_datetime(df["createdAt"], errors="coerce")
    df = df[df["createdAt"].notna()]
    df["mes"] = df["createdAt"].dt.to_period("M").astype(str)
    tendencias_by_month: Dict[str, str] = {}
    for mes, df_mes in df.groupby("mes"):
        print(f"🗓 Procesando mes {mes} ({len(df_mes)} conversaciones)")
        aggregated = aggregation.build_project_aggregates(df_mes)
        # filter conversations for the same clients
        client_ids_mes = set(df_mes["clientId"])
        conv_mes = conversaciones[conversaciones["clientId"].isin(client_ids_mes)]
        lenguaje_cliente = aggregation.build_lenguaje_cliente_global(conv_mes)
        aggregated["lenguaje_cliente"] = lenguaje_cliente
        structured_input = json.dumps(aggregated, ensure_ascii=False, indent=2)
        # inject month instruction
        prompt_with_date = (
            "El mes de estos datos es: " + mes + "\n"
            + "Debes usar EXACTAMENTE este valor en tendencias_mensuales.mes.\n\n"
            + prompt_tendencias
        )
        final_prompt = prompt_with_date + "\n\nDATOS_ESTRUCTURADOS:\n" + structured_input
        try:
            tendencias_json = insights.generate_tendencias(final_prompt)
        except Exception as exc:
            print(f"[ERROR] generate_tendencias failed for mes {mes}: {exc}")
            tendencias_json = "{}"
        tendencias_by_month[mes] = tendencias_json
    new_state["tendencias"] = tendencias_by_month
    return new_state


def node_validate_tendencias_json(pipeline_state: state.PipelineState) -> state.PipelineState:
    """Validate monthly tendencias JSON and accumulate errors.

    Each month is checked for the presence of the required keys within each
    tendencias_mensuales item.  If any month fails validation the entire
    pipeline will retry.  After ``config.MAX_RETRIES`` attempts a fatal
    error is recorded.
    """
    new_state: state.PipelineState = dict(pipeline_state)
    attempts = new_state.get("tendencias_attempts", 0) + 1
    new_state["tendencias_attempts"] = attempts
    label = new_state["label"]
    # define per‑item required keys
    if label in ("subsidio", "no_subsidio"):
        required_item_keys = [
            "mes", "atributos_mas_valorados", "topicos_principales",
            "segmentos_cliente", "afinidad_productos", "funnel",
            "temas_recurrentes", "sentimiento_promedio", "insight_mensual",
            "hallazgos_80_20", "acciones_recomendadas",
        ]
    else:
        required_item_keys = [
            "mes", "ubicaciones_mas_mencionadas", "caracteristicas_mas_valoradas",
            "atributos_mas_valorados", "topicos_principales", "segmentos_cliente",
            "afinidad_productos", "objeciones_principales", "funnel",
            "sentimiento_promedio", "insight_mensual", "hallazgos_80_20",
            "acciones_recomendadas",
        ]
    tendencias_dict: Dict[str, str] = new_state.get("tendencias", {})
    all_valid = True
    validated: Dict[str, str] = {}
    missing_summary = {}
    for mes, raw_json in tendencias_dict.items():
        is_valid, parsed, error = insights.extract_and_validate_tendencias_json(raw_json)
        if not is_valid:
            print(f"❌ Tendencias inválidas {mes} (attempt {attempts}): {error}")
            all_valid = False
            missing_summary[mes] = error
            continue
        # check each item
        for item in parsed.get("tendencias_mensuales", []):
            for k in required_item_keys:
                if k not in item:
                    all_valid = False
                    missing_summary.setdefault(mes, []).append(k)
        validated[mes] = json.dumps(parsed, ensure_ascii=False, indent=2)
    new_state["tendencias"] = validated
    new_state["tendencias_valid"] = all_valid
    # fatal if repeated failure
    if not all_valid and attempts >= config.MAX_RETRIES:
        fatal_reason = f"Tendencias inválidas tras {config.MAX_RETRIES} intentos"
        err_obj = state.build_fatal_error_object(
            label=label,
            stage="validate_tendencias_json",
            fatal_reason=fatal_reason,
            attempts=attempts,
            context={"errors_por_mes": missing_summary},
        )
        config.cloudwatch.put_metric_data(
            Namespace="InsightsPipeline",
            MetricData=[{
                "MetricName": "FatalError",
                "Dimensions": [
                    {"Name": "Label", "Value": label},
                    {"Name": "Stage", "Value": "validate_tendencias_json"},
                    {"Name": "Reason", "Value": fatal_reason[:255]},
                ],
                "Value": 1,
                "Unit": "Count",
            }],
        )
        new_state["fatal_error"] = True
        new_state["fatal_reason"] = fatal_reason
        new_state["fatal_error_obj"] = err_obj
    return new_state


def node_save_tendencias_to_s3(pipeline_state: state.PipelineState) -> state.PipelineState:
    """Persist monthly tendencias JSON blobs to S3.

    Each month is written under ``comercial/insights_tendencias_test/{label}/insights_tendencias_{mes}.json``.
    """
    label = pipeline_state["label"]
    tendencias = pipeline_state.get("tendencias", {})
    for mes, json_text in tendencias.items():
        key = f"comercial/insights_tendencias_test/{label}/insights_tendencias_{mes}.json"
        config.s3.put_object(
            Bucket=config.OUTPUT_BUCKET,
            Key=key,
            Body=json_text.encode("utf-8"),
            ContentType="application/json",
        )
        print(f"✅ Saved tendencias → s3://{config.OUTPUT_BUCKET}/{key}")
    return pipeline_state


# -----------------------------------------------------------------------------
# Subproject insights
# -----------------------------------------------------------------------------

def node_generate_insights_by_subproject(pipeline_state: state.PipelineState) -> state.PipelineState:
    """Generate insights per subProjectInfo using aggregated data.

    Only projects with a non‑empty ``subProjectInfo`` field are processed.
    Aggregated metrics and language statistics are computed for each project
    separately and fed into the appropriate project prompt.  The results
    are stored under ``insights_by_subproject`` keyed by the project name.
    """
    new_state: state.PipelineState = dict(pipeline_state)
    conversaciones = new_state["conversaciones"].copy()
    df = new_state["insights_df"].copy()
    label = new_state["label"]
    # choose base prompt
    if label == "recomendador":
        base_prompt = prompts.PROMPT_RECOMENDADOR_PROYECTO
    else:
        base_prompt = prompts.PROMPT_LIDZ_PROYECTO
    # filter valid projects
    df = df[
        df["subProjectInfo"].notna()
        & (df["subProjectInfo"].astype(str).str.strip() != "")
        & (~df["subProjectInfo"].astype(str).str.lower().isin([
            "sin proyecto", "none", "null", "nan"
        ]))
    ]
    insights_by_project: Dict[str, str] = {}
    for project, df_proj in df.groupby("subProjectInfo"):
        project_name = str(project).strip()
        print(f"📦 Project: {project_name} ({len(df_proj)} conversaciones)")
        aggregated = aggregation.build_project_aggregates(df_proj)
        conv_proj = conversaciones[conversaciones["subProjectInfo"] == project]
        lenguaje_cliente = aggregation.build_lenguaje_cliente_global(conv_proj)
        aggregated["lenguaje_cliente"] = lenguaje_cliente
        structured_input = json.dumps(aggregated, ensure_ascii=False, indent=2)
        prompt_con_proyecto = (
            "El nombre del proyecto es: " + project_name + "\n"
            + "Debes usar EXACTAMENTE este valor en resumen_general.nombre_proyecto.\n\n"
            + base_prompt
        )
        final_prompt = (
            prompt_con_proyecto
            + "\n\nDATOS_ESTRUCTURADOS:\n"
            + structured_input
        )
        try:
            insights_json = insights.generate_insights(final_prompt)
        except Exception as exc:
            print(f"[ERROR] generate_insights failed for project {project_name}: {exc}")
            insights_json = "{}"
        insights_by_project[project_name] = insights_json
    new_state["insights_by_subproject"] = insights_by_project
    return new_state


def node_validate_subproject_insights(pipeline_state: state.PipelineState) -> state.PipelineState:
    """Validate each per‑project insights JSON.

    In addition to checking required keys this function ensures that the
    project name specified in ``resumen_general.nombre_proyecto`` matches
    the key used to store the result.  On repeated failure the pipeline
    transitions to a fatal error state.
    """
    new_state: state.PipelineState = dict(pipeline_state)
    attempts = new_state.get("subproject_attempts", 0) + 1
    new_state["subproject_attempts"] = attempts
    label = new_state["label"]
    if label in ("subsidio", "no_subsidio"):
        required = [
            "resumen_general", "pain_points", "segmentacion_audiencias",
            "insights_producto", "momentos_abandono", "recomendaciones_estrategicas",
            "quick_wins", "kpis_recomendados", "analisis_funnel", "topicos_consulta",
            "conclusiones_y_focos", "oportunidad_principal",
        ]
    else:
        required = [
            "resumen_general", "preferencias_ubicacion", "caracteristicas_proyecto_prioritarias",
            "objeciones_comunes", "percepcion_marca_y_proyectos", "segmentos_cliente_detectados",
            "topicos_consulta", "analisis_funnel", "analisis_producto", "conclusiones_y_focos",
            "recomendaciones_comerciales",
        ]
    validated: Dict[str, str] = {}
    all_valid = True
    error_summary = {}
    for project, raw_json in new_state.get("insights_by_subproject", {}).items():
        is_valid, parsed, error = insights.extract_and_validate_json(raw_json, required)
        if not is_valid:
            print(f"❌ Invalid JSON for project '{project}': {error}")
            all_valid = False
            error_summary[project] = error
            continue
        expected = str(project).strip()
        actual = parsed.get("resumen_general", {}).get("nombre_proyecto")
        if actual != expected:
            print(
                f"❌ nombre_proyecto mismatch for {project}: expected='{expected}' got='{actual}'"
            )
            all_valid = False
            error_summary[project] = "nombre_proyecto mismatch"
            continue
        validated[project] = json.dumps(parsed, ensure_ascii=False, indent=2)
    new_state["insights_by_subproject"] = validated
    new_state["subproject_valid"] = all_valid
    if not all_valid and attempts >= config.MAX_RETRIES:
        fatal_reason = f"Subproject insights inválidos tras {config.MAX_RETRIES} intentos"
        err_obj = state.build_fatal_error_object(
            label=label,
            stage="validate_subproject_insights",
            fatal_reason=fatal_reason,
            attempts=attempts,
            context={"errores_por_proyecto": error_summary},
        )
        config.cloudwatch.put_metric_data(
            Namespace="InsightsPipeline",
            MetricData=[{
                "MetricName": "FatalError",
                "Dimensions": [
                    {"Name": "Label", "Value": label},
                    {"Name": "Stage", "Value": "validate_subproject_insights"},
                    {"Name": "Reason", "Value": fatal_reason[:255]},
                ],
                "Value": 1,
                "Unit": "Count",
            }],
        )
        new_state["fatal_error"] = True
        new_state["fatal_reason"] = fatal_reason
        new_state["fatal_error_obj"] = err_obj
    return new_state


def node_save_subproject_insights_to_s3(pipeline_state: state.PipelineState) -> state.PipelineState:
    """Persist per‑project insights to S3.

    Each project is saved under
    ``comercial/insights_by_project_test/{label}/{safe_project}/insights.json``.
    Spaces in the project name are replaced with underscores and converted
    to lower case to form a safe key.
    """
    label = pipeline_state["label"]
    insights_by_project = pipeline_state.get("insights_by_subproject", {})
    for project, json_text in insights_by_project.items():
        safe_project = project.replace(" ", "_").lower()
        key = (
            f"comercial/insights_by_project_test/{label}/{safe_project}/insights.json"
        )
        config.s3.put_object(
            Bucket=config.OUTPUT_BUCKET,
            Key=key,
            Body=json_text.encode("utf-8"),
            ContentType="application/json",
        )
        print(f"✅ Saved → s3://{config.OUTPUT_BUCKET}/{key}")
    return pipeline_state


# -----------------------------------------------------------------------------
# Fatal error handling
# -----------------------------------------------------------------------------

def node_save_fatal_error_to_s3(pipeline_state: state.PipelineState) -> state.PipelineState:
    """Serialise the fatal error object and persist it to S3.

    If the state does not contain a fatal error object the function is a no‑op.
    """
    error_obj = pipeline_state.get("fatal_error_obj")
    if not error_obj:
        return pipeline_state
    label = pipeline_state["label"]
    ts = error_obj["timestamp"].replace(":", "").replace(".", "")
    key = f"comercial/insights_errors/{label}/fatal_error_{ts}.json"
    config.s3.put_object(
        Bucket=config.OUTPUT_BUCKET,
        Key=key,
        Body=json.dumps(error_obj, ensure_ascii=False, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    print(f"🛑 Fatal error saved → s3://{config.OUTPUT_BUCKET}/{key}")
    return pipeline_state


__all__ = [
    "route_after_load",
    "node_load_data",
    "node_preprocess",
    "node_summarize_conversations",
    "node_generate_insights",
    "node_validate_json",
    "node_save_to_s3",
    "node_generate_monthly_tendencias",
    "node_validate_tendencias_json",
    "node_save_tendencias_to_s3",
    "node_generate_insights_by_subproject",
    "node_validate_subproject_insights",
    "node_save_subproject_insights_to_s3",
    "node_save_fatal_error_to_s3",
]