"""Utilities for compressing conversations and validating compressed output."""

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import config

# Schema for the compressed summary.  Fields with a ``None`` value accept any
# token; those with a set restrict the allowed values.
COMPRESS_SCHEMA_SPEC = {
    "sentimiento_promedio": {"positivo", "neutral", "negativo"},
    "pain_point_del_cliente": None,
    "perfil_del_cliente": None,
    "situacion_laboral": None,
    "ingreso": None,
    "abandono": {"si", "no"},
    "motivo_abandono": {
        "precio", "subsidio", "requisitos", "desconfianza", "silencio", "otro", "none"
    },
    "solucion_bot": None,
    "friccion_bot": {"baja", "media", "alta"},
    "topicos_consulta": None,
    "tipo_consulta": {
        "informativa", "comparativa", "financiera", "documental", "cierre"
    },
    "atributo_valorado": None,
    "topico_valorado": None,
    "etapa_funnel": {
        "descubrimiento", "evaluacion", "decision", "cierre", "abandono"
    },
    "intencion_compra": {"baja", "media", "alta"},
    "capacidad_info": {"alta", "media", "baja"},
}


def parse_compress_output(text):
    """Parse the raw text returned by the compress model into a dictionary.

    The compression model returns key–value pairs separated by commas or
    newlines.  Keys and values are separated by a colon or equals sign.
    Unknown keys are ignored.  All keys in ``COMPRESS_SCHEMA_SPEC`` are
    initialised to the string ``"desconocido"``.
    """
    parsed = {k: "desconocido" for k in COMPRESS_SCHEMA_SPEC}
    if not isinstance(text, str):
        return parsed
    lines = re.split(r"[,\n]", text)
    for line in lines:
        line = line.strip()
        if not line:
            continue
        m = re.match(r"([^:=]+)\s*[:=]\s*(.+)", line)
        if not m:
            continue
        key, value = m.groups()
        key = key.strip()
        value = value.strip().lower()
        if key in parsed:
            parsed[key] = value
    return parsed


def validate_compress_schema(parsed):
    """Validate the parsed compressed summary against ``COMPRESS_SCHEMA_SPEC``.

    Returns a tuple ``(is_valid, errors_dict)`` where ``errors_dict`` maps
    field names to error messages.  If no errors are found ``is_valid`` will
    be True and ``errors_dict`` will be empty.
    """
    errors = {}
    for field, allowed in COMPRESS_SCHEMA_SPEC.items():
        value = parsed.get(field)
        if value is None or value == "":
            errors[field] = "missing"
            continue
        if allowed and value not in allowed and value != "desconocido":
            errors[field] = "invalid_value: " + value
    return (len(errors) == 0), errors


def build_compress_repair_prompt(original_text, previous_output, errors):
    """Construct a repair prompt instructing the model to fix invalid fields.

    The prompt lists the fields that failed validation and reminds the model
    about the required output format.  Curly braces in the schema are
    deliberately doubled to avoid formatting in f-strings.
    """
    error_list = "\n".join(f"- {k}: {v}" for k, v in errors.items())
    parts = [
        "La salida anterior NO cumple el formato requerido.\n",
        "\nErrores detectados:\n",
        error_list,
        "\n\nDevuelve EXACTAMENTE una sola línea con TODOS los campos\n",
        "y SOLO los valores permitidos.\n\n",
        "Formateo obligatorio (una palabra por campo):\n\n",
        "sentimiento_promedio: positivo | neutral | negativo,\n",
        "pain_point_del_cliente: palabra,\n",
        "perfil_del_cliente: palabra,\n",
        "situacion_laboral: palabra,\n",
        "ingreso: palabra,\n",
        "abandono=si|no,\n",
        "motivo_abandono=precio|subsidio|requisitos|desconfianza|silencio|otro|none,\n",
        "solucion_bot=palabra,\n",
        "friccion_bot=baja|media|alta,\n",
        "topicos_consulta=palabra,\n",
        "tipo_consulta=informativa|comparativa|financiera|documental|cierre,\n",
        "atributo_valorado=palabra,\n",
        "topico_valorado=palabra,\n",
        "etapa_funnel=descubrimiento|evaluacion|decision|cierre|abandono,\n",
        "intencion_compra=baja|media|alta,\n",
        "capacidad_info=alta|media|baja\n\n",
        "Reglas:\n",
        "- Usa SOLO una palabra por campo\n",
        "- Si no hay info clara, usa \"desconocido\"\n",
        "- No agregues texto adicional\n\n",
        "Conversación:\n",
        original_text,
        "\n\nSalida previa (incorrecta):\n",
        previous_output,
    ]
    return "".join(parts)


def compress(text):
    """Call the Bedrock compression model to summarise a conversation.

    A small payload is constructed using the Anthropic message format and
    the configured compression model.  The assistant's reply is returned
    as raw text.
    """
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Analiza la siguiente conversación COMPLETA entre cliente y bot.\n"
                    + "Devuelve SOLO una línea en el formato especificado.\n\n"
                    + text
                ),
            }
        ],
        "max_tokens": 200,
        "temperature": 0.2,
    }
    response = config.bedrock.invoke_model(
        modelId=config.MODEL_COMPRESS,
        body=json.dumps(payload).encode("utf-8"),
    )
    body_text = response["body"].read().decode("utf-8")
    data = json.loads(body_text)
    return data["content"][0]["text"]


def compress_with_validation(text):
    """Compress a conversation with schema validation and retries.

    The compress model is called up to ``config.MAX_RETRIES`` times.  After
    each attempt the output is parsed and validated.  If validation fails a
    repair prompt is generated and used as the input for the next attempt.
    The function returns a parsed dictionary of the compressed fields.  If
    all attempts fail a dictionary with all fields set to "desconocido" is
    returned.
    """
    last_output = None
    errors = None
    for attempt in range(1, config.MAX_RETRIES + 1):
        if attempt == 1:
            raw_output = compress(text)
        else:
            repair_prompt = build_compress_repair_prompt(
                original_text=text,
                previous_output=last_output,
                errors=errors,
            )
            raw_output = compress(repair_prompt)
        parsed = parse_compress_output(raw_output)
        is_valid, errors = validate_compress_schema(parsed)
        if is_valid:
            return parsed
        last_output = raw_output
    # fallback
    return {k: "desconocido" for k in COMPRESS_SCHEMA_SPEC}
