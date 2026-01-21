"""
Centralised prompt definitions for the insights pipeline.

These multi‑line strings define the schemas that are passed to the LLM when
generating insights or monthly tendencias.  Keeping them in a separate
module makes it easier to maintain and update the formats without touching
the rest of the code base.
"""

# pylint: disable=pointless-string-statement

# ---------------------------------------------------------------------------
# Main insight prompts
# ---------------------------------------------------------------------------

PROMPT_LIDZ = """
Analiza los siguientes mensajes y devuelve solo JSON válido:

El JSON debe seguir esta estructura exacta (sin texto adicional):

{
  "resumen_general": {
    "total_conversaciones": 0,
    "tasa_abandono_temprano": 0.0,
    "tasa_conversion_lead": 0.0,
    "porcentaje_sin_pie": 0.0,
    "sentimiento_promedio": "positivo | neutral | negativo",
    "insight_principal": "string"
  },
  "pain_points": [
    {
      "titulo": "string",
      "descripcion": "string",
      "frecuencia": 0.0,
      "impacto": "alto | medio | bajo",
      "ejemplos_clientes": ["string"],
      "recomendaciones": ["string"]
    }
  ],
  "segmentacion_audiencias": [
    {
      "segmento": "string",
      "porcentaje": 0.0,
      "perfil": "string",
      "pain_principal": "string",
      "consulta_tipica": "string",
      "solucion_sugerida": "string",

      "caracteristicas_clientes": {
        "situacion_laboral": "string",
        "nivel_ingreso_estimado": "string",
        "capacidad_entrega_informacion": "alta | media | baja",
        "dudas_recurrentes": ["string"],
        "probabilidad_conversion": 0.0
      }
    }
  ],
  "insights_producto": {
    "proyecto_mas_consultado": "string",
    "caracteristicas_valoradas": [
      {"caracteristica": "string", "porcentaje": 0.0, "ranking": 1}
    ],
    "demandas_no_cubiertas": ["string"],
    "afinidad_productos": [
      {
        "producto_origen": "string",
        "producto_destino": "string",
        "porcentaje_movimiento": 0.0,
        "motivo_derivacion": "string"
      }
    ],
    "informacion_complementaria_requerida": [
      "ggcc", "referencias_geograficas", "conectividad", "documentacion_requerida"
    ]
},
  "momentos_abandono": [
    {
      "nombre": "string",
      "porcentaje_abandono": 0.0,
      "descripcion": "string",
      "frase_tipica": "string",
      "recomendacion": "string"
    }
  ],
  "recomendaciones_estrategicas": {
    "inmediatas": ["string"],
    "corto_plazo": ["string"],
    "mediano_plazo": ["string"]
  },
  "quick_wins": ["string"],
  "kpis_recomendados": [
    "string"
  ],
    "analisis_funnel": {
    "tasa_fuga": 0.0,
    "motivos_fuga": ["string"],
    "conversion_a_contacto_vendedor": 0.0,
    "tasa_respuesta_recomendador": 0.0
  },

  "topicos_consulta": {
    "producto": ["string"],
    "acceso": ["string"],
    "financiamiento": ["string"],
    "subsidios": ["string"]
  },
    "conclusiones_y_focos": {
    "principales_hallazgos_80_20": ["string"],
    "problematica_central": "string",
    "factores_relevantes": ["falta_informacion", "capacidad_respuesta", "oferta", "acceso"],
    "acciones_recomendadas": ["string"]
  },
  "oportunidad_principal": "string"
}
"""


PROMPT_LIDZ_PROYECTO = """
Analiza los siguientes mensajes y devuelve solo JSON válido:
El JSON debe seguir esta estructura exacta (sin texto adicional):

{
  "resumen_general": {
    "nombre_proyecto": "string"
    "total_conversaciones": 0,
    "tasa_abandono_temprano": 0.0,
    "tasa_conversion_lead": 0.0,
    "porcentaje_sin_pie": 0.0,
    "sentimiento_promedio": "positivo | neutral | negativo",
    "insight_principal": "string"
  },
  "pain_points": [
    {
      "titulo": "string",
      "descripcion": "string",
      "frecuencia": 0.0,
      "impacto": "alto | medio | bajo",
      "ejemplos_clientes": ["string"],
      "recomendaciones": ["string"]
    }
  ],
  "segmentacion_audiencias": [
    {
      "segmento": "string",
      "porcentaje": 0.0,
      "perfil": "string",
      "pain_principal": "string",
      "consulta_tipica": "string",
      "solucion_sugerida": "string",

      "caracteristicas_clientes": {
        "situacion_laboral": "string",
        "nivel_ingreso_estimado": "string",
        "capacidad_entrega_informacion": "alta | media | baja",
        "dudas_recurrentes": ["string"],
        "probabilidad_conversion": 0.0
      }
    }
  ],
  "insights_producto": {
    "proyecto_mas_consultado": "string",
    "caracteristicas_valoradas": [
      {"caracteristica": "string", "porcentaje": 0.0, "ranking": 1}
    ],
    "demandas_no_cubiertas": ["string"],
  "afinidad_productos": [
      {
        "producto_origen": "string",
        "producto_destino": "string",
        "porcentaje_movimiento": 0.0,
        "motivo_derivacion": "string"
      }
    ],
    "informacion_complementaria_requerida": [
      "ggcc", "referencias_geograficas", "conectividad", "documentacion_requerida"
    ]
},
  "momentos_abandono": [
    {
      "nombre": "string",
      "porcentaje_abandono": 0.0,
      "descripcion": "string",
      "frase_tipica": "string",
      "recomendacion": "string"
    }
  ],
  "recomendaciones_estrategicas": {
    "inmediatas": ["string"],
    "corto_plazo": ["string"],
    "mediano_plazo": ["string"]
  },
  "quick_wins": ["string"],
  "kpis_recomendados": [
    "string"
  ],
    "analisis_funnel": {
    "tasa_fuga": 0.0,
    "motivos_fuga": ["string"],
    "conversion_a_contacto_vendedor": 0.0,
    "tasa_respuesta_recomendador": 0.0
  },

  "topicos_consulta": {
    "producto": ["string"],
    "acceso": ["string"],
    "financiamiento": ["string"],
    "subsidios": ["string"]
  },
    "conclusiones_y_focos": {
    "principales_hallazgos_80_20": ["string"],
    "problematica_central": "string",
    "factores_relevantes": ["falta_informacion", "capacidad_respuesta", "oferta", "acceso"],
    "acciones_recomendadas": ["string"]
  },
  "oportunidad_principal": "string"
}
"""


# ---------------------------------------------------------------------------
# Insight prompts for the recomendador use case
# ---------------------------------------------------------------------------

PROMPT_RECOMENDADOR = """
Eres un analista de mercado inmobiliario. Analiza las conversaciones entre clientes y un recomendador automático de proyectos.

Tu objetivo es identificar:
- Preferencias de ubicación (comunas, conectividad, cercanía a metro)
- Características del proyecto que los clientes valoran (dormitorios, áreas comunes, estacionamiento, entrega)
- Objeciones al proyecto o zona (precio, distancia, stock, tamaño)
- Señales de intención de compra

Devuelve SOLO un JSON válido con el formato:
{
  "resumen_general": {
    "total_conversaciones": 0,
    "niveles_intencion_compra": {"exploracion": 0.0, "comparacion": 0.0, "decision": 0.0},
    "insight_principal": "string"
  },
  "preferencias_ubicacion": {"bla": "string"},
  "caracteristicas_proyecto_prioritarias": [
      {"caracteristica": "string", "relevancia": 0.0}
  ],
  "objeciones_comunes": [ 
      {"objecion": "string", "frecuencia": 0.0, "tipo": "precio | ubicacion | tamaño | stock | acceso"}
  ],
  "percepcion_marca_y_proyectos": {"bla": "string"},
  "segmentos_cliente_detectados": [{
      "segmento": "string",
      "perfil_cliente": "string",
      "capacidad_entrega_informacion": "alta | media | baja",
      "dudas_principales": "string",
      "probabilidad_conversion": 0.0,
      "consulta_tipica": "string"
  }
  ],
  "topicos_consulta": {
    "producto": ["string"],
    "acceso": ["string"],
    "financiamiento": ["string"],
    "preferencias_proyecto": ["string"]
  },
  "analisis_funnel": {
    "tasa_fuga": 0.0,
    "motivos_fuga": ["string"],
    "conversion_a_contacto_vendedor": 0.0,
    "tasa_respuesta_recomendador": 0.0
  },
  "analisis_producto": {
    "dudas_generales": ["string"],
    "informacion_complementaria": ["ggcc", "conectividad", "referencias_geograficas", "entrega"],
    "afinidad_productos": [
      {
        "producto_origen": "string",
        "producto_destino": "string",
        "porcentaje_movimiento": 0.0
      }
    ]
  },
  "conclusiones_y_focos": {
    "hallazgos_80_20": ["string"],
    "problema_central": "string",
    "acciones_sugeridas": ["string"]
  },
  "recomendaciones_comerciales": {
    "inmediatas": ["string"],
    "mediano_plazo": ["string"]
  }
}
"""


PROMPT_RECOMENDADOR_PROYECTO = """
Eres un analista de mercado inmobiliario. Analiza las conversaciones entre clientes y un recomendador automático de proyectos.
El nombre del proyecto es: {project_name}
Debes usar exactamente ese valor en resumen_general.nombre_proyecto.

Tu objetivo es identificar:
- Preferencias de ubicación (comunas, conectividad, cercanía a metro)
- Características del proyecto que los clientes valoran (dormitorios, áreas comunes, estacionamiento, entrega)
- Objeciones al proyecto o zona (precio, distancia, stock, tamaño)
- Señales de intención de compra

Devuelve SOLO un JSON válido con el formato:
{
  "resumen_general": {
    "nombre proyecto" : "string"
    "total_conversaciones": 0,
    "niveles_intencion_compra": {"exploracion": 0.0, "comparacion": 0.0, "decision": 0.0},
    "insight_principal": "string"
  },
  "preferencias_ubicacion": {"bla": "string"},
  "caracteristicas_proyecto_prioritarias": [
      {"caracteristica": "string", "relevancia": 0.0}
  ],
  "objeciones_comunes": [ 
      {"objecion": "string", "frecuencia": 0.0, "tipo": "precio | ubicacion | tamaño | stock | acceso"}
  ],
  "percepcion_marca_y_proyectos": {"bla": "string"},
  "segmentos_cliente_detectados": [{
      "segmento": "string",
      "perfil_cliente": "string",
      "capacidad_entrega_informacion": "alta | media | baja",
      "dudas_principales": "string",
      "probabilidad_conversion": 0.0,
      "consulta_tipica": "string"
  }
  ],
  "topicos_consulta": {
    "producto": ["string"],
    "acceso": ["string"],
    "financiamiento": ["string"],
    "preferencias_proyecto": ["string"]
  },
  "analisis_funnel": {
    "tasa_fuga": 0.0,
    "motivos_fuga": ["string"],
    "conversion_a_contacto_vendedor": 0.0,
    "tasa_respuesta_recomendador": 0.0
  },
  "analisis_producto": {
    "dudas_generales": ["string"],
    "informacion_complementaria": ["ggcc", "conectividad", "referencias_geograficas", "entrega"],
    "afinidad_productos": [
      {
        "producto_origen": "string",
        "producto_destino": "string",
        "porcentaje_movimiento": 0.0
      }
    ]
  },
  "conclusiones_y_focos": {
    "hallazgos_80_20": ["string"],
    "problema_central": "string",
    "acciones_sugeridas": ["string"]
  },
  "recomendaciones_comerciales": {
    "inmediatas": ["string"],
    "mediano_plazo": ["string"]
  }
}
"""


# ---------------------------------------------------------------------------
# Monthly tendencias prompts
# ---------------------------------------------------------------------------

PROMPT_TENDENCIAS_LIDZ = """
Eres un analista de datos inmobiliarios. Analiza los mensajes de clientes sobre subsidios habitacionales.
Identifica las tendencias mensuales de los temas y atributos más valorados o mencionados.

Devuelve EXCLUSIVAMENTE un JSON válido con este formato:

{
  "tendencias_mensuales": [
    {
      "mes": "YYYY-MM",

      "atributos_mas_valorados": [
        {"atributo": "string", "menciones": 0, "variacion_pct": 0.0}
      ],

      "topicos_principales": {
        "producto": ["string"],
        "acceso": ["string"],
        "financiamiento": ["string"],
        "subsidios": ["string"]
      },

      "segmentos_cliente": [
        {
          "segmento": "string",
          "tendencia": "alza | baja | estable",
          "cambio_pct": 0.0,
          "caracteristicas_predominantes": ["string"]
        }
      ],

      "afinidad_productos": [
        {
          "producto_origen": "string",
          "producto_destino": "string",
          "variacion_movimiento_pct": 0.0
        }
      ],

      "funnel": {
        "tasa_fuga": 0.0,
        "variacion_fuga_pct": 0.0,
        "motivos_fuga_principales": ["string"],
        "conversion_a_contacto_vendedor": 0.0,
        "variacion_conversion_pct": 0.0
      },

      "temas_recurrentes": ["string"],
      "sentimiento_promedio": "positivo | neutral | negativo",
      "insight_mensual": "string",

      "hallazgos_80_20": ["string"],
      "acciones_recomendadas": ["string"]
    }
  ],

  "insight_global": "string con observaciones generales sobre cambios de preferencias"
}
"""


PROMPT_TENDENCIAS_RECOMENDADOR = """
Eres un analista de mercado inmobiliario. Analiza las conversaciones entre clientes y el recomendador automático.
Identifica las tendencias mensuales en preferencias, objeciones, afinidad entre proyectos y comportamiento del funnel.

Devuelve EXCLUSIVAMENTE un JSON válido con este formato:

{
  "tendencias_mensuales": [
    {
      "mes": "YYYY-MM",

      "ubicaciones_mas_mencionadas": [
        {"comuna": "string", "menciones": 0, "variacion_pct": 0.0}
      ],

      "caracteristicas_mas_valoradas": [
        {"caracteristica": "string", "menciones": 0, "variacion_pct": 0.0}
      ],

      "atributos_mas_valorados": [
        {"atributo": "string", "menciones": 0, "variacion_pct": 0.0}
      ],

      "topicos_principales": {
        "producto": ["string"],
        "acceso": ["string"],
        "financiamiento": ["string"],
        "preferencias_proyecto": ["string"]
      },

      "segmentos_cliente": [
        {
          "segmento": "string",
          "tendencia": "alza | baja | estable",
          "cambio_pct": 0.0,
          "dudas_recurrentes": ["string"],
          "probabilidad_conversion_promedio": 0.0
        }
      ],

      "afinidad_productos": [
        {
          "producto_origen": "string",
          "producto_destino": "string",
          "variacion_movimiento_pct": 0.0
        }
      ],

      "objeciones_principales": [
        {"tipo": "string", "frecuencia": 0.0, "variacion_pct": 0.0}
      ],

      "funnel": {
        "tasa_fuga": 0.0,
        "variacion_fuga_pct": 0.0,
        "motivos_fuga": ["string"],
        "conversion_a_contacto_vendedor": 0.0,
        "variacion_conversion_pct": 0.0
      },

      "sentimiento_promedio": "positivo | neutral | negativo",
      "insight_mensual": "string",

      "hallazgos_80_20": ["string"],
      "acciones_recomendadas": ["string"]
    }
  ],

  "insight_global": "string sobre cómo evolucionan las preferencias a lo largo de los meses"
}
"""


# ---------------------------------------------------------------------------
# Retry prompt for invalid JSON responses
# ---------------------------------------------------------------------------

STRICT_RETRY_PROMPT = """
❗ Tu respuesta anterior NO fue un JSON válido.

Debes devolver únicamente un JSON VÁLIDO.
Sin explicaciones, sin texto adicional, sin comentarios y sin comillas incorrectas.

Sigue EXACTAMENTE la siguiente estructura JSON:

{schema}

Regenera el JSON completo desde cero si es necesario.
"""
