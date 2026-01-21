"""Aggregation and analytical utilities."""

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from .preprocessing import SPANISH_STOPWORDS


def safe_tfidf_vectorizer(n_docs):
    """Return a TF–IDF vectorizer configured based on the number of documents.

    For small corpora fewer features and a lower min_df are used.  This helper
    prevents high dimensionality when only a handful of conversations are
    available.
    """
    if n_docs < 5:
        return TfidfVectorizer(
            max_features=100,
            ngram_range=(1, 2),
            min_df=1,
            stop_words=SPANISH_STOPWORDS,
        )
    return TfidfVectorizer(
        max_features=300,
        ngram_range=(1, 2),
        min_df=2,
        stop_words=SPANISH_STOPWORDS,
    )


def build_lenguaje_cliente_global(conversaciones, top_k=15):
    """Extract the dominant terms across all conversations using TF–IDF.

    Args:
        conversaciones: A dataframe with a ``conversacion`` column containing
            concatenated conversation texts per client.
        top_k: The number of top terms to return.

    Returns:
        A dictionary mapping dominant terms to their average TF–IDF score.
    """
    textos = (
        conversaciones["conversacion"]
        .dropna()
        .astype(str)
        .tolist()
    )
    n_docs = len(textos)
    if n_docs < 2:
        return {}
    vectorizer = safe_tfidf_vectorizer(n_docs)
    X = vectorizer.fit_transform(conversaciones["conversacion"])
    terms = vectorizer.get_feature_names_out()
    scores = np.asarray(X.mean(axis=0)).ravel()
    df_terms = (
        pd.DataFrame({"term": terms, "score": scores})
        .sort_values("score", ascending=False)
        .head(top_k)
    )
    return {
        "terminos_dominantes": {
            row.term: round(row.score, 3) for _, row in df_terms.iterrows()
        }
    }


def build_project_aggregates(df):
    """Compute aggregated metrics from structured conversation-level insights.

    The returned dictionary contains high level KPIs segmented by sentiment,
    funnel progression, pain points, client profile and bot performance.
    The schema is intentionally flat to make serialisation to JSON trivial.
    """

    # Helper functions for percentages and top-k frequencies
    def pct(series, value):
        if len(series) == 0:
            return 0.0
        return round((series == value).mean(), 3)

    def top_k(series, k=15):
        cnts = (
            series
            .value_counts()
            .drop(labels=["desconocido"], errors="ignore")
            .head(k)
            .to_dict()
        )
        return cnts

    total = df["clientId"].nunique()

    aggregates = {
        "total_conversaciones": total,
        "sentimiento": {
            "positivo": pct(df["sentimiento"], "positivo"),
            "neutral": pct(df["sentimiento"], "neutral"),
            "negativo": pct(df["sentimiento"], "negativo"),
        },
        "funnel": {
            "abandono_temprano_pct": pct(df["abandono"], "si"),
            "cierre_conversacion_pct": pct(df["cierre"], "si"),
            "etapas_funnel": top_k(df["etapa_funnel"], k=5),
            "intencion_compra": top_k(df["intencion_compra"], k=5),
        },
        "pain_points": top_k(df["pain_point"], k=7),
        "motivos_abandono": top_k(
            df.loc[df["abandono"] == "si", "motivo_abandono"],
            k=5,
        ),
        "segmentos_cliente": top_k(df["perfil_cliente"], k=5),
        "situacion_laboral": top_k(df["situacion_laboral"], k=5),
        "ingreso": top_k(df["ingreso"], k=5),
        "capacidad_entrega_informacion": top_k(df["capacidad_info"], k=3),
        "topicos_consulta": top_k(df["topicos"], k=7),
        "tipo_consulta": top_k(df["tipo_consulta"], k=5),
        "atributos_valorados": top_k(df["atributo_valorado"], k=7),
        "topicos_valorados": top_k(df["topico_valorado"], k=7),
        "soluciones_bot_mas_usadas": top_k(df["solucion_bot"], k=5),
        "fricciones_bot": top_k(df["friccion_bot"], k=5),
    }
    return aggregates
