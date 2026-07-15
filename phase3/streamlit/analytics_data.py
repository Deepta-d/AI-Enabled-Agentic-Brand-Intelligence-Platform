"""Analytics dashboard data helpers (cached MySQL / metrics)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from phase3.services import mysql_service
from phase3.services.ml_service import get_best_metrics, get_model_comparison


@st.cache_data(ttl=300, show_spinner=False)
def load_sentiment_rows() -> list[dict[str, Any]]:
    result = mysql_service.sentiment_summary()
    if not result.get("ok"):
        return []
    return list(result.get("rows") or [])


@st.cache_data(ttl=300, show_spinner=False)
def load_platform_rows() -> list[dict[str, Any]]:
    result = mysql_service.run_sql(
        """
        SELECT COALESCE(NULLIF(TRIM(platform), ''), '(unknown)') AS platform,
               COUNT(*) AS n
        FROM social_posts
        GROUP BY COALESCE(NULLIF(TRIM(platform), ''), '(unknown)')
        ORDER BY n DESC
        """
    )
    if not result.get("ok"):
        return []
    return list(result.get("rows") or [])


@st.cache_data(ttl=300, show_spinner=False)
def load_country_rows(limit: int = 10) -> list[dict[str, Any]]:
    result = mysql_service.run_sql(
        f"""
        SELECT COALESCE(NULLIF(TRIM(country), ''), '(unknown)') AS country,
               COUNT(*) AS n
        FROM social_posts
        GROUP BY COALESCE(NULLIF(TRIM(country), ''), '(unknown)')
        ORDER BY n DESC
        LIMIT {int(limit)}
        """
    )
    if not result.get("ok"):
        return []
    return list(result.get("rows") or [])


@st.cache_data(ttl=300, show_spinner=False)
def load_agreement_pct(model_version: str = "best_v1") -> float | None:
    result = mysql_service.prediction_agreement(model_version)
    if not result.get("ok"):
        return None
    rows = result.get("rows") or []
    if not rows:
        return None
    try:
        return float(rows[0].get("agreement_pct"))
    except (TypeError, ValueError):
        return None


@st.cache_data(ttl=300, show_spinner=False)
def load_val_accuracy() -> float | None:
    metrics = get_best_metrics()
    if not metrics.get("ok"):
        return None
    data = metrics.get("data") or {}
    if isinstance(data, dict):
        for key in ("val_accuracy", "accuracy"):
            if key in data:
                try:
                    return float(data[key])
                except (TypeError, ValueError):
                    pass
        nested = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
        for key in ("val_accuracy", "accuracy"):
            if key in nested:
                try:
                    return float(nested[key])
                except (TypeError, ValueError):
                    pass
    comparison = get_model_comparison()
    if comparison.get("ok"):
        block = comparison.get("data") or {}
        models = block.get("models") or {}
        winner = block.get("best_model_version") or block.get("best_source_version")
        if winner and isinstance(models.get(winner), dict):
            try:
                return float(models[winner].get("val_accuracy"))
            except (TypeError, ValueError):
                return None
    return None


def sentiment_kpis(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    counts: dict[str, int] = {}
    total = 0
    for row in rows:
        label = str(row.get("sentiment_group") or "Unknown")
        n = int(row.get("n") or 0)
        counts[label] = counts.get(label, 0) + n
        total += n
    pos = counts.get("Positive", 0)
    neg = counts.get("Negative", 0)
    return {
        "total": total,
        "positive": pos,
        "negative": neg,
        "neutral": counts.get("Neutral", 0),
        "positive_pct": round(100.0 * pos / total, 1) if total else 0.0,
        "negative_pct": round(100.0 * neg / total, 1) if total else 0.0,
    }


def rows_to_frame(rows: list[dict[str, Any]], *, index_col: str, value_col: str = "n") -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=[index_col, value_col])
    df = pd.DataFrame(rows)
    if index_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame(columns=[index_col, value_col])
    return df[[index_col, value_col]].copy()
