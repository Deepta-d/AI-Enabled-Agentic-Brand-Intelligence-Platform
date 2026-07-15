"""LangChain tool wrappers over the same services as the Unified MCP."""

from __future__ import annotations

import json
from typing import Callable

from langchain_core.tools import tool

from phase3.services import email_service, fred_service, ml_service, mysql_service, whatsapp_service


def _j(payload: dict) -> str:
    return json.dumps(payload, indent=2, default=str)


@tool
def mysql_query(sql: str) -> str:
    """Run a read-only SQL query against the sentiment MySQL database."""
    return _j(mysql_service.run_sql(sql, allow_write=False))


@tool
def mysql_sentiment_summary() -> str:
    """Summarize social_posts counts by sentiment_group."""
    return _j(mysql_service.sentiment_summary())


@tool
def mysql_latest_metrics(model_version: str = "best_v1") -> str:
    """Fetch model_metrics for a model version (default best_v1)."""
    return _j(mysql_service.latest_metrics(model_version))


@tool
def mysql_prediction_agreement(model_version: str = "best_v1") -> str:
    """Compute agreement between predictions and true labels."""
    return _j(mysql_service.prediction_agreement(model_version))


@tool
def ml_get_comparison() -> str:
    """Get Phase 2 validation comparison and winner."""
    return _j(ml_service.get_model_comparison())


@tool
def ml_get_best_metrics() -> str:
    """Get best model test metrics from Phase 2 artifacts."""
    return _j(ml_service.get_best_metrics())


@tool
def ml_predict(
    text: str = "",
    texts_json: str = "",
    model_version: str = "best_v1",
) -> str:
    """Predict sentiment_group for one text or a JSON list of texts (texts_json)."""
    texts: list[str] = []
    if texts_json and texts_json.strip():
        try:
            parsed = json.loads(texts_json)
            if isinstance(parsed, str):
                texts = [parsed]
            elif isinstance(parsed, list):
                texts = [str(t) for t in parsed]
            else:
                return _j({"ok": False, "error": "texts_json must be a JSON list of strings"})
        except json.JSONDecodeError as exc:
            return _j({"ok": False, "error": f"Invalid texts_json: {exc}"})
    elif text and text.strip():
        texts = [text.strip()]
    else:
        return _j({"ok": False, "error": "Provide text=... or texts_json='[\"...\"]'"})
    return _j(ml_service.predict_texts(texts, model_version))


@tool
def email_draft_alert(to: str, subject: str, body: str) -> str:
    """Draft an email alert (logged; does not send)."""
    return _j(email_service.draft_email(to, subject, body))


@tool
def email_send_alert(to: str, subject: str, body: str) -> str:
    """Send email if ALERTS_ENABLED=true, else draft only."""
    return _j(email_service.send_email(to, subject, body))


@tool
def whatsapp_draft_alert(to: str, body: str) -> str:
    """Draft a WhatsApp alert (logged; does not send)."""
    return _j(whatsapp_service.draft_whatsapp(to, body))


@tool
def whatsapp_send_alert(to: str, body: str) -> str:
    """Send WhatsApp if ALERTS_ENABLED=true, else draft only."""
    return _j(whatsapp_service.send_whatsapp(to, body))


@tool
def fred_get_series(series_id: str = "UNRATE", limit: int = 12) -> str:
    """Fetch FRED economic series observations for brand/macro context."""
    return _j(fred_service.fred_series_observations(series_id, limit=limit))


@tool
def fred_search_series(search_text: str) -> str:
    """Search FRED series by keyword."""
    return _j(fred_service.fred_search(search_text))


SQL_TOOLS = [
    mysql_query,
    mysql_sentiment_summary,
    mysql_latest_metrics,
    mysql_prediction_agreement,
]
ML_TOOLS = [ml_get_comparison, ml_get_best_metrics, ml_predict, fred_get_series, fred_search_series]
EMAIL_TOOLS = [email_draft_alert, email_send_alert]
WHATSAPP_TOOLS = [whatsapp_draft_alert, whatsapp_send_alert]


def all_tools() -> list[Callable]:
    return SQL_TOOLS + ML_TOOLS + EMAIL_TOOLS + WHATSAPP_TOOLS
