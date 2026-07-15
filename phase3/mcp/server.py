"""Unified MCP Server — MySQL | ML | Email | WhatsApp | FRED enrichment.

Run (stdio):
    python -m phase3.mcp.server

Cursor / Claude Desktop can point at this module as an MCP server.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from contextlib import redirect_stdout
from functools import wraps
from io import StringIO
from pathlib import Path

# Keep stdio JSON-RPC clean: no library chatter on stdout; avoid loky under MCP.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.WARNING)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp.server.fastmcp import FastMCP

from phase3.services import email_service, fred_service, ml_service, mysql_service, whatsapp_service


def _stdio_safe(fn):
    """Run tool body without writing to stdout (protects MCP framing)."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        sink = StringIO()
        with redirect_stdout(sink):
            return fn(*args, **kwargs)

    return wrapper

mcp = FastMCP(
    "unified-sentiment-mcp",
    instructions=(
        "Unified MCP for Social Media Sentiment & Brand Intelligence. "
        "Tools: MySQL analytics, ML metrics/prediction, email/WhatsApp alerts "
        "(draft by default), and optional FRED economic enrichment."
    ),
)


def _dumps(payload: dict) -> str:
    return json.dumps(payload, indent=2, default=str)


@mcp.tool()
@_stdio_safe
def mysql_query(sql: str, allow_write: bool = False) -> str:
    """Run SQL against the sentiment MySQL database (read-only by default)."""
    return _dumps(mysql_service.run_sql(sql, allow_write=allow_write))


@mcp.tool()
@_stdio_safe
def mysql_sentiment_summary() -> str:
    """Count social_posts by sentiment_group."""
    return _dumps(mysql_service.sentiment_summary())


@mcp.tool()
@_stdio_safe
def mysql_prediction_agreement(model_version: str = "best_v1") -> str:
    """Agreement between model_predictions and true sentiment_group."""
    return _dumps(mysql_service.prediction_agreement(model_version))


@mcp.tool()
@_stdio_safe
def mysql_latest_metrics(model_version: str = "best_v1") -> str:
    """Return model_metrics rows for a model_version."""
    return _dumps(mysql_service.latest_metrics(model_version))


@mcp.tool()
@_stdio_safe
def ml_list_artifacts() -> str:
    """List Phase 2 artifact files."""
    return _dumps(ml_service.list_artifacts())


@mcp.tool()
@_stdio_safe
def ml_get_comparison() -> str:
    """Return Phase 2 model comparison JSON (validation winner)."""
    return _dumps(ml_service.get_model_comparison())


@mcp.tool()
@_stdio_safe
def ml_get_best_metrics() -> str:
    """Return metrics for the best Phase 2 model."""
    return _dumps(ml_service.get_best_metrics())


@mcp.tool()
@_stdio_safe
def ml_predict(texts_json: str, model_version: str = "best_v1") -> str:
    """Predict sentiment_group for a JSON list of texts, e.g. '[\"I love this\"]'."""
    try:
        texts = json.loads(texts_json)
        if isinstance(texts, str):
            texts = [texts]
        if not isinstance(texts, list):
            return _dumps({"ok": False, "error": "texts_json must be a JSON list of strings"})
    except json.JSONDecodeError as exc:
        return _dumps({"ok": False, "error": f"Invalid JSON: {exc}"})
    return _dumps(ml_service.predict_texts([str(t) for t in texts], model_version))


@mcp.tool()
@_stdio_safe
def email_draft(to: str, subject: str, body: str) -> str:
    """Draft an email alert and log it (never sends)."""
    return _dumps(email_service.draft_email(to, subject, body))


@mcp.tool()
@_stdio_safe
def email_send(to: str, subject: str, body: str) -> str:
    """Send email only if ALERTS_ENABLED=true; otherwise draft/log."""
    return _dumps(email_service.send_email(to, subject, body))


@mcp.tool()
@_stdio_safe
def whatsapp_draft(to: str, body: str) -> str:
    """Draft a WhatsApp alert and log it (never sends)."""
    return _dumps(whatsapp_service.draft_whatsapp(to, body))


@mcp.tool()
@_stdio_safe
def whatsapp_send(to: str, body: str) -> str:
    """Send WhatsApp only if ALERTS_ENABLED=true; otherwise draft/log."""
    return _dumps(whatsapp_service.send_whatsapp(to, body))


@mcp.tool()
@_stdio_safe
def fred_search(search_text: str, limit: int = 5) -> str:
    """Search FRED economic series (requires FRED_API_KEY)."""
    return _dumps(fred_service.fred_search(search_text, limit=limit))


@mcp.tool()
@_stdio_safe
def fred_get_series(series_id: str = "UNRATE", limit: int = 12) -> str:
    """Fetch recent FRED observations (e.g. UNRATE, CPIAUCSL, GDPC1)."""
    return _dumps(fred_service.fred_series_observations(series_id, limit=limit))


def main() -> None:
    # Preload model so the first CallToolRequest is not the cold path.
    try:
        ml_service._load_pipeline("best_v1")
    except Exception:  # noqa: BLE001
        pass
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
