"""MySQL query helpers for Unified MCP / SQL Agent."""

from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text

from phase3.config import get_database_url

logger = logging.getLogger(__name__)

_READ_ONLY = re.compile(
    r"^\s*(SELECT|SHOW|DESCRIBE|DESC|EXPLAIN)\b",
    re.IGNORECASE | re.DOTALL,
)

# Short codes only — UI formatters never show raw driver/SQL dumps.
_ERR_EMPTY = "Empty SQL"
_ERR_WRITE_BLOCKED = "Only SELECT/SHOW/DESCRIBE/EXPLAIN allowed unless allow_write=true"
_ERR_QUERY_FAILED = "query_failed"


def run_sql(sql: str, *, allow_write: bool = False, limit: int = 200) -> dict[str, Any]:
    """Execute SQL against sentiment_brand_intel. Read-only by default."""
    sql = sql.strip().rstrip(";")
    if not sql:
        return {"ok": False, "error": _ERR_EMPTY}
    if not allow_write and not _READ_ONLY.match(sql):
        return {
            "ok": False,
            "error": _ERR_WRITE_BLOCKED,
        }

    engine = create_engine(get_database_url(), pool_pre_ping=True)
    try:
        if _READ_ONLY.match(sql):
            with engine.connect() as conn:
                df = pd.read_sql(text(sql), conn)
                if len(df) > limit:
                    df = df.head(limit)
                return {
                    "ok": True,
                    "rows": df.to_dict(orient="records"),
                    "row_count": len(df),
                    "columns": list(df.columns),
                }
        with engine.begin() as conn:
            result = conn.execute(text(sql))
            return {"ok": True, "rowcount": result.rowcount}
    except Exception as exc:  # noqa: BLE001 — details stay in logs only
        logger.exception("MySQL run_sql failed: %s", exc)
        return {"ok": False, "error": _ERR_QUERY_FAILED}
    finally:
        engine.dispose()


def sentiment_summary() -> dict[str, Any]:
    return run_sql(
        """
        SELECT sentiment_group, COUNT(*) AS n
        FROM social_posts
        GROUP BY sentiment_group
        ORDER BY n DESC
        """
    )


def prediction_agreement(model_version: str = "best_v1") -> dict[str, Any]:
    return run_sql(
        f"""
        SELECT
          COUNT(*) AS n,
          SUM(CASE WHEN mp.predicted_sentiment = sp.sentiment_group THEN 1 ELSE 0 END) AS matches,
          ROUND(
            100.0 * SUM(CASE WHEN mp.predicted_sentiment = sp.sentiment_group THEN 1 ELSE 0 END)
            / COUNT(*), 2
          ) AS agreement_pct
        FROM model_predictions mp
        JOIN social_posts sp ON sp.id = mp.post_id
        WHERE mp.model_version = '{model_version.replace("'", "")}'
        """
    )


def latest_metrics(model_version: str = "best_v1") -> dict[str, Any]:
    return run_sql(
        f"""
        SELECT metric_name, metric_value
        FROM model_metrics
        WHERE model_version = '{model_version.replace("'", "")}'
        ORDER BY metric_name
        """
    )
