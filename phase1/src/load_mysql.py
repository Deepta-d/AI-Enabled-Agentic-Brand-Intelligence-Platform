"""Load cleaned social posts into MySQL (Phase 1).

Usage (from project root):
    python -m phase1.src.load_mysql
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError, ProgrammingError

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase1.src.clean import load_and_clean, save_cleaned_csv
from phase1.src.config import (
    CLEANED_CSV_PATH,
    INDEXES_SQL_PATH,
    RAW_CSV_PATH,
    SCHEMA_SQL_PATH,
    get_database_url,
    get_mysql_settings,
    get_server_url,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LOAD_COLUMNS = [
    "source_row_id",
    "text",
    "sentiment",
    "sentiment_group",
    "timestamp",
    "username",
    "platform",
    "hashtags",
    "retweets",
    "likes",
    "country",
    "year",
    "month",
    "day",
    "hour",
]


def _split_sql_statements(sql_text: str) -> list[str]:
    """Split a SQL script into statements, ignoring comment-only lines."""
    # Remove block comments
    cleaned = re.sub(r"/\*.*?\*/", "", sql_text, flags=re.DOTALL)
    statements: list[str] = []
    for part in cleaned.split(";"):
        lines = []
        for line in part.splitlines():
            stripped = line.strip()
            if stripped.startswith("--") or not stripped:
                continue
            lines.append(line)
        stmt = "\n".join(lines).strip()
        if stmt:
            statements.append(stmt)
    return statements


def _is_ignorable_duplicate(exc: Exception) -> bool:
    msg = str(getattr(exc, "orig", None) or exc)
    return "Duplicate key name" in msg or "already exists" in msg.lower()


def apply_sql_file(engine, path: Path, *, ignore_duplicate_index: bool = False) -> None:
    sql_text = path.read_text(encoding="utf-8")
    statements = _split_sql_statements(sql_text)
    for stmt in statements:
        # One transaction per statement so a duplicate-index error does not
        # abort the rest of the script (MySQL invalidates the whole txn).
        try:
            with engine.begin() as conn:
                conn.execute(text(stmt))
        except (ProgrammingError, OperationalError) as exc:
            if ignore_duplicate_index and _is_ignorable_duplicate(exc):
                logger.warning("Skipping (already exists): %s", stmt.splitlines()[0][:80])
                continue
            raise


def ensure_schema() -> None:
    settings = get_mysql_settings()
    logger.info(
        "Connecting to MySQL %s@%s:%s",
        settings["user"],
        settings["host"],
        settings["port"],
    )
    server_engine = create_engine(get_server_url(), pool_pre_ping=True)
    try:
        apply_sql_file(server_engine, SCHEMA_SQL_PATH)
        logger.info("Applied schema from %s", SCHEMA_SQL_PATH.name)
    finally:
        server_engine.dispose()

    db_engine = create_engine(get_database_url(), pool_pre_ping=True)
    try:
        apply_sql_file(db_engine, INDEXES_SQL_PATH, ignore_duplicate_index=True)
        logger.info("Applied indexes from %s", INDEXES_SQL_PATH.name)
    finally:
        db_engine.dispose()


def load_social_posts(df: pd.DataFrame) -> None:
    engine = create_engine(get_database_url(), pool_pre_ping=True)
    payload = df[LOAD_COLUMNS].copy()
    # SQLAlchemy / MySQL prefer native Python types for nullable ints
    for col in ("source_row_id", "retweets", "likes", "year", "month", "day", "hour"):
        if col in payload.columns:
            payload[col] = payload[col].astype(object).where(payload[col].notna(), None)

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM social_posts"))
        payload.to_sql(
            "social_posts",
            con=conn,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=200,
        )
        total = conn.execute(text("SELECT COUNT(*) FROM social_posts")).scalar()
        logger.info("Loaded %s rows into social_posts", total)

        rows = conn.execute(
            text(
                "SELECT platform, COUNT(*) AS n "
                "FROM social_posts GROUP BY platform ORDER BY n DESC"
            )
        ).mappings().all()
        logger.info("Validation - posts by platform:")
        for row in rows:
            logger.info("  %s: %s", row["platform"], row["n"])

        groups = conn.execute(
            text(
                "SELECT sentiment_group, COUNT(*) AS n "
                "FROM social_posts GROUP BY sentiment_group ORDER BY n DESC"
            )
        ).mappings().all()
        logger.info("Validation - posts by sentiment_group:")
        for row in groups:
            logger.info("  %s: %s", row["sentiment_group"], row["n"])

    engine.dispose()


def main() -> int:
    if not RAW_CSV_PATH.exists():
        logger.error("Raw CSV not found: %s", RAW_CSV_PATH)
        return 1

    try:
        ensure_schema()
    except OperationalError as exc:
        msg = str(getattr(exc, "orig", None) or exc)
        if "Can't connect" in msg or "2003" in msg or "1045" in msg:
            logger.error(
                "Could not connect to MySQL. Check that MySQL is running and "
                ".env credentials are correct. Details: %s",
                exc,
            )
        else:
            logger.error("MySQL error while applying schema/indexes: %s", exc)
        return 1

    cleaned = load_and_clean(RAW_CSV_PATH)
    save_cleaned_csv(cleaned, CLEANED_CSV_PATH)
    logger.info("Wrote cleaned CSV -> %s (%s rows)", CLEANED_CSV_PATH, len(cleaned))

    try:
        load_social_posts(cleaned)
    except OperationalError as exc:
        logger.error("MySQL error while loading data: %s", exc)
        return 1

    logger.info("Phase 1 load complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
