"""Load MySQL connection settings from environment / .env."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv

# Project root: .../Social Media Sentiment & Brand Intelligence Platform
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

RAW_CSV_PATH = PROJECT_ROOT / "dataset" / "sentimentdataset.csv"
CLEANED_CSV_PATH = PROJECT_ROOT / "dataset" / "cleaned_sentimentdataset.csv"
SCHEMA_SQL_PATH = PROJECT_ROOT / "phase1" / "sql" / "01_schema.sql"
INDEXES_SQL_PATH = PROJECT_ROOT / "phase1" / "sql" / "02_indexes.sql"


def get_mysql_settings() -> dict[str, str | int]:
    return {
        "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD", ""),
        "database": os.getenv("MYSQL_DATABASE", "sentiment_brand_intel"),
    }


def _build_url(*, include_database: bool) -> str:
    """Build a SQLAlchemy URL with URL-encoded user/password.

    Passwords with @, :, /, #, etc. break unencoded URLs (host becomes mangled).
    """
    s = get_mysql_settings()
    user = quote_plus(str(s["user"]))
    password = quote_plus(str(s["password"]))
    host = s["host"]
    port = s["port"]
    if include_database:
        db = quote_plus(str(s["database"]))
        return f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}?charset=utf8mb4"
    return f"mysql+pymysql://{user}:{password}@{host}:{port}/?charset=utf8mb4"


def get_database_url() -> str:
    return _build_url(include_database=True)


def get_server_url() -> str:
    """URL without database name (for CREATE DATABASE)."""
    return _build_url(include_database=False)
